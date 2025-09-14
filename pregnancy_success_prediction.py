import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import numpy as np

# 데이터 파일 경로
train_path = "train.csv"
test_path = "test.csv"
sample_submission_path = "sample_submission.csv"

# 데이터 로딩 (메모리 최적화)
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# ID 컬럼 분리
train_ids = train_df["ID"]
test_ids = test_df["ID"]
train_df.drop(columns=["ID"], inplace=True)
test_df.drop(columns=["ID"], inplace=True)

# 타겟 변수 분리
X = train_df.drop(columns=["임신 성공 여부"])
y = train_df["임신 성공 여부"]

# 결측치 처리
for col in X.columns:
    if X[col].dtype == 'object':
        X[col].fillna("Unknown", inplace=True)
        test_df[col].fillna("Unknown", inplace=True)
    else:
        X[col].fillna(X[col].mean(), inplace=True)
        test_df[col].fillna(test_df[col].mean(), inplace=True)

# 모든 범주형 변수 인코딩 (훈련 데이터 기준으로 변환)
categorical_cols = X.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
    
    # 테스트 데이터 변환 (훈련 데이터에 없는 값은 -1로 처리)
    test_df[col] = test_df[col].apply(lambda s: le.transform([s])[0] if s in le.classes_ else -1)

# 특성 스케일링
scaler = StandardScaler()
X = scaler.fit_transform(X)
test_df = scaler.transform(test_df)

# Stratified K-Fold Cross Validation 적용
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test_df))

# 하이퍼파라미터 튜닝
rf_params = {
    'n_estimators': [300, 500, 700],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}

best_params_rf = {}
best_score_rf = 0

for train_idx, valid_idx in kf.split(X, y):
    X_train, X_valid = X[train_idx], X[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    rf_model = RandomForestClassifier(random_state=42)
    rf_search = RandomizedSearchCV(rf_model, rf_params, n_iter=10, scoring='roc_auc', cv=3, random_state=42, n_jobs=-1)
    rf_search.fit(X_train, y_train)
    
    if rf_search.best_score_ > best_score_rf:
        best_score_rf = rf_search.best_score_
        best_params_rf = rf_search.best_params_
    
    rf_model = RandomForestClassifier(**rf_search.best_params_, random_state=42)
    rf_model.fit(X_train, y_train)
    
    y_pred_rf = rf_model.predict_proba(X_valid)[:, 1]
    oof_preds[valid_idx] = y_pred_rf
    
    # 테스트 데이터 예측 평균
    test_preds += rf_model.predict_proba(test_df)[:, 1] / kf.n_splits

# 검증 데이터 평가
roc_auc = roc_auc_score(y, oof_preds)
print(f"Best RandomForest Params: {best_params_rf}")
print(f"Optimized ROC-AUC Score: {roc_auc:.4f}")

# 결과 저장
test_submission = pd.DataFrame({"ID": test_ids, "probability": test_preds})
assert len(test_submission) == len(test_df), "Error: submission.csv의 행 개수가 test.csv와 다릅니다"
test_submission.to_csv("submission.csv", index=False)
print(" Submission file saved successfully")
