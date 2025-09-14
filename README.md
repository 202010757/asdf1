# 임신 성공 예측 프로젝트

이 프로젝트는 임신 성공 여부를 예측하는 머신러닝 모델을 구현합니다. 데이터 전처리, 하이퍼파라미터 튜닝, 교차 검증, 예측 결과 저장까지 전 과정을 자동화합니다.

##  파일 구조

- `pregnancy_success_prediction.py` : 전체 예측 파이프라인 코드
- `train.csv` : 학습 데이터
- `test.csv` : 테스트 데이터
- `sample_submission.csv` : 제출 예시 파일
- `submission.csv` : 예측 결과 파일

##  실행 방법

1. 필요한 CSV 파일을 프로젝트 폴더에 넣어주세요.
2. 아래 명령어로 필요한 라이브러리를 설치합니다.
	```bash
	pip install pandas scikit-learn numpy
	```
3. 예측 코드 실행:
	```bash
	python pregnancy_success_prediction.py
	```
4. `submission.csv` 파일이 생성됩니다.

##  코드 주요 기능

- 결측치 자동 처리
- 범주형 변수 라벨 인코딩
- 특성 스케일링
- Stratified K-Fold 교차 검증
- RandomForest 하이퍼파라미터 튜닝
- ROC-AUC 기반 성능 평가
- 예측 결과 저장

##  참고

* 데이터 컬럼명은 실제 파일에 맞게 수정해주세요.
* 추가적인 모델(GBM 등)도 쉽게 적용 가능합니다.

---
문의 및 개선 제안은 언제든 환영합니다
# asdf1