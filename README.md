# NSMC 한국어 영화 리뷰 감성 분석 (Deep Learning NLP Project)

네이버 영화 리뷰 데이터셋(NSMC)을 활용하여 한국어 형태소 기반 어휘 구축부터 통계적 기준선(Naive Bayes) 및 딥러닝(Bi-LSTM) 모델 구현, 구어체 노이즈 정규화 요인 비교 실험, 길이별 성능 및 오류 분석까지 수행한 프로젝트입니다.

## 1. 데이터 출처 및 전처리
- **출처**: [NSMC (Naver Sentiment Movie Corpus)](https://github.com/e9t/nsmc) (`ratings_train.txt`, `ratings_test.txt`)
- **전처리**: 결측치 및 공백 문장 제거, 중복 리뷰 제거(Train/Test 간 데이터 누수 차단), Train 데이터에서 15% 층화 추출(Stratified Split)하여 Validation 분할
  - Train: 124,254개 / Valid: 21,928개 / Test: 48,361개

## 2. 설치 및 실행 방법

### 환경 설치
```bash
pip install -r requirements.txt
