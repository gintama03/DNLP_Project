# NSMC 한국어 영화 리뷰 감성 분석 (Deep Learning NLP Project)

네이버 영화 리뷰 데이터셋(NSMC)을 활용하여 한국어 형태소 기반 어휘 구축부터 통계적 기준선(Naive Bayes) 및 딥러닝(Bi-LSTM) 모델 구현, 구어체 노이즈 정규화 요인 비교 실험, 길이별 성능 및 오류 분석까지 수행한 프로젝트입니다.

## 1. 데이터 출처 및 전처리
- **출처**: [NSMC (Naver Sentiment Movie Corpus)](https://github.com/e9t/nsmc) (`ratings_train.txt`, `ratings_test.txt`)
- **전처리**: 결측치 및 공백 문장 제거, 중복 리뷰 제거(Train/Test 간 데이터 누수 차단), Train 데이터에서 15% 층화 추출(Stratified Split)하여 Validation 분할
  - **Train**: 124,254개 / **Valid**: 21,928개 / **Test**: 48,361개

## 2. 설치 및 실행 방법

### 환경 설치
    pip install -r requirements.txt

### 단계별 실행 명령어
    # 1. 데이터 다운로드, 정제 및 분할
    python src/preprocess.py

    # 2. Train 데이터 기반 어휘 구축 (outputs/vocab.json 생성)
    python src/build_vocab.py

    # 3. Model A (Baseline: Multinomial Naive Bayes) 학습
    python src/train_baseline.py

    # 4. Model B (Bi-LSTM, 정규화 미적용) 및 Model C (Bi-LSTM, 구어체 정규화 적용) 학습
    python src/train_dl.py --model_type B
    python src/train_dl.py --model_type C

    # 5. Test 최종 평가, 길이별 성능 분석 및 오답 10개 추출
    python src/evaluate.py

    # 6. 저장된 모델과 vocab.json을 이용한 새 리뷰 5개 추론 시연
    python src/predict.py

## 3. 주요 실험 결과 (Test Set 기준)

| 모델 | 비교 요인 (Factor) | Test Accuracy (%) | Test Macro-F1 (%) |
| :--- | :--- | :---: | :---: |
| **Model A (Naive Bayes Baseline)** | BoW + MultinomialNB | 83.47 | 83.47 |
| **Model B (Bi-LSTM)** | Kiwi 형태소 (정규화 X) | 86.23 | 86.22 |
| **Model C (Bi-LSTM + Norm)** | Kiwi 형태소 + 구어체 반복문자 정규화 | **86.43** | **86.43** |

## 4. 산출물 안내
- `outputs/vocab.json`: Train 데이터만으로 구축한 어휘 사전 (크기: 16,539, Valid UNK 비율: 1.78%)
- `outputs/experiment_results.csv`: A / B / C 모델 Test 비교 결과표
- `outputs/length_analysis.csv`: 리뷰 토큰 길이 구간별 성능 분석 결과
- `outputs/error_analysis.md`: Test 오답 대표 사례 10개 심층 분석
- `outputs/demo_predictions.txt`: 직접 작성한 새로운 입력 5개 추론 결과
