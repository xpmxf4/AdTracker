# **프로젝트 제목**

**기계학습을 통한 사이트 자동 차단 프로그램**

## **소개**

이 연구는 기계 학습 모델을 활용하여 광고 사이트를 효과적으로 차단하는 프로그램을 개발하는 것을 목표로 합니다. 기존의 광고 차단 프로그램은 브라우저의 주소창 도메인이나 광고 도메인과 URL 규칙을 바탕으로 작동하지만, 이러한 규칙을 공개하면 광고 사이트가 쉽게 규칙을 우회할 수 있습니다.
 
따라서 이 연구에서는 광고 사이트가 규칙을 우회하더라도 지속적으로 감지할 수 있는 새로운 광고 차단 방법을 제시합니다. 기계 학습 모델인 BERT를 사용하여 광고 사이트와 비광고 사이트를 효과적으로 분류하였으며, 결과적으로 기존의 규칙 기반 방식보다 더욱 높은 성능을 보였습니다.
 
<키워드 : 웹사이트 분류, HTML 태그, 자연어 데이터, 앙상블 학습, 랜덤 포레스트, BERT>

## **폴더 구조**

```yaml
.
├── data: 데이터 관련 폴더
│   ├── forTest: 테스트 데이터셋
│   ├── processed: 전처리된 데이터셋
│   ├── raw: 원본 데이터
│   └── results: 모델 예측 결과와 레이블 불일치 데이터
├── models: 훈련된 모델 파일들
├── performance: 모델 성능 및 수정되지 않은 행 정보
├── share: Networkx 라이브러리와 관련된 문서 및 예제
└── src: 소스 코드 폴더
    ├── analysis: 결과 분석 코드
    ├── models: 모델 학습 및 예측 코드
    └── utils: 유틸리티 코드

```

## **주요 기능**

1. **데이터 전처리**: HTML에서 자연어를 추출하며, raw 데이터에서 필요한 데이터셋을 생성합니다.
2. **모델 학습 및 예측**: Bert, RoBERTa, RandomForest 등의 모델을 활용하여 학습과 예측을 수행합니다.
3. **결과 분석**: 예측 결과를 시각적으로 분석합니다.

## **사용 방법**

1. 필요한 라이브러리와 종속성을 설치합니다.
2. https://drive.google.com/file/d/1kT4d62dQ_tt3oNwVo1wlsE_65UwtvP3Z/view?usp=sharing 에서 data.zip 을 프로젝트 최상단에서 다운 받습니다.
3. data.zip 을 압축해제 합니다.
4. https://drive.google.com/file/d/1u5FlBeYSyi4yRzJfMzV8n8fOcWmN2R1T/view?usp=sharing
에서 models.zip 을 프로젝트 최상단에 다운 받습니다.
5. models.zip 을 압축해제합니다.
6. **`src/utils`** 폴더에 있는 유틸리티 코드를 사용하여 데이터 전처리를 진행합니다.
7. **`src/models`**에서 모델을 훈련시키고 예측합니다.
8. 결과는 **`data/results`** 폴더에 저장됩니다.
9. **`src/analysis`**에서 결과 분석을 수행합니다.

## 의존성

```
bs4: 4.12.2
google_trends: 1.1
pytrends: 4.9.2
sklearn: 0.0
torch: 2.0.1
transformers: 4.30.2
urllib3: 1.26.15
joblib: 1.2.0
matplotlib: 3.7.2
numpy: 1.24.3
pandas: 2.0.3
requests: 2.30.0
```

## **라이선스**

이 프로젝트는 MIT 라이선스에 따라 라이선스되어 있습니다. 자세한 내용은 **[LICENSE.md](https://chat.openai.com/c/LICENSE.md)** 파일을 참조하세요.