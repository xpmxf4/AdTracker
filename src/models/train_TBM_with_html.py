import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from bs4 import BeautifulSoup
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 데이터 불러오기
df1 = pd.read_csv('../../data/processed/keyword_url_label_ori.csv')
df2 = pd.read_csv('../../data/processed/HTML_NL_ori.csv')

# df1과 df2를 합치기
df = pd.concat([df1, df2], axis=1)

# 결측값 제거
df.dropna(inplace=True)

# HTML 데이터만 추출
html_data = df['HTML']

# HTML 태그를 파싱하여 계층적 구조를 반영한 피처 생성
parsed_html_features = []
for html in html_data:
    soup = BeautifulSoup(html, 'html.parser')
    tag_features = []
    for tag in soup.descendants:
        if tag.name is not None:
            tag_features.append(tag.name)
    parsed_html_features.append(' '.join(tag_features))

# 피처 벡터화
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(parsed_html_features)

# 레이블 데이터 추출
Y = df['Label']

# 학습 데이터와 테스트 데이터 분리
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 랜덤 포레스트 모델 학습
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, Y_train)

# 모델과 vectorizer 저장
joblib.dump({'model': rf, 'vectorizer': vectorizer}, '../../models/rf_model_and_vectorizer.pkl')

# 테스트 데이터에 대한 예측 수행
Y_pred = rf.predict(X_test)

# 성능 평가
print("Accuracy: ", accuracy_score(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))
