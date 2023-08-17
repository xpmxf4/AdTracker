import pandas as pd
from bs4 import BeautifulSoup
import joblib

# 모델과 벡터라이저 불러오기
loaded = joblib.load('../../models/rf_model_and_vectorizer.pkl')
rf = loaded['model']
vectorizer = loaded['vectorizer']

# 라벨 데이터 불러오기
df = pd.read_csv('keyword_url_label.csv')

# HTML 데이터 불러오기
html_df = pd.read_csv('HTML_NL.csv')

# 결측값 처리
df.fillna('', inplace=True)
html_df.fillna('', inplace=True)

# HTML 데이터 가져오기
html_data = html_df['HTML']

# HTML 태그를 파싱하여 계층적 구조를 반영한 피처 생성
parsed_html_features = []
error_count = 0
for html in html_data:
    try:
        soup = BeautifulSoup(html, 'html.parser')
        tag_features = []
        for tag in soup.descendants:
            if tag.name is not None:
                tag_features.append(tag.name)
        parsed_html_features.append(' '.join(tag_features))
    except Exception as e:
        print(f"HTML 파싱 중 에러 발생: {e}")
        parsed_html_features.append('')
        error_count += 1

# 피처 벡터화
X = vectorizer.transform(parsed_html_features)

# 예측 수행
df['Infer_Label_RF'] = rf.predict(X)

# 레이블과 예측 레이블이 다른 행 찾기
mismatch_rows = df[df['Label'] != df['Infer_Label_RF']]

# 불일치율 계산 (에러가 발생한 행은 제외)
mismatch_rate = len(mismatch_rows) / (len(df) - error_count) * 100

print(f"불일치율: {mismatch_rate}%")

# 결과를 csv 파일에 저장
df.to_csv('keyword_url_label.csv', index=False)
