import pandas as pd

# 기존 파일을 로드합니다.
df = pd.read_csv('../../url_data.csv')

# Keyword, URL, Label 열들만 포함하는 새로운 데이터프레임을 생성합니다.
df1 = df[['Keyword', 'URL', 'Label']].copy()  # 명시적으로 복사를 생성

# df1을 keyword_url_label.csv 파일로 저장합니다.
df1.to_csv('keyword_url_label.csv', index=False)

# HTML, Natural Language 열들만 포함하는 새로운 데이터프레임을 생성합니다.
df2 = df[['HTML', 'Natural_Language']].copy()  # 명시적으로 복사를 생성

# df2을 HTML_NL.csv 파일로 저장합니다.
df2.to_csv('HTML_NL.csv', index=False)
