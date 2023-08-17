import pandas as pd
from bs4 import BeautifulSoup

# CSV 파일 읽기
df = pd.read_csv('HTML_NL.csv')

# 결과를 저장할 새로운 열 생성
df['Natural_Language'] = ''

# 각 HTML 문자열에 대해 파싱하고, script 태그를 제외한 나머지 태그의 텍스트를 추출
for index, row in df.iterrows():
    if pd.notnull(row['HTML']) and isinstance(row['HTML'], str):  # 추가된 코드
        soup = BeautifulSoup(row['HTML'], 'html.parser')

        # script 태그 제거
        for script in soup("script"):
            script.decompose()

        # body 태그 내의 텍스트만 추출
        body = soup.body
        if body is not None:
            df.at[index, 'Natural_Language'] = body.get_text(separator=' ', strip=True)

# CSV 파일로 저장
df.to_csv('HTML_NL.csv', index=False)
