import pandas as pd

# 데이터 읽기
df = pd.read_csv('result_csvs/keyword_url_label.csv')

# 'label'과 'Infer_Label'이 일치하지 않는 행만 추출
df_mismatch = df[df['Label'] != df['Infer_Label']].copy()  # Avoid SettingWithCopyWarning

# 불일치하는 행의 원래 인덱스를 새로운 열로 추가
df_mismatch['Mismatched_Row_Index'] = df_mismatch.index

# 비율 계산
mismatch_ratio = len(df_mismatch) / len(df)

# 비율을 DataFrame에 추가
df_mismatch['Mismatch_Ratio'] = '{:.2f}%'.format(mismatch_ratio * 100)

# 결과물을 CSV 파일로 저장
df_mismatch.to_csv('mismatched_labels.csv', index=False)
