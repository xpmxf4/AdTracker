import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 읽기
df1 = pd.read_csv('keyword_url_label.csv')
df2 = pd.read_csv('HTML_NL.csv')
# df1 = pd.read_csv('result_csvs/keyword_url_label_ori.csv')
# df2 = pd.read_csv('result_csvs/HTML_NL_ori.csv')

# 두 데이터프레임 병합
df = pd.concat([df1, df2['Natural_Language']], axis=1)

# BERT 토크나이저
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')


# 데이터셋 정의
class MyDataset(Dataset):
    def __init__(self, sentences, tokenizer):
        self.sentences = sentences
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }


# 데이터셋 생성
dataset = MyDataset(df['Natural_Language'].values, tokenizer)

# 데이터로더 설정
dataloader = DataLoader(dataset, batch_size=32)

# BERT 모델 로드
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)
model.load_state_dict(torch.load('../../models/bert_ver3.pt'))
model = model.to(device)
model.eval()

# 추론 및 결과 기록
df['Infer_Label_BERT'] = None
problematic_rows = []
for i, row in df.iterrows():
    try:
        inputs = tokenizer(row['Natural_Language'], return_tensors='pt', padding='longest', truncation=True)
        inputs = {key: inputs[key].to(device) for key in inputs}
        outputs = model(**inputs)
        _, predicted = torch.max(outputs.logits, 1)
        df.loc[i, 'Infer_Label_BERT'] = predicted.item()
    except Exception as e:
        problematic_rows.append(i)

# 문제가 있는 행 출력
print("Problematic Rows:")
for row_index in problematic_rows:
    print(row_index)

# 오차 비율 계산
total_rows = len(df)
mismatch_rows = df[(df['Label'] != df['Infer_Label_BERT']) & (~df.index.isin(problematic_rows))]
mismatch_ratio = len(mismatch_rows) / total_rows

# 결과 출력
print(f"Total Rows: {total_rows}")
print(f"Mismatch Rows: {len(mismatch_rows)}")
print(f"Mismatch Ratio: {mismatch_ratio}")
