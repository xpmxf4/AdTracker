import pandas as pd
from bs4 import BeautifulSoup
import joblib
import torch
from transformers import BertTokenizer, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification

# 모델과 벡터라이저 불러오기
loaded = joblib.load('../../models/rf_model_and_vectorizer.pkl')
rf = loaded['model']
vectorizer = loaded['vectorizer']

# Device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BERT 모델 불러오기
bert_model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)
bert_model.load_state_dict(torch.load('../../models/bert_ver3.pt'))
bert_model = bert_model.to(device)
bert_model.eval()

# BERT 토크나이저
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# RoBERTa 모델 불러오기
roberta_model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
roberta_model.load_state_dict(torch.load('../../models/RoBERTa_ver2.pt'))
roberta_model = roberta_model.to(device)
roberta_model.eval()

# RoBERTa 토크나이저
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Data load
df = pd.read_csv('../../data/processed/keyword_url_label_ori.csv')
html_df = pd.read_csv('../../data/processed/HTML_NL_ori.csv')

# 결측값 처리
df.fillna('', inplace=True)
html_df.fillna('', inplace=True)

# HTML 데이터 가져오기
html_data = html_df['HTML']
natural_language_data = html_df['Natural_Language']

# HTML 파싱
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

# RF 예측 확률 계산
df['Infer_Prob_RF'] = rf.predict_proba(X)[:, 1]

# BERT 예측 확률 계산
infer_probs_bert = []
for text in natural_language_data:
    encoding = bert_tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        output = bert_model(**encoding)
        probabilities = torch.nn.functional.softmax(output[0], dim=1)
        infer_probs_bert.append(float(probabilities[0][1]))

df['Infer_Prob_BERT'] = infer_probs_bert

# RoBERTa 예측 확률 계산
infer_probs_roberta = []
for text in natural_language_data:
    encoding = roberta_tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        output = roberta_model(**encoding)
        probabilities = torch.nn.functional.softmax(output[0], dim=1)
        infer_probs_roberta.append(float(probabilities[0][1]))

df['Infer_Prob_RoBERTa'] = infer_probs_roberta

# 가중평균을 사용하여 최적 가중치 찾기
best_mismatch_rate = 100
best_weights = (0, 0, 0)

for bert_weight in range(1, 1001):
    for roberta_weight in range(1, 1001 - bert_weight):
        rf_weight = 1001 - bert_weight - roberta_weight
        bert_weight /= 1000
        roberta_weight /= 1000
        rf_weight /= 1000

        df['Infer_Prob'] = bert_weight * df['Infer_Prob_BERT'] + roberta_weight * df['Infer_Prob_RoBERTa'] + rf_weight * df['Infer_Prob_RF']

        df['Infer_Label'] = df['Infer_Prob'].apply(lambda x: 1 if x > 0.5 else 0)

        mismatch_rows = df[df['Label'] != df['Infer_Label']]
        mismatch_rate = len(mismatch_rows) / (len(df) - error_count) * 100

        if mismatch_rate < best_mismatch_rate:
            best_mismatch_rate = mismatch_rate
            best_weights = (bert_weight, roberta_weight, rf_weight)

print(f"Best weights --> BERT: {best_weights[0]}, RoBERTa: {best_weights[1]}, RF: {best_weights[2]}, 불일치율: {best_mismatch_rate}%")
