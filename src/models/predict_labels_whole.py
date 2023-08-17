import pandas as pd
from bs4 import BeautifulSoup
import joblib
import torch
from transformers import BertTokenizer, BertForSequenceClassification, RobertaTokenizer, \
    RobertaForSequenceClassification
from sklearn.metrics import accuracy_score

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

# 각 모델의 가중 평균을 계산하고 가장 낮은 불일치율을 가진 조합 찾기
best_mismatch_rate = 100
best_weights = None

# 가중치 조합 생성
for bert_weight in range(0, 1001):
    for roberta_weight in range(0, 1001 - bert_weight):
        rf_weight = 1001 - bert_weight - roberta_weight

        bert_weight /= 1000
        roberta_weight /= 1000
        rf_weight /= 1000

        # 가중 평균 확률 계산
        df['Infer_Prob'] = bert_weight * df['Infer_Prob_BERT'] + roberta_weight * df['Infer_Prob_RoBERTa'] + rf_weight * \
                           df['Infer_Prob_RF']

        # 최종 라벨 결정
        df['Infer_Label'] = df['Infer_Prob'].apply(lambda x: 1 if x > 0.5 else 0)

        # 불일치율 계산
        mismatch_rows = df[df['Label'] != df['Infer_Label']]
        mismatch_rate = len(mismatch_rows) / len(df) * 100

        # 가장 좋은 불일치율 갱신
        if mismatch_rate < best_mismatch_rate:
            best_mismatch_rate = mismatch_rate
            best_weights = (bert_weight, roberta_weight, rf_weight)

        # 불일치율이 같으면 가중치 출력
        elif mismatch_rate == best_mismatch_rate:
            print(f"BERT: {bert_weight}, RoBERTa: {roberta_weight}, RF: {rf_weight}, 불일치율: {mismatch_rate}%")

print(
    f"최적의 가중치 조합 - BERT: {best_weights[0]}, RoBERTa: {best_weights[1]}, RF: {best_weights[2]}, 불일치율: {best_mismatch_rate}%")

# ... 이전 코드 (변수 및 모델 불러오기, 데이터 준비 등) ...

# 각 모델의 예측 확률을 계산하고 정답율을 계산
df['Infer_Label_RF'] = df['Infer_Prob_RF'].apply(lambda x: 1 if x > 0.5 else 0)
df['Infer_Label_BERT'] = df['Infer_Prob_BERT'].apply(lambda x: 1 if x > 0.5 else 0)
df['Infer_Label_RoBERTa'] = df['Infer_Prob_RoBERTa'].apply(lambda x: 1 if x > 0.5 else 0)

accuracy_rf = accuracy_score(df['Label'], df['Infer_Label_RF'])
accuracy_bert = accuracy_score(df['Label'], df['Infer_Label_BERT'])
accuracy_roberta = accuracy_score(df['Label'], df['Infer_Label_RoBERTa'])

print(f"RF 모델 정답율: {accuracy_rf * 100}%")
print(f"BERT 모델 정답율: {accuracy_bert * 100}%")
print(f"RoBERTa 모델 정답율: {accuracy_roberta * 100}%")

# 가중 평균 모델 1 - BERT, RF 모델만 합치기
df['Infer_Prob_avg1'] = 0.5 * df['Infer_Prob_BERT'] + 0.5 * df['Infer_Prob_RF']
df['Infer_Label_avg1'] = df['Infer_Prob_avg1'].apply(lambda x: 1 if x > 0.5 else 0)
accuracy_avg1 = accuracy_score(df['Label'], df['Infer_Label_avg1'])
print(f"가중 평균 모델 1 정답율: {accuracy_avg1 * 100}%")

# 가중 평균 모델 2 - BERT, RoBERTa, RF 모델 합치기
df['Infer_Prob_avg2'] = (1 / 3) * df['Infer_Prob_BERT'] + (1 / 3) * df['Infer_Prob_RoBERTa'] + (1 / 3) * df[
    'Infer_Prob_RF']
df['Infer_Label_avg2'] = df['Infer_Prob_avg2'].apply(lambda x: 1 if x > 0.5 else 0)
accuracy_avg2 = accuracy_score(df['Label'], df['Infer_Label_avg2'])
print(f"가중 평균 모델 2 정답율: {accuracy_avg2 * 100}%")

# 가중 평균 모델 3 - 모델 1과 2를 합치기
df['Infer_Prob_avg3'] = 0.5 * df['Infer_Prob_avg1'] + 0.5 * df['Infer_Prob_avg2']
df['Infer_Label_avg3'] = df['Infer_Prob_avg3'].apply(lambda x: 1 if x > 0.5 else 0)
accuracy_avg3 = accuracy_score(df['Label'], df['Infer_Label_avg3'])
print(f"가중 평균 모델 3 정답율: {accuracy_avg3 * 100}%")

# 모든 모델들을 동일하게 가중치를 둔 후, 합치기
df['Infer_Prob_all'] = (1 / 5) * df['Infer_Prob_BERT'] + (1 / 5) * df['Infer_Prob_RoBERTa'] + (1 / 5) * df[
    'Infer_Prob_RF'] + (1 / 5) * df['Infer_Prob_avg1'] + (1 / 5) * df['Infer_Prob_avg2']
df['Infer_Label_all'] = df['Infer_Prob_all'].apply(lambda x: 1 if x > 0.5 else 0)
accuracy_all = accuracy_score(df['Label'], df['Infer_Label_all'])
print(f"모든 모델 합치기 정답율: {accuracy_all * 100}%")
