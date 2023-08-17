import pandas as pd
from bs4 import BeautifulSoup
import joblib
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 모델과 벡터라이저 불러오기
loaded = joblib.load('../../models/rf_model_and_vectorizer.pkl')
rf = loaded['model']
vectorizer = loaded['vectorizer']

# BERT 모델 불러오기
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)
model.load_state_dict(torch.load('../../models/bert_ver3.pt'))
model = model.to(device)
model.eval()

# BERT 토크나이저
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# 라벨 데이터 불러오기
# df = pd.read_csv('keyword_url_label.csv')
df = pd.read_csv('../../data/processed/keyword_url_label_ori.csv')

# HTML 데이터 불러오기
# html_df = pd.read_csv('HTML_NL.csv')
html_df = pd.read_csv('../../data/processed/HTML_NL_ori.csv')

# 결측값 처리
df.fillna('', inplace=True)
html_df.fillna('', inplace=True)

# HTML 데이터 가져오기
html_data = html_df['HTML']
natural_language_data = html_df['Natural_Language']

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

# RF 모델로 예측 확률 계산
df['Infer_Prob_RF'] = rf.predict_proba(X)[:, 1]

# BERT 모델로 예측 확률 계산
infer_probs_bert = []
for text in natural_language_data:
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        output = model(**encoding)
        probabilities = torch.nn.functional.softmax(output[0], dim=1)
        infer_probs_bert.append(float(probabilities[0][1]))

df['Infer_Prob_BERT'] = infer_probs_bert

best_mismatch_rate = 100  # Initialize with the worst possible rate
best_weights = (0, 0)  # Initialize with dummy weights

# Iterate over all possible weight combinations
for bert_weight in range(1, 1001):
    rf_weight = 1001 - bert_weight
    bert_weight /= 1000
    rf_weight /= 1000

    # Calculate weighted average of probabilities
    df['Infer_Prob'] = bert_weight * df['Infer_Prob_BERT'] + rf_weight * df['Infer_Prob_RF']

    # Determine the final label based on the weighted average probability
    df['Infer_Label'] = df['Infer_Prob'].apply(lambda x: 1 if x > 0.5 else 0)

    # Calculate mismatch rate
    mismatch_rows = df[df['Label'] != df['Infer_Label']]
    mismatch_rate = len(mismatch_rows) / (len(df) - error_count) * 100

    # If this weight combination results in a better (lower) mismatch rate, update the best rate and weights
    if mismatch_rate < best_mismatch_rate:
        best_mismatch_rate = mismatch_rate
        best_weights = (bert_weight, rf_weight)

    # If the mismatch rate is same, print the weights
    elif mismatch_rate == best_mismatch_rate:
        print(f"BERT: {bert_weight}, RF: {rf_weight}, 불일치율: {mismatch_rate}%")

# Print the best weights and their corresponding mismatch rate
print(f"Best weights --> BERT: {best_weights[0]}, RF: {best_weights[1]}, 불일치율: {best_mismatch_rate}%")
