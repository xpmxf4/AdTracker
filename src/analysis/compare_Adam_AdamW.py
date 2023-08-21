import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.optim import Adam
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 읽기
df1 = pd.read_csv('../../data/processed/keyword_url_label_ori.csv')
df2 = pd.read_csv('../../data/processed/HTML_NL_ori.csv')

# 두 데이터프레임 병합
df = pd.concat([df1['Label'], df2['Natural_Language']], axis=1)

# 결측값이 있는 행 제거
df = df.dropna()

sentences = df['Natural_Language'].values
# 레이블 값 2를 1로 대체
df['Label'] = df['Label'].replace(2, 1)
labels = df['Label'].values

# BERT 토크나이저
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')


# 데이터셋 정의
class MyDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
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
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


# DataLoader 설정
dataset = MyDataset(sentences, labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=32)

model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)
model = model.to(device)

# Adam 옵티마이저와 AdamW 옵티마이저 설정
optimizers = {
    'Adam': Adam(model.parameters(), lr=1e-5),
    'AdamW': AdamW(model.parameters(), lr=1e-5)
}

losses = {
    'Adam': [],
    'AdamW': []
}

# 옵티마이저별 학습
for opt_name, optimizer in optimizers.items():
    # 모델 초기 상태 저장
    original_state = model.state_dict().copy()
    model.train()

    for epoch in range(5):
        epoch_loss = 0.0

        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

        # 에포크의 평균 손실 저장
        losses[opt_name].append(epoch_loss / len(dataloader))
        print(f'Optimizer: {opt_name}, Epoch {epoch + 1}, Loss {epoch_loss / len(dataloader)}')

    # 모델을 초기 상태로 복원
    model.load_state_dict(original_state)

# 손실 그래프를 통한 비교
plt.plot(losses['Adam'], label='Adam')
plt.plot(losses['AdamW'], label='AdamW')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Optimizer Comparison')
plt.show()
