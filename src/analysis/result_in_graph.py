import matplotlib.pyplot as plt
import pandas as pd

# 모델 별 불일치율 결과
results = {
    'BERT': 89.610,  # BERT 모델의 일치율
    'RF': 74.725,  # RF 모델의 일치율
    'RoBERTa': 89.810, # RoBERTa 모델의 일치율
    'BERT+RF': 74.825, # BERT+RF 가중평균 모델1
    'BERT+RF+RoBERTa': 74.925, #BERT+RF+RoBERTa 가중평균 모델2
}

# 결과를 DataFrame으로 변환
results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Match Rate'])

# Bar 그래프를 그립니다.
plt.figure(figsize=(10, 5))
plt.bar(results_df['Model'], results_df['Match Rate'], color=['blue', 'green', 'red'])
plt.xlabel('Model')
plt.ylabel('Match Rate (%)')
plt.title('Match Rate by Model')

# 이미지로 저장
plt.savefig("model_match_rate.png", dpi=300, bbox_inches='tight')

plt.show()
