import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

# =============================
# 설정
# =============================
model_sources = [
    ('small-117M-k40', ['train', 'valid', 'test']),
    ('medium-345M-k40', ['train', 'valid', 'test']),
    ('large-762M-k40', ['train', 'valid', 'test']),
    ('xl-1542M-k40', ['train', 'valid', 'test'])
]
data_dir = './data'
out_dir = './data/merged_gpt2_k40'
os.makedirs(out_dir, exist_ok=True)

# =============================
# JSONL 파일 로딩 함수
# =============================
def load_jsonl(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line)['text'] for line in f]

# =============================
# 병합
# =============================
all_data = []
for model, splits in model_sources:
    for split in splits:
        path = os.path.join(data_dir, f'{model}.{split}.jsonl')
        if os.path.exists(path):
            texts = load_jsonl(path)
            for t in texts:
                all_data.append({'text': t, 'label': 1, 'source': model})
        else:
            print(f"[경고] 파일 없음: {path}")

# =============================
# Split 비율에 따라 분할
# =============================
total_size = 260_000
train_size = 250_000
valid_size = 5_000

if len(all_data) < total_size:
    raise ValueError(f"데이터 부족: {len(all_data)}개 로드됨, 최소 {total_size}개 필요")

# 데이터프레임화 및 셔플
df = pd.DataFrame(all_data).sample(frac=1, random_state=42).reset_index(drop=True)

# Split
train_df = df.iloc[:train_size]
valid_df = df.iloc[train_size:train_size + valid_size]
test_df = df.iloc[train_size + valid_size:total_size]

# =============================
# 저장 (원래 형식대로 jsonl)
# =============================
def save_jsonl(df, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            json.dump({'text': row['text']}, f)
            f.write('\n')

save_jsonl(train_df, os.path.join(out_dir, 'gpt2_k40.train.jsonl'))
save_jsonl(valid_df, os.path.join(out_dir, 'gpt2_k40.valid.jsonl'))
save_jsonl(test_df, os.path.join(out_dir, 'gpt2_k40.test.jsonl'))

print("✅ GPT-2 -k40 데이터 병합 및 분할 완료.")
