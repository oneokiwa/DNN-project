# # ========================================
# # SONL 형식 데이터 로딩
# # 문장 길이 분포 히스토그램
# # 자주 등장하는 단어 출력 (불용어 제거 포함)
# # WordCloud 시각화
# # ========================================

import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud
import nltk
import json
import os
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

plt.rcParams['font.family'] = ['Malgun Gothic', 'Segoe UI Emoji'] # 한글은 맑은 고딕, 이모지는 시고 이모지 폰트로 대체 렌더링
plt.rcParams['axes.unicode_minus'] = False
nltk.download('stopwords')

# ========================
# 데이터 로딩 함수
# ========================
def load_jsonl(path, limit=None):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if limit:
        lines = lines[:limit]
    return [json.loads(line)["text"] for line in lines]

# ========================
# 전처리 함수
# ========================
def tokenize(texts):
    stop_words = set(stopwords.words('english'))
    tokens = []
    for t in texts:
        tokens += [w.lower() for w in t.split() if w.isalpha() and w.lower() not in stop_words]
    return tokens

# ========================
# EDA 시각화 함수
# ========================
def plot_label_distribution(human_len, gpt_len):
    plt.hist(human_len, alpha=0.5, label='Human', bins=50)
    plt.hist(gpt_len, alpha=0.5, label='GPT-2', bins=50)
    plt.xlabel("문장 길이 (단어 수)")
    plt.ylabel("문장 수")
    plt.legend()
    plt.title("문장 길이 분포")
    plt.show()

def print_top_words(tokens, label):
    counter = Counter(tokens)
    print(f"[{label}] 상위 20개 단어:")
    for word, freq in counter.most_common(20):
        print(f"{word}: {freq}")

def show_wordcloud(texts, title):
    wc = WordCloud(width=800, height=400, background_color='white').generate(" ".join(texts))
    plt.figure(figsize=(15, 7))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

# ========================
# 특수문자, 숫자 등 통계 함수
# ========================
def get_special_stats(texts):
    return pd.DataFrame({
        'length': [len(t) for t in texts],
        'num_caps': [sum(1 for c in t if c.isupper()) for t in texts],
        'num_digits': [sum(1 for c in t if c.isdigit()) for t in texts],
        'num_punct': [len(re.findall(r'[.,!?;]', t)) for t in texts]
    })

# ========================
# n-gram 출력 함수
# ========================
def get_top_ngrams(texts, ngram_range=(2,2), top_k=20):
    vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=5000)
    X = vectorizer.fit_transform(texts)
    freqs = zip(vectorizer.get_feature_names_out(), X.sum(axis=0).tolist()[0])
    return sorted(freqs, key=lambda x: x[1], reverse=True)[:top_k]

# ========================
# TF-IDF 클래스별 주요 단어
# ========================
def show_tfidf_class_words(h_texts, g_texts, top_k=15):
    vect = TfidfVectorizer(max_features=5000)
    X = vect.fit_transform(h_texts + g_texts)
    features = vect.get_feature_names_out()
    human_mean = X[:len(h_texts)].mean(axis=0).A1
    gpt_mean = X[len(h_texts):].mean(axis=0).A1
    human_top = sorted(zip(features, human_mean), key=lambda x: x[1], reverse=True)[:top_k]
    gpt_top = sorted(zip(features, gpt_mean), key=lambda x: x[1], reverse=True)[:top_k]
    print("\n[Human TF-IDF 상위 단어]")
    for word, score in human_top:
        print(f"{word}: {score:.4f}")
    print("\n[GPT-2 TF-IDF 상위 단어]")
    for word, score in gpt_top:
        print(f"{word}: {score:.4f}")

# ========================
# 텍스트 유사도 분석
# ========================
def cosine_similarity_stats(texts, label):
    vect = TfidfVectorizer(max_features=1000)
    X = vect.fit_transform(texts[:1000])
    sim_matrix = cosine_similarity(X)
    triu = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
    print(f"[{label}] 평균 Cosine 유사도: {np.mean(triu):.4f}, 표준편차: {np.std(triu):.4f}")

# ========================
# 실행 코드
# ========================
if __name__ == '__main__':
    data_dir = './data'
    webtext = load_jsonl(os.path.join(data_dir, 'webtext.test.jsonl'))
    gpt2 = load_jsonl(os.path.join(data_dir, 'small-117M.test.jsonl'))

    # 문장 길이 분포
    webtext_lens = [len(text.split()) for text in webtext]
    gpt2_lens = [len(text.split()) for text in gpt2]
    plot_label_distribution(webtext_lens, gpt2_lens)

    # 단어 토큰화 및 상위 단어
    human_tokens = tokenize(webtext)
    gpt2_tokens = tokenize(gpt2)
    print_top_words(human_tokens, 'Human')
    print_top_words(gpt2_tokens, 'GPT-2')

    # 워드클라우드
    show_wordcloud(human_tokens, "WordCloud: Human Text")
    show_wordcloud(gpt2_tokens, "WordCloud: GPT-2 Text")

    # 특수문자/숫자 분석
    print("\n[Human 특수문자/숫자 통계 요약]")
    print(get_special_stats(webtext).describe())
    print("\n[GPT-2 특수문자/숫자 통계 요약]")
    print(get_special_stats(gpt2).describe())

    # n-gram 분석
    print("\n[Human 2-gram 상위 20개]")
    print(get_top_ngrams(webtext))
    print("\n[GPT-2 2-gram 상위 20개]")
    print(get_top_ngrams(gpt2))

    # TF-IDF 클래스별 주요 단어
    show_tfidf_class_words(webtext, gpt2)

    # 텍스트 유사도
    cosine_similarity_stats(webtext, 'Human')
    cosine_similarity_stats(gpt2, 'GPT-2')
