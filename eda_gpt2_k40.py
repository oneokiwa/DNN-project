# ========================================
## EDA 파이프라인 요약 ##
# JSONL 형식 데이터 로딩 및 전처리
# 1. 문장 길이 분포 히스토그램
# 2. 자주 등장하는 단어 출력
# 3. WordCloud 시각화
# 4. 특수문자 통계 요약
# 5. n-gram 분석
# 6. TF-IDF 시각화
# 7. 텍스트 코사인 유사도 분석
# ========================================
import os
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from collections import Counter
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 한글 지원
plt.rcParams['font.family'] = ['Malgun Gothic', 'Segoe UI Emoji']
plt.rcParams['axes.unicode_minus'] = False
nltk.download('stopwords')

# 데이터 로딩 함수
def load_jsonl(path, limit=None):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if limit:
        lines = lines[:limit]
    return [json.loads(line)["text"] for line in lines]

# 전처리 함수 (불용어 제거 후 토큰화)
def tokenize(texts):
    stop_words = set(stopwords.words('english'))
    tokens = []
    for t in texts:
        tokens += [w.lower() for w in t.split() if w.isalpha() and w.lower() not in stop_words]
    return tokens

# 문장 길이 시각화
def plot_label_distribution(human_len, gpt_len):
    plt.hist(human_len, alpha=0.5, label='Human', bins=50)
    plt.hist(gpt_len, alpha=0.5, label='GPT-2', bins=50)
    plt.xlabel("문장 길이 (단어 수)")
    plt.ylabel("문장 수")
    plt.legend()
    plt.title("문장 길이 분포")
    plt.savefig("output/sentence_length_histogram.png")
    plt.close()

# 단어 상위 출력
def print_top_words(tokens, label):
    counter = Counter(tokens)
    print(f"[{label}] 상위 20개 단어:")
    for word, freq in counter.most_common(20):
        print(f"{word}: {freq}")

# 워드클라우드
def show_wordcloud(texts, title):
    wc = WordCloud(width=800, height=400, background_color='white').generate(" ".join(texts))
    plt.figure(figsize=(15, 7))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.savefig(f"output/{title.replace(' ', '_').lower()}.png")
    plt.close()

# 특수문자 통계
def get_special_stats(texts):
    return pd.DataFrame({
        'length': [len(t) for t in texts],
        'num_caps': [sum(1 for c in t if c.isupper()) for t in texts],
        'num_digits': [sum(1 for c in t if c.isdigit()) for t in texts],
        'num_punct': [len(re.findall(r'[.,!?;]', t)) for t in texts]
    })

# n-gram 분석
def get_top_ngrams(texts, ngram_range=(2,2), top_k=20):
    vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=5000)
    X = vectorizer.fit_transform(texts)
    freqs = zip(vectorizer.get_feature_names_out(), X.sum(axis=0).tolist()[0])
    return sorted(freqs, key=lambda x: x[1], reverse=True)[:top_k]

# TF-IDF 분석
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
    return human_top, gpt_top

# Cosine 유사도 분석
def cosine_similarity_stats(texts, label):
    vect = TfidfVectorizer(max_features=1000)
    X = vect.fit_transform(texts[:1000])
    sim_matrix = cosine_similarity(X)
    triu = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
    print(f"[{label}] 평균 Cosine 유사도: {np.mean(triu):.4f}, 표준편차: {np.std(triu):.4f}")

# 시각화 함수들
def plot_special_stats(df1, df2):
    metrics = ['length', 'num_caps', 'num_digits', 'num_punct']
    df1_mean = df1.describe().loc['mean', metrics]
    df2_mean = df2.describe().loc['mean', metrics]
    x = range(len(metrics))
    plt.figure(figsize=(10, 5))
    plt.bar([i - 0.2 for i in x], df1_mean, width=0.4, label='Human')
    plt.bar([i + 0.2 for i in x], df2_mean, width=0.4, label='GPT-2')
    plt.xticks(x, metrics)
    plt.ylabel("평균 값")
    plt.title("특수문자/숫자 평균 비교")
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/special_char_stats_comparison.png")
    plt.close()

def plot_ngram_comparison(h_df, g_df):
    df = pd.merge(h_df, g_df, on="2-gram", how="outer").fillna(0)
    x = range(len(df))
    plt.figure(figsize=(14, 6))
    plt.bar([i - 0.2 for i in x], df["Human 빈도"], width=0.4, label="Human")
    plt.bar([i + 0.2 for i in x], df["GPT-2 빈도"], width=0.4, label="GPT-2")
    plt.xticks(x, df["2-gram"], rotation=45)
    plt.ylabel("빈도")
    plt.title("2-gram 상위 20개 비교")
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/ngram_comparison.png")
    plt.close()

def plot_tfidf_comparison(h_top, g_top):
    df = pd.DataFrame(h_top, columns=["단어", "Human TF-IDF"])
    df_g = pd.DataFrame(g_top, columns=["단어", "GPT-2 TF-IDF"])
    df = pd.merge(df, df_g, on="단어", how="outer").fillna(0)
    x = range(len(df))
    plt.figure(figsize=(14, 6))
    plt.bar([i - 0.2 for i in x], df["Human TF-IDF"], width=0.4, label="Human")
    plt.bar([i + 0.2 for i in x], df["GPT-2 TF-IDF"], width=0.4, label="GPT-2")
    plt.xticks(x, df["단어"], rotation=45)
    plt.ylabel("TF-IDF 점수")
    plt.title("TF-IDF 상위 단어 비교")
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/tfidf_comparison.png")
    plt.close()

def plot_cosine_comparison():
    labels = ['Human', 'GPT-2']
    means = [0.2662, 0.2968]
    stds = [0.1292, 0.1203]
    x = range(len(labels))
    plt.figure(figsize=(6, 5))
    plt.bar(x, means, yerr=stds, capsize=10, tick_label=labels)
    plt.ylabel("평균 Cosine 유사도")
    plt.title("Cosine Similarity 비교")
    plt.tight_layout()
    plt.savefig("output/cosine_similarity_comparison.png")
    plt.close()

# 실행
if __name__ == '__main__':
    os.makedirs("output", exist_ok=True)
    data_dir_gpt = './data/merged_gpt2_k40'
    data_dir_human = './data'
    webtext = load_jsonl(os.path.join(data_dir_human, 'webtext.test.jsonl'))
    gpt2_k40 = load_jsonl(os.path.join(data_dir_gpt, 'gpt2_k40.test.jsonl'))

    webtext_lens = [len(text.split()) for text in webtext]
    gpt2_lens = [len(text.split()) for text in gpt2_k40]
    plot_label_distribution(webtext_lens, gpt2_lens)

    human_tokens = tokenize(webtext)
    gpt2_tokens = tokenize(gpt2_k40)
    print_top_words(human_tokens, 'Human')
    print_top_words(gpt2_tokens, 'GPT-2')
    show_wordcloud(human_tokens, "WordCloud Human Text")
    show_wordcloud(gpt2_tokens, "WordCloud GPT-2 Text")

    web_stats = get_special_stats(webtext)
    gpt2_stats = get_special_stats(gpt2_k40)
    print("\n[Human 특수문자/숫자 통계 요약]")
    print(web_stats.describe())
    print("\n[GPT-2 특수문자/숫자 통계 요약]")
    print(gpt2_stats.describe())
    plot_special_stats(web_stats, gpt2_stats)

    web_ngrams = pd.DataFrame(get_top_ngrams(webtext), columns=["2-gram", "Human 빈도"])
    gpt2_ngrams = pd.DataFrame(get_top_ngrams(gpt2_k40), columns=["2-gram", "GPT-2 빈도"])
    print("\n[Human 2-gram 상위 20개]")
    print(web_ngrams)
    print("\n[GPT-2 2-gram 상위 20개]")
    print(gpt2_ngrams)
    plot_ngram_comparison(web_ngrams, gpt2_ngrams)

    h_top, g_top = show_tfidf_class_words(webtext, gpt2_k40)
    plot_tfidf_comparison(h_top, g_top)

    cosine_similarity_stats(webtext, 'Human')
    cosine_similarity_stats(gpt2_k40, 'GPT-2')
    plot_cosine_comparison()

    print("\n전체 분석 및 시각화 완료: output 폴더 확인")