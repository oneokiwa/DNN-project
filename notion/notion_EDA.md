# EDA

**목차**

---

**EDA에 사용한 데이터**:
webtext.test.jsonl

gpt2_k40.test.jsonl

---

## 1. 문장 길이 분포 히스토그램

- 목적: 두 클래스의 구조적 차이를 확인하기 위해, 각 문장의 단어 수 분포를 시각화하여 비교.
- **결과 해석**

| 구간 | 해석 |
| --- | --- |
| 0~100 | Human: 짧은 문장 비율이 더 높음 
→ 단문 중심 구조 존재 |
| 800~900 | GPT2: 정형화된 길이로 문장을 생성하는 경향
→ 샘플링 파라미터 (`max_length`) 영향 가능성 |
| ~1000 | GPT2: 거의 1000단어에 근접한 문장을 자주 생성 
→ 샘플링 파라미터 (`max_length`) 영향 가능성

 |

![image.png](image.png)

---

## 2. 등장 빈도수 정리

- 목적: 이진 분류의 유효 Feature 탐색
- 그래프 기반 유효 Feature 후보 정리

| Lexical Feature | 이유 |
| --- | --- |
| `also` | 불필요한 접속부사 반복 경향 |
| `would`, `could` | 조건/가정문 과다 사용 |
| `get`, `make`, `want`, `think`, `see` | 서술/의지/행동 중심 표현 반복 |
| `new`, `one`, `people` | 모호하거나 일반화된 명사 
→ 서사 반복 구조 가능성 |

![image.png](image%201.png)

| 단어 | Human 빈도 |
| --- | --- |
| one | 4994 |
| new | 4028 |
| said | 4028 |
| would | 3985 |
| also | 3766 |
| like | 3364 |
| people | 2928 |
| first | 2701 |
| two | 2604 |
| get | 2579 |
| even | 2332 |
| time | 2315 |
| could | 2307 |
| may | 2185 |
| many | 2078 |
| last | 2043 |
| make | 1983 |
| use | 1819 |
| see | 1720 |
| us | 1685 |

| 단어 | GPT-2 빈도 |
| --- | --- |
| new | 7047 |
| would | 6723 |
| one | 6634 |
| also | 6295 |
| said | 6126 |
| people | 6114 |
| get | 4731 |
| like | 4422 |
| first | 4368 |
| make | 4209 |
| going | 3682 |
| could | 3551 |
| time | 3409 |
| want | 3174 |
| two | 3144 |
| many | 2982 |
| even | 2845 |
| think | 2820 |
| see | 2818 |
| use | 2733 |

---

## 3. Word Cloud

- 위의 빈도수를 Word Cloud로 시각화

### 1) Human Text Word Cloud

![image.png](image%202.png)

### 2) GPT-2 Text Word Cloud

![image.png](image%203.png)

---

## 4. 특수 문자, 숫자, 대문자 비율 확인 (통계 요약)

- 목적: AI는 일반적으로 잘못된 문장 부호나 숫자 조합을 다르게 처리할 수 있음.
- **EDA 분석**

| 항목 | Human 평균 | GPT2 평균 | 차이 해석 |
| --- | --- | --- | --- |
| length (텍스트 길이) | 2593.76 | **2881.80** | GPT2 문장이 약 **11%** 더 긺 |
| num_caps (대문자 수) | **93.30** | 82.77 | GPT2 모델에서 문장 시작/고유명사 사용 부족 가능성 존재 |
| num_digits (숫자 수) | **28.50** | 20.17 | Human이 숫자가 포함된 문장(날짜, 수치)을 더 자주 다룸 |
| num_punct (문장부호 수) | 47.88 | **50.83** | 자연스러운 문장 흐름 유지 시도로 해석 가능 |

![image.png](image%204.png)

**Human                                                                                       GPT2_k40**

| 항목 | length | num_caps | num_digits | num_punct |
| --- | --- | --- | --- | --- |
| count | 5000 | 5000 | 5000 | 5000 |
| mean | 2593.76 | 93.3 | 28.5 | 47.88 |
| std | 1619.72 | 80.37 | 55.65 | 34.4 |
| min | 201 | 0 | 0 | 0 |
| 25% | 1097.75 | 39 | 4 | 19 |
| 50% | 2426.5 | 79 | 14 | 42 |
| 75% | 4268.25 | 127.25 | 32 | 75 |
| max | 5764 | 2132 | 979 | 680 |

| 항목 | length | num_caps | num_digits | num_punct |
| --- | --- | --- | --- | --- |
| count | 5000 | 5000 | 5000 | 5000 |
| mean | 2881.8 | 82.77 | 20.17 | 50.83 |
| std | 1575.97 | 61.01 | 39.84 | 31.5 |
| min | 1 | 0 | 0 | 0 |
| 25% | 1465.75 | 41 | 2 | 25 |
| 50% | 2913.5 | 72 | 10 | 49 |
| 75% | 4425.75 | 111 | 24 | 75 |
| max | 6137 | 1023 | 1229 | 410 |

---

## 5. 자주 쓰는 문장 파악

### **n-gram 분석 (2-gram)**

- 목적: 단어 조합 패턴을 통해 GPT-2가 자주 쓰는 문장 흐름을 찾음.
- ex). ‘it is’, ‘in the’, ‘this is not’ 등
- **EDA 분석**
1. Human, GPT2 모두 `'of the'`, `'in the'`, `'to the'` 등 전치사 + 관사 조합을 많이 사용.
2. GPT-2에서 `'going to'`, `'you can'`, `'this is'` 등이 추가로 상위권에 위치.
    
    → 구어체적 표현이 두드러짐
    
3. GPT-2의 2-gram 빈도수가 전반적으로 Human보다 높음. 
    
    → 더 반복적이고 특정한 문장 구조에 편중
    
- **결과 해석**
    
    **: GPT-2**
    
1. 상대적으로 고정된 패턴을 자주 사용하여 일관성은 있지만 다양성이 부족함. (패턴 기반 생성 특성)
2. 미래 표현을 많이 사용 ("will be", "you can", "going to")

![image.png](image%205.png)

**Human                                                                 GPT2_k40**

| 2-gram | Human 빈도 |
| --- | --- |
| of the | 12688 |
| in the | 10088 |
| to the | 5694 |
| on the | 4729 |
| for the | 3619 |
| and the | 3231 |
| to be | 3157 |
| at the | 2950 |
| with the | 2763 |
| from the | 2410 |
| that the | 2395 |
| by the | 1881 |
| it is | 1870 |
| if you | 1586 |
| is the | 1586 |
| one of | 1577 |
| will be | 1540 |
| this is | 1512 |
| it was | 1507 |
| the first | 1369 |

| 2-gram | GPT-2 빈도 |
| --- | --- |
| of the | 16388 |
| in the | 14291 |
| to the | 6927 |
| on the | 5731 |
| to be | 5453 |
| for the | 4974 |
| and the | 4762 |
| that the | 4383 |
| with the | 3891 |
| at the | 3601 |
| if you | 3123 |
| it is | 3093 |
| from the | 2968 |
| the first | 2884 |
| going to | 2871 |
| this is | 2495 |
| you can | 2457 |
| will be | 2430 |
| by the | 2419 |
| it was | 2410 |

---

## 6. TF-IDF 시각화

- 목적: 두 클래스에 대해 중요한 단어를 TF-IDF로 추출해보고, 클래스를 구분하는 단어들을 시각화.
- TF-IDF 점수:  해당 클래스 (ex. GPT-2 텍스트)에만 상대적으로 자주 등장하는 단어를 찾기 위한 수단.
- top_k=15로 설정
- **EDA 분석**

| 공통 단어 | the, to, of, and, in, that, it, is, for, on |
| --- | --- |
| 차이점 | GPT2: we, this / Human: he, as |
- **결과 해석**
1. 공통 단어 다수 → 주제 자체는 유사할 수 있음.
2. GPT2: `'we'`, `'this'` 같은 일반적 주어 표현을 강조.
    
    → 주관적 문장 경향
    
3. Human: `'with'`, `'as'` 등의 단어가 등장 (연결어).
    
    → 정보 전달 및 설명 중심 경향
    

![image.png](image%206.png)

**Human                                                         GPT2_k40**

| 단어 | Human TF-IDF |
| --- | --- |
| the | 0.2243 |
| to | 0.1125 |
| and | 0.1025 |
| of | 0.1017 |
| in | 0.078 |
| that | 0.0508 |
| is | 0.0486 |
| for | 0.0451 |
| it | 0.0414 |
| on | 0.0388 |
| you | 0.038 |
| with | 0.0349 |
| he | 0.033 |
| as | 0.0307 |
| was | 0.0307 |

| 단어 | GPT-2 TF-IDF |
| --- | --- |
| the | 0.2516 |
| to | 0.1338 |
| of | 0.1076 |
| and | 0.098 |
| in | 0.0853 |
| that | 0.0727 |
| it | 0.0551 |
| is | 0.0533 |
| you | 0.0532 |
| for | 0.0458 |
| was | 0.0407 |
| we | 0.0404 |
| on | 0.039 |
| he | 0.0363 |
| this | 0.0356 |

---

## 7. 텍스트 유사도 분석 (Cosine Similarity)

- 목적: GPT 텍스트들끼리는 서로 유사한 경향이 강한지, Human 텍스트는 더 다양성을 띄는지 분석.
- **결과 해석**
1. GPT2 문장들끼리는 더 유사한 단어 구조를 공유. 
    
    → 표현 다양성이 낮고 반복적
    
2. Human 문장은 서로 다양한 주제와 표현 방식을 포함.
    
    → 유사도는 낮고 표현력이 풍부
    

![image.png](image%207.png)

| 측정 항목 | **Human** | **GPT-2** |
| --- | --- | --- |
| 평균 Cosine 유사도 | 0.2662 | 0.2968 |
| 표준편차 | 0.1292 | 0.1203 |

---

## 8. 해석 종합 결론

| 항목 | GPT-2 특성 | Human 특성 |
| --- | --- | --- |
| 단어 조합 (n-gram) | 반복적, 일상 표현 위주 | 정보 전달 중심의 구문 |
| 숫자/대문자 사용 | 부족 | 다양함 |
| TF-IDF | 고빈도 일반 단어 위주 | 맥락 의미 중심 단어 포함 |
| 문장 유사도 | 높은 유사도 → 패턴화 | 낮은 유사도 → 다양성 |

### **총평!**

<aside>

 GPT2_k40 모델은 유창성은 높지만, 창의성과 **정보성** 측면에서 Human 텍스트보다 부족한 것으로 나타난다. 

이는 모델의 제한된 문맥 이해 및 패턴 학습 방식의 한계를 반영하며, 분류 모델 구축 시 이러한 반복성/유사성이 주요 feature로 활용될 수 있다.

</aside>

---

# 추가) 전체 코드

- eda_gpt2_k40.py
    
    ```python
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
    ```