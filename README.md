# GPT2 Dataset - EDA

**EDA에 사용한 데이터**:
webtext.test.jsonl, gpt2_k40.test.jsonl

- GPT-2 모델 전체를 하나로 묶어서,
- label = 1로 지정하고,
- EDA는 전체 GPT-2 묶음 vs Human만 수행.

### 병합 플로우

1. train, test, valid 싹 다 병합
2. 하나로 병합된 GPT-2 모델을 train, test, valid로 split
3. test 데이터만 사용해서 Human과 비교 EDA

### split 비율

- **Train**: 250,000개 샘플
- **Validation**: 5,000개 샘플
- **Test**: 5,000개 샘플

전체 260,000개 샘플 중 약 96%가 훈련용 데이터로, 나머지 4%가 검증 및 테스트용 데이터로 할당

**최종 디렉토리 구조**
```
📂 gpt-2-output-dataset/
├── data/
│   ├── webtext.train.jsonl             # Human 텍스트 (Train)
│   ├── webtext.valid.jsonl             # Human 텍스트 (Validation)
│   ├── webtext.test.jsonl              # Human 텍스트 (Test)
│
├── data/merged_gpt2_k40/               # GPT-2 (top-k=40) 병합본
│   ├── gpt2_k40.train.jsonl            # GPT-2 텍스트 (Train)
│   ├── gpt2_k40.valid.jsonl            # GPT-2 텍스트 (Validation)
│   ├── gpt2_k40.test.jsonl             # GPT-2 텍스트 (Test)
```
---

## 1. 문장 길이 분포 히스토그램

- 목적: 두 클래스의 구조적 차이를 확인하기 위해, 각 문장의 단어 수 분포를 시각화하여 비교.
- **결과 해석**

| 구간 | 해석 |
| --- | --- |
| 0~100 | Human: 짧은 문장 비율이 더 높음 → 단문 중심 구조 존재
| 800~900 | GPT2: 정형화된 길이로 문장을 생성하는 경향 → 샘플링 파라미터 (`max_length`) 영향 가능성
| ~1000 | GPT2: 거의 1000단어에 근접한 문장을 자주 생성→ 샘플링 파라미터 (`max_length`) 영향 가능성 

![Image](https://github.com/user-attachments/assets/5bca2e9e-abde-4e0e-b68f-75e2968af7c8)

---

## 2. 등장 빈도수 정리

- 목적: 이진 분류의 유효 Feature 탐색
- 그래프 기반 유효 Feature 후보 정리

| Lexical Feature | 이유 |
| --- | --- |
| `also` | 불필요한 접속부사 반복 경향 |
| `would`, `could` | 조건/가정문 과다 사용 |
| `get`, `make`, `want`, `think`, `see` | 서술/의지/행동 중심 표현 반복 |
| `new`, `one`, `people` | 모호하거나 일반화된 명사 → 서사 반복 구조 가능성

![Image](https://github.com/user-attachments/assets/b216bf3c-e781-4669-8749-616483119689)

### 단어 빈도 비교 (Top 20 Words)

<table>
<tr>
<td>

####  Human 상위 20개 단어

| 단어 | 빈도 |
|------|------|
| one   | 4994 |
| new   | 4028 |
| said  | 4028 |
| would | 3985 |
| also  | 3766 |
| like  | 3364 |
| people| 2928 |
| first | 2701 |
| two   | 2604 |
| get   | 2579 |
| even  | 2332 |
| time  | 2315 |
| could | 2307 |
| may   | 2185 |
| many  | 2078 |
| last  | 2043 |
| make  | 1983 |
| use   | 1819 |
| see   | 1720 |
| us    | 1685 |

</td>
<td>

#### GPT-2 상위 20개 단어

| 단어 | 빈도 |
|------|------|
| new    | 7047 |
| would  | 6723 |
| one    | 6634 |
| also   | 6295 |
| said   | 6126 |
| people | 6114 |
| get    | 4731 |
| like   | 4422 |
| first  | 4368 |
| make   | 4209 |
| going  | 3682 |
| could  | 3551 |
| time   | 3409 |
| want   | 3174 |
| two    | 3144 |
| many   | 2982 |
| even   | 2845 |
| think  | 2820 |
| see    | 2818 |
| use    | 2733 |

</td>
</tr>
</table>

---

## 3. Word Cloud

- 위의 빈도수를 Word Cloud로 시각화

### 1) Human Text Word Cloud

![Image](https://github.com/user-attachments/assets/eb7d659e-a54d-4e69-9e4a-8cbb6a61b143)

### 2) GPT-2 Text Word Cloud

![Image](https://github.com/user-attachments/assets/7cacbb75-d045-4180-a187-f27f254d7b16)

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

![Image](https://github.com/user-attachments/assets/26b79a60-7bb6-43ad-bccb-3891278af503)

**Human**

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

**GPT2_k40**

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

![Image](https://github.com/user-attachments/assets/01c50a3d-0957-48a8-beab-9fb70a94898c)

### 2-gram 빈도 비교 (Top 20 Pairs)

<table>
<tr>
<td>

####  Human 상위 2-gram

| 2-gram | Human 빈도 |
|--------|-------------|
| of the     | 12688 |
| in the     | 10088 |
| to the     | 5694  |
| on the     | 4729  |
| for the    | 3619  |
| and the    | 3231  |
| to be      | 3157  |
| at the     | 2950  |
| with the   | 2763  |
| from the   | 2410  |
| that the   | 2395  |
| by the     | 1881  |
| it is      | 1870  |
| if you     | 1586  |
| is the     | 1586  |
| one of     | 1577  |
| will be    | 1540  |
| this is    | 1512  |
| it was     | 1507  |
| the first  | 1369  |

</td>
<td>

####  GPT-2 상위 2-gram

| 2-gram | GPT-2 빈도 |
|--------|-------------|
| of the     | 16388 |
| in the     | 14291 |
| to the     | 6927  |
| on the     | 5731  |
| to be      | 5453  |
| for the    | 4974  |
| and the    | 4762  |
| that the   | 4383  |
| with the   | 3891  |
| at the     | 3601  |
| if you     | 3123  |
| it is      | 3093  |
| from the   | 2968  |
| the first  | 2884  |
| going to   | 2871  |
| this is    | 2495  |
| you can    | 2457  |
| will be    | 2430  |
| by the     | 2419  |
| it was     | 2410  |

</td>
</tr>
</table>

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
    

![Image](https://github.com/user-attachments/assets/cd590346-beb9-476a-9cee-ea589bf44548)

### TF-IDF 상위 단어 비교

<table>
<tr>
<td>

####  Human TF-IDF 상위 단어

| 단어 | Human TF-IDF |
|------|---------------|
| the   | 0.2243 |
| to    | 0.1125 |
| and   | 0.1025 |
| of    | 0.1017 |
| in    | 0.0780 |
| that  | 0.0508 |
| is    | 0.0486 |
| for   | 0.0451 |
| it    | 0.0414 |
| on    | 0.0388 |
| you   | 0.0380 |
| with  | 0.0349 |
| he    | 0.0330 |
| as    | 0.0307 |
| was   | 0.0307 |

</td>
<td>

####  GPT-2 TF-IDF 상위 단어

| 단어 | GPT-2 TF-IDF |
|------|----------------|
| the   | 0.2516 |
| to    | 0.1338 |
| of    | 0.1076 |
| and   | 0.0980 |
| in    | 0.0853 |
| that  | 0.0727 |
| it    | 0.0551 |
| is    | 0.0533 |
| you   | 0.0532 |
| for   | 0.0458 |
| was   | 0.0407 |
| we    | 0.0404 |
| on    | 0.0390 |
| he    | 0.0363 |
| this  | 0.0356 |

</td>
</tr>
</table>

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
---
# original readme
# gpt-2-output-dataset

This dataset contains:
- 250K documents from the WebText test set
- For each GPT-2 model (trained on the WebText training set), 250K random samples (temperature 1, no truncation) and 250K samples generated with Top-K 40 truncation

We look forward to the research produced using this data!

### Download

For each model, we have a training split of 250K generated examples, as well as validation and test splits of 5K examples.

All data is located in Google Cloud Storage, under the directory `gs://gpt-2/output-dataset/v1`.  (NOTE: everything has been migrated to Azure `https://openaipublic.blob.core.windows.net/gpt-2/output-dataset/v1/`)

There, you will find files:

- `webtext.${split}.jsonl`
- `small-117M.${split}.jsonl`
- `small-117M-k40.${split}.jsonl`
- `medium-345M.${split}.jsonl`
- `medium-345M-k40.${split}.jsonl`
- `large-762M.${split}.jsonl`
- `large-762M-k40.${split}.jsonl`
- `xl-1542M.${split}.jsonl`
- `xl-1542M-k40.${split}.jsonl`

where split is one of `train`, `test`, and `valid`.

We've provided a script to download all of them, in `download_dataset.py`.

#### Finetuned model samples

Additionally, we encourage research on detection of finetuned models.  We have released data under `gs://gpt-2/output-dataset/v1-amazonfinetune/` with samples from a GPT-2 full model finetuned to output Amazon reviews.

### Detectability baselines

We're interested in seeing research in detectability of GPT-2 model family generations.

We provide some [initial analysis](detection.md) of two baselines, as well as [code](./baseline.py) for the better baseline.

Overall, we are able to achieve accuracies in the mid-90s for Top-K 40 generations, and mid-70s to high-80s (depending on model size) for random generations.  We also find some evidence that adversaries can evade detection via finetuning from released models.

### Data removal requests

If you believe your work is included in WebText and would like us to remove it, please let us know at webtextdata@openai.com.
