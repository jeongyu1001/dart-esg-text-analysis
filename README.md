# DART 사업보고서 비정형 텍스트 기반 ESG 평가

> DART 전자공시 사업보고서의 ESG 관련 텍스트를 3가지 층위(Intensity · Content · Semantic)로 분석하여  
> KCGS ESG 등급과의 연관성을 실증하고, G 영역 Cheap-talk 및 그린워싱 위험도를 탐지한 연구 프로젝트입니다.

---

## 연구 개요

| 항목 | 내용 |
|------|------|
| **분석 대상** | 127개 상장기업 × 2022·2023·2024년 = **381개 기업-연도** |
| **텍스트 원천** | DART 사업보고서 (`II. 사업의 내용`, `IV. 경영진단`, `VI. 이사회`) |
| **외부 기준** | KCGS ESG 등급 (S/A+/A/B+/B/C/D) |
| **핵심 방법론** | 3-Layer 텍스트 피처 + OLS 회귀 + TriSignal 그린워싱 탐지 |

### 핵심 연구 질문

> "사업보고서의 ESG 서술 신호는 외부 ESG 등급을 유의미하게 설명하는가?  
> 그리고 서술 신호 내에 실질과 괴리된 Cheap-talk 패턴이 존재하는가?"

---

## 분석 파이프라인

```
DART OpenAPI
    │
    ▼
[PART 1] 데이터 수집 · 전처리          01_전처리.ipynb
    │  · OpenDART API ZIP 수집
    │  · ESG 문장 추출 + 오탐 제거
    │
    ▼
[PART 2] Layer 1 — Intensity           02_텍스트_Intensity.ipynb
    │  · Kiwi 형태소 분석 (NNG)
    │  · Seed Dictionary 키워드 빈도
    │  · FastText 유사어 확장
    │
    ▼
[PART 3] Layer 2 — Content             03_텍스트_Content.ipynb
    │  · TF-IDF (max_features=500)
    │  · TruncatedSVD → 유의 PC 선별
    │
    ▼
[PART 4] Layer 3 — Semantic            04_텍스트_Semantic.ipynb
    │  · KR-FinBert 768차원 임베딩
    │  · 문서 평균 풀링 → PCA
    │
    ▼
[PART 5] 통합 OLS 회귀 분석            05_통합회귀분析.ipynb
    │  · M1(Seed) → M2(+Content) → M3b(+Semantic) 순차 추가
    │  · R² 점진적 향상 검증
    │
    ▼
[PART 6] G Cheap-talk · Robustness     06_알파분析.ipynb
    │  · G_seed 음(−) 계수 → G 공시 과잉 탐지
    │  · 업종·규모·연도 조건별 Robustness
    │
    ▼
[PART 7] TriSignal 그린워싱 탐지       07_그린워싱탐지.ipynb
       · L1(텍스트↑) + L2(CAPEX↓) + L3(KCGS↑) 교차 점수
       · High / Medium / Low Risk 분류
```

---

## 주요 결과

### 3-Layer 모델 설명력 (OLS, n=364)

| 모델 | 피처 | R² | Adj. R² |
|------|------|----|---------|
| M1 | Seed Score | 0.164 | 0.156 |
| M2 | + Content (TF-IDF SVD) | 0.241 | 0.229 |
| M3b ★ | + Semantic (KR-FinBert) | **0.298** | **0.282** |

### G 영역 Cheap-talk 발견

- G_seed 계수 **−0.808 (p=0.003\*\*)** — G 키워드 공시량이 많을수록 오히려 등급 낮음
- 업종·규모·연도 전 구간에서 방향(−) 일관 유지 → 구조적 현상
- KR-FinBert 투입 시 G_seed 비유의 → 모델이 G 공시의 **질**을 포착

### TriSignal 그린워싱 탐지

- 비금융 + CAPEX 있음: `GW = 0.35·L1 − 0.45·L2 − 0.20·L3`
- High Risk 상위 20% 기업 분류

---

## 기술 스택

| 분야 | 라이브러리 |
|------|-----------|
| 데이터 수집 | `requests`, `python-dotenv` (OpenDART API) |
| 형태소 분석 | `kiwipiepy` |
| 텍스트 임베딩 | `transformers` (KR-FinBert), `gensim` (FastText) |
| 통계 분석 | `statsmodels`, `scipy` |
| 머신러닝 | `scikit-learn` (TF-IDF, SVD, PCA) |
| 번역·분류 | `Helsinki-NLP/opus-mt-ko-en`, `FinBERT-ESG` |

---

## 프로젝트 구조

```
├── code/
│   ├── config.py                   # 공통 경로·상수·유틸
│   ├── 00_연구개요.ipynb
│   ├── 01_전처리.ipynb              # DART 수집 + ESG 문장 추출
│   ├── 02_텍스트_Intensity.ipynb    # Seed Score + FastText 확장
│   ├── 03_텍스트_Content.ipynb      # TF-IDF + SVD (Content Layer)
│   ├── 04_텍스트_Semantic.ipynb     # KR-FinBert 임베딩 (Semantic Layer)
│   ├── 05_통합회귀분析.ipynb         # 통합 OLS + FinBERT-ESG 검증
│   ├── 06_알파분析.ipynb             # G Cheap-talk + Robustness
│   ├── 07_그린워싱탐지.ipynb         # TriSignal GW 탐지
│   └── data/
│       ├── company_master.csv
│       ├── seed_dictionary.csv
│       ├── ESG_passage_기업-연도_final.csv
│       ├── ESG_passages_tokenized.csv
│       ├── ESG_키워드index_v2.csv
│       ├── tfidf_sig_pcs.csv
│       ├── krfinbert_sig_pcs.csv
│       ├── 4차_ESG_integrated_dataset.csv
│       ├── 4차_ESG_consistency.csv
│       ├── capex_df.csv / fin_df.csv
│       ├── kcgs_grade.csv
│       └── greenwashing_final_result.csv
├── Short Research Paper.docx
├── 서비스 제안.html
├── requirements.txt
└── README.md
```

---

## 실행 방법

```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. OpenDART API 키 설정
echo "OPENDART_API_KEY=your_key_here" > code/.env

# 3. 노트북 순서대로 실행 (01 → 07)
jupyter notebook code/
```

> **참고**: `01_전처리.ipynb`는 OpenDART API 키가 필요합니다.  
> `02_07`은 `data/` 폴더의 전처리 완료 파일로 바로 실행 가능합니다.

---

## 데이터 출처

- **DART 전자공시**: [OpenDART API](https://opendart.fss.or.kr)
- **ESG 등급**: KCGS (한국기업지배구조원) 2022–2024
- **재무 데이터**: KRX 시가총액, CAPEX
