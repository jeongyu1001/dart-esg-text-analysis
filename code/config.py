"""
ESG DART 프로젝트 공통 설정
모든 노트북에서 import config 후 사용
"""
import os

# ── 기본 경로 ─────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

PATHS = {
    # 공통 데이터
    "master":        f"{BASE}/data/company_master.csv",
    "seed_dict":     f"{BASE}/data/seed_dictionary.csv",
    "industry":      f"{BASE}/data/company_industry_dart.csv",
    "marcap":        f"{BASE}/data/company_marcap.csv",

    # STEP 산출물
    "tokenized":     f"{BASE}/data/ESG_passages_tokenized.csv",
    "seed_score":    f"{BASE}/data/ESG_키워드index_v2.csv",
    "tfidf_sig":     f"{BASE}/data/tfidf_sig_pcs.csv",
    "tfidf_full":    f"{BASE}/data/svd_tfidf_optimal.csv",
    "embedding":     f"{BASE}/data/ESG_임베딩_KRFinBert.csv",
    "semc_sig":      f"{BASE}/data/krfinbert_sig_pcs.csv",
    "semc_full":     f"{BASE}/data/ESG_임베딩_KRFinBert.csv",
    "finbert":       f"{BASE}/data/ESG_finbert_scores.csv",
    "passage_en":    f"{BASE}/data/ESG_passage_en.csv",

    # 통합 산출물
    "integrated":    f"{BASE}/data/4차_ESG_integrated_dataset.csv",
    "passage_final": f"{BASE}/data/ESG_passage_기업-연도_final.csv",
}

# ── 공통 상수 ─────────────────────────────────────────────────
KEY = ["stock_code", "fiscal_year"]
YEARS = [2022, 2023, 2024]

GRADE_MAP = {"S": 6, "A+": 5, "A": 4, "B+": 3, "B": 2, "C": 1, "D": 0}
GRADE_MAP_INV = {v: k for k, v in GRADE_MAP.items()}

# E-intensive 업종 (KSIC 상위 2자리)
E_INTENSIVE_TOP2 = {'07','08','09','19','20','23','24','35','06','29','22','17','25'}

# 분석에 사용할 핵심 피처
FEAT_SEED     = ["E_seed_norm_log", "S_seed_norm_log", "G_seed_norm_log"]
FEAT_RATIO    = ["E_ratio", "S_ratio", "G_ratio"]
FEAT_CONTENT  = ["neg_TFPC2", "TFPC1"]
FEAT_SEMANTIC = ["neg_KRFPC1", "KRFPC2"]
FEAT_CONTROL  = ["E_intensive"]
FEAT_FULL     = FEAT_SEED + FEAT_CONTENT + FEAT_SEMANTIC + FEAT_CONTROL + FEAT_RATIO

GRADE_COLS = ["esg_numeric", "e_numeric", "s_numeric", "g_numeric"]

# ── 유틸 함수 ─────────────────────────────────────────────────
def load(key):
    """config.PATHS[key] CSV 로드 + stock_code zfill(6)"""
    import pandas as pd
    df = pd.read_csv(PATHS[key])
    if "stock_code" in df.columns:
        df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
    return df

def add_grade_numeric(df):
    """esg/e/s/g_grade → numeric 컬럼 추가"""
    for col in ["esg_grade", "e_grade", "s_grade", "g_grade"]:
        if col in df.columns:
            df[col.replace("grade", "numeric")] = df[col].map(GRADE_MAP)
    return df
