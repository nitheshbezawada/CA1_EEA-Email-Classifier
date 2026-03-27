# ─────────────────────────────────────────────
#  preprocessor.py  |  NLP Cleaning Pipeline
#  Each function handles one cleaning concern.
#  Completely independent of modelling code.
# ─────────────────────────────────────────────

import re
import pandas as pd
from config import Config


# ── Step 1: Remove duplicate rows ───────────

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"  [PREPROCESS] Removed {before - len(df)} duplicate rows → {len(df)} remain")
    return df


# ── Step 2: Strip noise from text columns ───

def _clean_text(text: str) -> str:
    """Lower-cases, strips emails, URLs, special chars."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df[Config.SUMMARY_COL] = df[Config.SUMMARY_COL].apply(_clean_text)
    df[Config.BODY_COL]    = df[Config.BODY_COL].apply(_clean_text)
    print(f"  [PREPROCESS] Text columns cleaned (emails, URLs, special chars removed)")
    return df


# ── Step 3: Drop rows with missing targets ──

def drop_missing_targets(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=Config.TARGET_COLS).reset_index(drop=True)
    print(f"  [PREPROCESS] Dropped {before - len(df)} rows with missing target labels")
    return df


# ── Step 4: Drop rare classes ───────────────

def drop_rare_classes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes rows whose Type 2 class has fewer than
    Config.MIN_CLASS_SAMPLES instances. This ensures
    every class has enough data for a train/test split.
    """
    before = len(df)
    counts = df[Config.TYPE2].value_counts()
    valid  = counts[counts >= Config.MIN_CLASS_SAMPLES].index
    df = df[df[Config.TYPE2].isin(valid)].reset_index(drop=True)
    print(f"  [PREPROCESS] Dropped {before - len(df)} rows from rare Type-2 classes")
    return df


# ── Full pipeline wrapper ────────────────────

def run_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[PIPELINE] Running preprocessing...")
    df = remove_duplicates(df)
    df = clean_text_columns(df)
    df = drop_missing_targets(df)
    df = drop_rare_classes(df)
    print(f"[PIPELINE] Preprocessing complete. Final shape: {df.shape}\n")
    return df
