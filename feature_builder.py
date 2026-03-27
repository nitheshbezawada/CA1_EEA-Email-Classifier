# ─────────────────────────────────────────────
#  feature_builder.py  |  Text → Numbers
#  Converts cleaned text into a TF-IDF matrix.
#  Separate from preprocessing and modelling.
# ─────────────────────────────────────────────

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from config import Config


def build_tfidf_features(df: pd.DataFrame) -> np.ndarray:
    """
    Concatenates Ticket Summary and Interaction Content,
    then fits a TF-IDF vectoriser and returns the matrix X.
    """
    # Combine both text columns into one string per row
    combined_text = (
        df[Config.SUMMARY_COL].fillna("") + " " +
        df[Config.BODY_COL].fillna("")
    ).values.astype(str)

    vectoriser = TfidfVectorizer(
        max_features = Config.TFIDF_MAX_FEATURES,
        ngram_range  = Config.TFIDF_NGRAM_RANGE,
        sublinear_tf = True
    )

    X = vectoriser.fit_transform(combined_text).toarray()
    print(f"[FEATURES] TF-IDF matrix shape: {X.shape}\n")
    return X
