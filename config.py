# ─────────────────────────────────────────────
#  config.py  |  Central Configuration Store
#  All shared constants are defined here so
#  that no other file needs hard-coded values.
# ─────────────────────────────────────────────

class Config:
    # ── Raw data files ──────────────────────
    DATA_FILES = ["AppGallery.csv", "Purchasing.csv"]

    # ── Text feature columns ────────────────
    SUMMARY_COL    = "Ticket Summary"
    BODY_COL       = "Interaction content"

    # ── Classification target columns ───────
    # Type 1 always has a single class per file → ignored
    TYPE2 = "Type 2"
    TYPE3 = "Type 3"
    TYPE4 = "Type 4"
    TARGET_COLS = [TYPE2, TYPE3, TYPE4]

    # ── Min samples required per class ──────
    # Classes with fewer instances are dropped
    MIN_CLASS_SAMPLES = 3

    # ── TF-IDF settings ─────────────────────
    TFIDF_MAX_FEATURES = 1000
    TFIDF_NGRAM_RANGE  = (1, 2)

    # ── Train / test split ──────────────────
    TEST_SIZE   = 0.2
    RANDOM_SEED = 42
