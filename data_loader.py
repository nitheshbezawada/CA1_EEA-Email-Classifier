# ─────────────────────────────────────────────
#  data_loader.py  |  Raw Data Ingestion
#  Responsible solely for reading the CSV
#  files from disk and merging them into one
#  unified DataFrame. No preprocessing here.
# ─────────────────────────────────────────────

import os
import pandas as pd
from config import Config


def load_all_data() -> pd.DataFrame:
    """
    Reads every CSV listed in Config.DATA_FILES,
    concatenates them into a single DataFrame,
    resets the index, and returns the result.
    """
    frames = []
    for filename in Config.DATA_FILES:
        path = os.path.join(os.path.dirname(__file__), filename)
        if not os.path.exists(path):
            print(f"  [WARNING] File not found, skipping: {path}")
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        # Keep only the columns we actually need
        needed = [Config.SUMMARY_COL, Config.BODY_COL] + Config.TARGET_COLS
        available = [c for c in needed if c in df.columns]
        frames.append(df[available])
        print(f"  [INFO] Loaded {len(df)} rows from '{filename}'")

    if not frames:
        raise FileNotFoundError("No data files could be loaded.")

    combined = pd.concat(frames, ignore_index=True)
    print(f"  [INFO] Combined dataset: {len(combined)} rows total\n")
    return combined
