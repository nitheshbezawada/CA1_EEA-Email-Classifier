# ─────────────────────────────────────────────
#  modelling/email_data.py  |  Data Capsule
#  Encapsulates training and testing arrays so
#  that every ML model receives an identical
#  interface regardless of how the targets are
#  structured.  (Feature 2 of the spec)
# ─────────────────────────────────────────────

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from config import Config


class EmailData:
    """
    Holds X_train, X_test, y_train, y_test for
    Design Choice 1 (Chained Multi-Output).

    The target matrix Y has three columns:
      · Level 1  →  Type 2
      · Level 2  →  Type 2 + "_" + Type 3
      · Level 3  →  Type 2 + "_" + Type 3 + "_" + Type 4
    """

    def __init__(self, X: np.ndarray, df: pd.DataFrame) -> None:
        self._raw_df = df.copy()

        # Fill any remaining NaNs in target columns
        for col in Config.TARGET_COLS:
            self._raw_df[col] = self._raw_df[col].fillna("Unknown")

        # ── Build the three chained target levels ──
        t2 = self._raw_df[Config.TYPE2]
        t3 = self._raw_df[Config.TYPE3]
        t4 = self._raw_df[Config.TYPE4]

        lvl1 = t2
        lvl2 = t2 + "_" + t3
        lvl3 = t2 + "_" + t3 + "_" + t4

        self._Y = pd.concat([lvl1, lvl2, lvl3], axis=1)
        self._Y.columns = ["Level_1", "Level_2", "Level_3"]

        # ── Train / test split ─────────────────────
        (self.X_train, self.X_test,
         self.y_train, self.y_test,
         self.train_df, self.test_df) = train_test_split(
            X, self._Y, self._raw_df,
            test_size   = Config.TEST_SIZE,
            random_state= Config.RANDOM_SEED
        )

        print(f"[DATA] Train size: {len(self.X_train)}  |  Test size: {len(self.X_test)}")
        print(f"[DATA] Target levels: {list(self._Y.columns)}\n")

    # ── Getters ───────────────────────────────
    def get_X_train(self):  return self.X_train
    def get_X_test(self):   return self.X_test
    def get_y_train(self):  return self.y_train
    def get_y_test(self):   return self.y_test
    def get_full_y(self):   return self._Y
    def get_train_df(self): return self.train_df
    def get_test_df(self):  return self.test_df
