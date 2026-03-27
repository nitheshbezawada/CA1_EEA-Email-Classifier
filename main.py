#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────
#  main.py  |  Top-Level Controller
#  Orchestrates the full pipeline in strict order:
#    1. Load raw data
#    2. Preprocess (clean / deduplicate / drop rare classes)
#    3. Build TF-IDF feature matrix
#    4. Encapsulate data (train/test split + chained targets)
#    5. Interactive model selection + execution
#
#  Design Choice 1: Chained Multi-Output Classification
#  Each model is trained ONCE and simultaneously predicts
#  all three chain levels:
#    Level 1  →  Type 2
#    Level 2  →  Type 2 + Type 3
#    Level 3  →  Type 2 + Type 3 + Type 4
# ─────────────────────────────────────────────────────────────────

import sys
import random
import numpy as np
from config             import Config
from data_loader        import load_all_data
from preprocessor       import run_pipeline
from feature_builder    import build_tfidf_features
from modelling.email_data import EmailData
from modelling.runner     import execute_model, MODEL_REGISTRY, MODEL_NAMES

# Fix random seeds for reproducibility
random.seed(Config.RANDOM_SEED)
np.random.seed(Config.RANDOM_SEED)


def main():
    print("\n" + "═" * 60)
    print("   Multi-Label Email Classifier  |  Design Choice 1")
    print("   Chained Multi-Output Architecture")
    print("═" * 60)

    # ── Stage 1: Load ────────────────────────────────────────────
    print("\n[STAGE 1] Loading data...")
    df = load_all_data()

    # ── Stage 2: Preprocess ──────────────────────────────────────
    print("[STAGE 2] Preprocessing...")
    df = run_pipeline(df)

    # ── Stage 3: Feature Extraction ──────────────────────────────
    print("[STAGE 3] Building TF-IDF features...")
    X = build_tfidf_features(df)

    # ── Stage 4: Encapsulate Data ─────────────────────────────────
    print("[STAGE 4] Encapsulating data...")
    data = EmailData(X, df)

    # ── Stage 5: Interactive Model Selection ─────────────────────
    while True:
        print("\n" + "─" * 45)
        print("  SELECT A MODEL")
        print("─" * 45)
        for i, name in enumerate(MODEL_NAMES, start=1):
            print(f"  [{i}] {name}")
        print(f"  [{len(MODEL_NAMES) + 1}] Run ALL models")
        print("  [0] Exit")
        print("─" * 45)

        choice = input("  Enter choice: ").strip()

        if choice == "0":
            print("\n  Goodbye!\n")
            sys.exit(0)

        try:
            idx = int(choice)
            if idx == len(MODEL_NAMES) + 1:
                for name in MODEL_NAMES:
                    execute_model(name, data)
                print("\n  All models complete. Check the 'results/' folder.\n")
            elif 1 <= idx <= len(MODEL_NAMES):
                execute_model(MODEL_NAMES[idx - 1], data)
            else:
                print("  Invalid choice — please try again.")
        except ValueError:
            print("  Please enter a number.")


if __name__ == "__main__":
    main()
