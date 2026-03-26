# ─────────────────────────────────────────────
#  model/results_mixin.py  |  Shared Evaluation
#  Mixed into every model class to avoid
#  code duplication across the six classifiers.
# ─────────────────────────────────────────────

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


LEVELS = [
    "Level 1  (Type 2 only)",
    "Level 2  (Type 2 + Type 3)",
    "Level 3  (Type 2 + Type 3 + Type 4)",
]
RESULTS_DIR = "results"


class ResultsMixin:
    """
    Provides print_results() for any classifier that sets
    self.name and self.predictions after calling predict().
    """

    def print_results(self, data) -> None:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        y_test = data.get_y_test()
        accuracies = []

        print(f"\n{'═'*55}")
        print(f"  EVALUATION  —  {self.name}")
        print(f"{'═'*55}")

        all_reports = []

        for i, label in enumerate(LEVELS):
            y_true = y_test.iloc[:, i]
            y_pred = self.predictions[:, i]
            acc    = accuracy_score(y_true, y_pred)
            accuracies.append(acc * 100)

            print(f"\n  ▸ {label}")
            print(f"    Accuracy : {acc*100:.2f}%")
            report_str = classification_report(y_true, y_pred, zero_division=0)
            for line in report_str.split("\n"):
                print("    " + line)

            # Confusion matrix image
            col_name = ["Level_1", "Level_2", "Level_3"][i]
            self._save_confusion_matrix(y_true, y_pred, col_name)

            # Collect report rows for CSV
            rpt = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            rpt_df = pd.DataFrame(rpt).T
            rpt_df["chain_level"] = col_name
            all_reports.append(rpt_df)

        # Accuracy trend bar chart
        self._save_accuracy_chart(accuracies)

        # Combined CSV
        pd.concat(all_reports).to_csv(
            os.path.join(RESULTS_DIR, f"{self.name}_metrics.csv")
        )
        print(f"\n  ✔  Results saved to '{RESULTS_DIR}/' folder\n")

    # ── Private helpers ──────────────────────

    def _save_confusion_matrix(self, y_true, y_pred, level_name):
        labels = sorted(y_true.unique())
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(f"{self.name} — {level_name}", fontsize=13)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual",    fontsize=11)
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, f"{self.name}_cm_{level_name}.png")
        plt.savefig(path, dpi=120)
        plt.close()

    def _save_accuracy_chart(self, accuracies):
        short_labels = ["Level 1\n(Type 2)", "Level 2\n(T2+T3)", "Level 3\n(T2+T3+T4)"]
        colours = ["#2E74B5", "#5BA4CF", "#A8C8E8"]
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(short_labels, accuracies, color=colours, edgecolor="white", width=0.5)
        for bar, val in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", va="bottom",
                    fontweight="bold", fontsize=11)
        ax.set_ylim(0, 110)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title(f"{self.name} — Accuracy vs Chain Depth", fontsize=13, pad=12)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, f"{self.name}_accuracy_trend.png")
        plt.savefig(path, dpi=120)
        plt.close()
