# EEA-CA1: Multi-Label Email Classification
**National College of Ireland** | PGDAI_SEP24 / MSCAI1 / MSCAI1B

---

## Team Members
| Name | Student ID |
| [Jaswanth Dasu] | [24335380] |
| [Nithesh Kumar Bezawada] | [24230910] |
| [Nikhil reddy Mandapati] | [24266426] |

**GitHub:** `https://github.com/nitheshbezawada/CA1_EEA-Email-Classifier`

---

## Overview
Multi-label email classification system built using two architectural approaches:
- **Design Choice 1 — Chained Multi-Output:** One model predicts Type 2, Type 2+3, and Type 2+3+4 simultaneously.
- **Design Choice 2 — Hierarchical Modelling:** Cascading model instances where each level's prediction filters data for the next.

---

## Project Structure
```
EEA-CA1/
├── main.py                  # Entry point + interactive CLI
├── config.py                # All shared constants
├── data_loader.py           # Loads CSVs
├── preprocessor.py          # Cleans text data
├── feature_builder.py       # TF-IDF vectorisation
├── modelling/
│   ├── email_data.py        # Data encapsulator
│   └── runner.py            # Model controller
├── model/
│   ├── base_classifier.py   # Abstract base class
│   ├── results_mixin.py     # Shared evaluation logic
│   ├── random_forest.py
│   ├── decision_tree.py
│   ├── knn_model.py
│   ├── neural_net.py
│   ├── logistic_reg.py
│   └── grad_boost.py
├── AppGallery.csv
└── Purchasing.csv
```

---

## Setup & Run

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
python main.py
```

Select a model from the interactive menu (1–6) or enter `7` to run all models. Results are saved automatically to `results/`.

---

## Results

| Model | Level 1 | Level 2 | Level 3 |
|-------|:-------:|:-------:|:-------:|
| Random Forest | 78.79% | 72.73% | 57.58% |
| Decision Tree | 75.76% | 66.67% | 48.48% |
| KNN | 72.73% | 63.64% | 51.52% |
| Neural Network | 60.61% | 66.67% | 42.42% |
| Logistic Regression | 78.79% | 57.58% | 36.36% |
| **Gradient Boosting** | **78.79%** | **72.73%** | **63.64% ** |

> Accuracy drops at each deeper level as the concatenated label space grows. Gradient Boosting achieved the best Level-3 accuracy.

## Adding a New Model
1. Create `model/my_model.py` inheriting from `ResultsMixin` and `BaseClassifier`
2. Implement `_prepare()`, `train()`, and `predict()`
3. Register it in `modelling/runner.py` — no other files need changing
