import numpy as np
from sklearn.linear_model  import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from model.base_classifier import BaseClassifier
from model.results_mixin   import ResultsMixin
from config import Config


class LogRegClassifier(ResultsMixin, BaseClassifier):
    """
    Three independent Logistic Regression classifiers,
    one per chain level, with LabelEncoder for safe
    string-target handling.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._prepare()

    def _prepare(self) -> None:
        self._models   = []
        self._encoders = []
        for _ in range(3):
            self._models.append(LogisticRegression(
                max_iter     = 1000,
                solver       = 'lbfgs',
                class_weight = 'balanced',
                random_state = Config.RANDOM_SEED,
                C            = 1.0
            ))
            self._encoders.append(LabelEncoder())

    def train(self, data) -> None:
        X = data.get_X_train().astype(np.float64)
        Y = data.get_y_train()
        for i in range(3):
            y_enc = self._encoders[i].fit_transform(Y.iloc[:, i])
            self._models[i].fit(X, y_enc)

    def predict(self, X_test) -> None:
        X = X_test.astype(np.float64)
        cols = []
        for i in range(3):
            y_enc = self._models[i].predict(X)
            y_lbl = self._encoders[i].inverse_transform(y_enc)
            cols.append(y_lbl)
        self.predictions = np.column_stack(cols)
