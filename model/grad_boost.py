import numpy as np
from sklearn.ensemble      import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from model.base_classifier import BaseClassifier
from model.results_mixin   import ResultsMixin
from config import Config


class GradBoostClassifier(ResultsMixin, BaseClassifier):
    """
    Three independent Gradient Boosting classifiers,
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
            self._models.append(GradientBoostingClassifier(
                n_estimators      = 200,
                learning_rate     = 0.1,
                max_depth         = 4,
                subsample         = 0.8,
                random_state      = Config.RANDOM_SEED
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
