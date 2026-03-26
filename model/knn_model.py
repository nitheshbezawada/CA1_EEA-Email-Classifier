from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from model.base_classifier import BaseClassifier
from model.results_mixin   import ResultsMixin


class KNNClassifier(ResultsMixin, BaseClassifier):

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._prepare()

    def _prepare(self) -> None:
        # KNN does not natively support multi-output,
        # so we wrap it with MultiOutputClassifier.
        self._mdl = MultiOutputClassifier(
            KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
            n_jobs=-1
        )

    def train(self, data) -> None:
        self._mdl.fit(data.get_X_train(), data.get_y_train())

    def predict(self, X_test) -> None:
        import numpy as np
        raw = self._mdl.predict(X_test)
        # MultiOutputClassifier returns shape (n_outputs, n_samples) or (n_samples, n_outputs)
        arr = np.array(raw)
        self.predictions = arr.T if arr.shape[0] == 3 else arr
