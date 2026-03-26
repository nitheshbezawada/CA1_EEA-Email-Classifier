from sklearn.tree import DecisionTreeClassifier
from model.base_classifier import BaseClassifier
from model.results_mixin   import ResultsMixin
from config import Config


class DTClassifier(ResultsMixin, BaseClassifier):

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._prepare()

    def _prepare(self) -> None:
        self._mdl = DecisionTreeClassifier(
            max_depth    = 20,
            class_weight = "balanced",
            random_state = Config.RANDOM_SEED
        )

    def train(self, data) -> None:
        self._mdl.fit(data.get_X_train(), data.get_y_train())

    def predict(self, X_test) -> None:
        self.predictions = self._mdl.predict(X_test)
