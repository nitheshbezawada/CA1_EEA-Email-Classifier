# ─────────────────────────────────────────────
#  modelling/runner.py  |  Model Controller
#  Instantiates the chosen model and drives
#  train → predict → evaluate via the abstract
#  BaseClassifier interface.  (Feature 3)
# ─────────────────────────────────────────────

from model.random_forest  import RFClassifier
from model.decision_tree  import DTClassifier
from model.knn_model      import KNNClassifier
from model.neural_net     import MLPClassifier_
from model.logistic_reg   import LogRegClassifier
from model.grad_boost     import GradBoostClassifier
from modelling.email_data import EmailData


# Registry maps menu names → model classes
MODEL_REGISTRY = {
    "RandomForest"     : RFClassifier,
    "DecisionTree"     : DTClassifier,
    "KNN"              : KNNClassifier,
    "NeuralNet"        : MLPClassifier_,
    "LogisticReg"      : LogRegClassifier,
    "GradientBoosting" : GradBoostClassifier,
}

MODEL_NAMES = list(MODEL_REGISTRY.keys())


def execute_model(name: str, data: EmailData) -> None:
    """
    Looks up the model class by name, constructs one
    instance, and runs the full train/predict/evaluate
    pipeline through the uniform BaseClassifier interface.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODEL_REGISTRY)}")

    print(f"\n{'─'*55}")
    print(f"  Running model: {name}")
    print(f"{'─'*55}")

    ModelClass = MODEL_REGISTRY[name]
    model = ModelClass(name)

    model.train(data)
    model.predict(data.get_X_test())
    model.print_results(data)
