# ─────────────────────────────────────────────
#  model/base_classifier.py  |  Abstract Base
#  Defines the uniform interface that EVERY
#  ML model must implement.  This hides all
#  algorithmic differences from the controller.
#  (Feature 3 of the spec)
# ─────────────────────────────────────────────

from abc import ABC, abstractmethod


class BaseClassifier(ABC):

    def __init__(self, name: str) -> None:
        self.name        = name
        self.predictions = None

    @abstractmethod
    def train(self, data) -> None:
        """Fit the model on training data."""

    @abstractmethod
    def predict(self, X_test) -> None:
        """Generate predictions and store in self.predictions."""

    @abstractmethod
    def print_results(self, data) -> None:
        """Compute and display accuracy metrics."""

    @abstractmethod
    def _prepare(self) -> None:
        """Any model-specific setup (called inside __init__)."""
