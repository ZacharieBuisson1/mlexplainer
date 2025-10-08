"""Core abstractions and base classes for ML explainers and predictors."""

from .base_explainer import BaseMLExplainer
from .base_predictor import BaseMLPredictor

__all__ = ["BaseMLExplainer", "BaseMLPredictor"]
