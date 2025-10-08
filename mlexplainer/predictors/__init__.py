"""Predictors module for single observation predictions with SHAP contributions.

This module provides predictors for calculating SHAP-based feature contributions
for individual observations, complementing the explainers module which focuses on
global model interpretation.
"""

from .binary_predictor import BinaryMLPredictor
from .multilabel_predictor import MultilabelMLPredictor

__all__ = [
    "BinaryMLPredictor",
    "MultilabelMLPredictor",
]
