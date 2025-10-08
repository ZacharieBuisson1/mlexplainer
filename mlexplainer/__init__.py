"""MLExplainer: Advanced ML explanation library for data scientists using modern frameworks."""

from os import path

# Core abstractions
from .core import BaseMLExplainer, BaseMLPredictor

# SHAP explainers and utilities
from .explainers.shap import (
    ShapWrapper,
    BinaryMLExplainer,
    MultilabelMLExplainer,
)

# Predictors for inference with SHAP contributions
from .predictors import (
    BinaryMLPredictor,
    MultilabelMLPredictor,
)

ROOT_DIR_MODULE = path.dirname(__file__)

__all__ = [
    "BaseMLExplainer",
    "BaseMLPredictor",
    "ShapWrapper",
    "BinaryMLExplainer",
    "MultilabelMLExplainer",
    "BinaryMLPredictor",
    "MultilabelMLPredictor",
]
