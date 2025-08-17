"""MLExplainer: Advanced ML explanation library for data scientists using modern frameworks."""

from os import path

# Core abstractions
from .core import BaseMLExplainer

# SHAP explainers and utilities
from .explainers.shap import (
    ShapWrapper,
    BinaryMLExplainer,
    MultilabelMLExplainer,
)

ROOT_DIR_MODULE = path.dirname(__file__)

__all__ = [
    "BaseMLExplainer",
    "ShapWrapper",
    "BinaryMLExplainer",
    "MultilabelMLExplainer",
]
