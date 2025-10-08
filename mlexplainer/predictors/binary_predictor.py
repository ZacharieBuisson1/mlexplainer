"""BinaryMLPredictor for binary classification with SHAP contributions.

This module provides an implementation of BaseMLPredictor for binary classification,
calculating predictions and SHAP-based feature contributions for individual observations.
"""

from typing import Any, Callable, Dict, Optional, Union

import numpy as np
from pandas import DataFrame
from sklearn.pipeline import Pipeline

from mlexplainer.core import BaseMLPredictor
from mlexplainer.explainers.shap.wrapper import ShapWrapper


class BinaryMLPredictor(BaseMLPredictor):
    """BinaryMLPredictor for binary classification with SHAP contributions.

    This class extends BaseMLPredictor to provide prediction and contribution
    calculation for binary classification tasks. It calculates SHAP values
    to explain how each feature contributes to the predicted probability.

    Usage:
        >>> predictor = BinaryMLPredictor(model, x_train)
        >>> observation = {'age': 35, 'income': 50000}
        >>> result = predictor.predict_with_contributions(observation)
        >>> print(result['prediction'])  # 0.78
        >>> print(result['contributions'])  # {'age': 0.15, 'income': 0.21, ...}
    """

    def __init__(
        self,
        model: Callable,
        x_train: DataFrame,
        pipeline: Optional[Pipeline] = None,
    ):
        """Initialize the BinaryMLPredictor.

        Args:
            model (Callable): The binary classification model.
            x_train (DataFrame): Training feature values (processed data).
            pipeline (Optional[Pipeline]): Optional scikit-learn Pipeline for preprocessing.

        Raises:
            ValueError: If x_train is None or features cannot be extracted.
        """
        super().__init__(model, x_train, pipeline)

        # Initialize SHAP wrapper
        self.shap_wrapper = ShapWrapper(self.model, model_output="raw")

    def predict_with_contributions(
        self, observation: Union[DataFrame, Dict[str, Any]], **kwargs: Any
    ) -> Dict[str, Any]:
        """Make a prediction and calculate SHAP contributions for a single observation.

        Args:
            observation (Union[DataFrame, Dict[str, Any]]): A single observation.
                Can be a dict (e.g., from JSON) or a single-row DataFrame.
            **kwargs (Any): Additional keyword arguments (unused, for compatibility).

        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'prediction': Predicted probability for positive class (float)
                - 'contributions': Dict mapping feature names to SHAP contributions
                - 'values_before_processing': Dict with raw input values
                - 'values_after_processing': Dict with processed values

        Example:
            >>> result = predictor.predict_with_contributions({'age': 35, 'income': 50000})
            >>> {
            ...     'prediction': 0.78,
            ...     'contributions': {'age': 0.15, 'income': 0.21, ...},
            ...     'values_before_processing': {'age': 35, 'income': 50000},
            ...     'values_after_processing': {'age': 35, 'income': 50000}
            ... }
        """
        # Prepare observation (handle dict/DataFrame, apply pipeline, convert types)
        observation_processed, values_before, values_after = self._prepare_observation(observation)

        # Calculate SHAP values using standard calculate method
        shap_values = self.shap_wrapper.calculate(
            observation_processed, self.features
        )

        # Get model prediction
        prediction = self.model.predict_proba(observation_processed[self.features])[
            0, 1
        ]

        # Handle SHAP values format
        # Binary case: matrix (n_features, 1) -> take [0] (first observation)
        # List format (multiclass): list of matrices -> take [-1][0] (positive class, first obs)
        if isinstance(shap_values, list):
            shap_values_single = shap_values[-1][0]  # Positive class, first observation
        else:
            shap_values_single = shap_values[0]  # First observation

        # Build contributions dictionary
        # Ensure each value is a scalar by using np.asarray().item() if needed
        contributions = {}
        for i, feature in enumerate(self.features):
            value = shap_values_single[i]
            # If value is still an array, extract the scalar
            if isinstance(value, np.ndarray):
                contributions[feature] = float(value.item()) if value.size == 1 else float(value.flat[0])
            else:
                contributions[feature] = float(value)

        return {
            "prediction": float(prediction),
            "contributions": contributions,
            "values_before_processing": values_before,
            "values_after_processing": values_after,
        }
