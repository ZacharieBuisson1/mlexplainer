"""MultilabelMLPredictor for multilabel classification with SHAP contributions.

This module provides an implementation of BaseMLPredictor for multilabel classification,
calculating predictions and SHAP-based feature contributions for individual observations
across multiple labels.
"""

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from pandas import DataFrame
from sklearn.pipeline import Pipeline

from mlexplainer.core import BaseMLPredictor
from mlexplainer.explainers.shap.wrapper import ShapWrapper


class MultilabelMLPredictor(BaseMLPredictor):
    """MultilabelMLPredictor for multilabel classification with SHAP contributions.

    This class extends BaseMLPredictor to provide prediction and contribution
    calculation for multilabel classification tasks. It calculates SHAP values
    per label to explain how each feature contributes to each label's prediction.

    Usage:
        >>> predictor = MultilabelMLPredictor(model, x_train, label_names=['A', 'B', 'C'])
        >>> observation = {'age': 35, 'income': 50000}
        >>> result = predictor.predict_with_contributions(observation)
        >>> print(result['label_A']['prediction'])  # 0.78
        >>> print(result['label_B']['contributions'])  # {'age': 0.15, ...}
    """

    def __init__(
        self,
        model: Callable,
        x_train: DataFrame,
        label_names: Optional[List[str]] = None,
        pipeline: Optional[Pipeline] = None,
    ):
        """Initialize the MultilabelMLPredictor.

        Args:
            model (Callable): The multilabel classification model.
            x_train (DataFrame): Training feature values (processed data).
            label_names (Optional[List[str]]): Names of the output labels/classes.
                If None, will use numeric indices (label_0, label_1, ...).
                Note: These are the TARGET labels (outputs), not input features.
            pipeline (Optional[Pipeline]): Optional scikit-learn Pipeline for preprocessing.

        Raises:
            ValueError: If x_train is None or features cannot be extracted.
        """
        super().__init__(model, x_train, pipeline)

        # Initialize SHAP wrapper
        self.shap_wrapper = ShapWrapper(self.model, model_output="raw")

        # Set label names (these are output labels, not input features)
        self.label_names = label_names

    def predict_with_contributions(
        self, observation: Union[DataFrame, Dict[str, Any]], **kwargs: Any
    ) -> Dict[str, Dict[str, Any]]:
        """Make predictions and calculate SHAP contributions for all labels.

        Args:
            observation (Union[DataFrame, Dict[str, Any]]): A single observation.
                Can be a dict (e.g., from JSON) or a single-row DataFrame.
            **kwargs (Any): Additional keyword arguments (unused, for compatibility).

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary with label names as keys, each containing:
                - 'prediction': Predicted probability for the label (float)
                - 'contributions': Dict mapping feature names to SHAP contributions
                - 'values_before_processing': Dict with raw input values
                - 'values_after_processing': Dict with processed values

        Example:
            >>> result = predictor.predict_with_contributions({'age': 35, 'income': 50000})
            >>> {
            ...     'label_A': {
            ...         'prediction': 0.78,
            ...         'contributions': {'age': 0.15, 'income': 0.21}
            ...     },
            ...     'label_B': {
            ...         'prediction': 0.42,
            ...         'contributions': {'age': -0.05, 'income': 0.10}
            ...     },
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

        # Get model predictions for all labels
        predictions = self.model.predict_proba(observation_processed[self.features])

        # Handle different prediction formats
        # For multilabel: predictions is typically a list of arrays (one per label)
        # or a 2D array with shape (n_samples, n_classes) for multiclass
        if isinstance(predictions, list):
            # List of arrays, one per label (common in multilabel)
            n_labels = len(predictions)
            predictions_array = np.array([pred[0, 1] for pred in predictions])
        elif len(predictions.shape) == 2:
            # 2D array (n_samples, n_classes)
            n_labels = predictions.shape[1]
            predictions_array = predictions[0, :]
        else:
            raise ValueError(
                f"Unexpected prediction shape: {predictions.shape}. "
                "Expected list of arrays or 2D array."
            )

        # Determine label names
        if self.label_names is None:
            label_names = [f"label_{i}" for i in range(n_labels)]
        else:
            if len(self.label_names) != n_labels:
                raise ValueError(
                    f"Number of label_names ({len(self.label_names)}) does not match "
                    f"number of model outputs ({n_labels})"
                )
            label_names = self.label_names

        # Handle SHAP values format
        # For multiclass: either list of matrices or 2D array
        # List format: one matrix per class, each with shape (1, n_features) or (n_features,)
        # Array format: shape (1, n_features, n_classes) or (n_features, n_classes)
        if isinstance(shap_values, list):
            # List format: one matrix per class, take [0] for first observation
            shap_values_per_label = [shap_class[0] for shap_class in shap_values]
        elif isinstance(shap_values, np.ndarray):
            # Array format: take first observation [0] and transpose to get per-label arrays
            if len(shap_values.shape) == 2:
                # Shape: (1, n_features) for single class or (n_features, n_classes) for multiclass
                if shap_values.shape[0] == 1:
                    # Single observation, multiple features: (1, n_features)
                    # This is actually single-class binary, not multiclass
                    shap_values_per_label = [shap_values[0]]
                else:
                    # Multiple features, multiple classes: (n_features, n_classes)
                    shap_values_per_label = [shap_values[:, i] for i in range(n_labels)]
            elif len(shap_values.shape) == 3:
                # Shape: (1, n_features, n_classes)
                shap_values_per_label = [shap_values[0, :, i] for i in range(n_labels)]
            else:
                # Shape: (n_features,) - single class
                shap_values_per_label = [shap_values]
        else:
            raise ValueError(
                f"Unexpected SHAP values format for multilabel: {type(shap_values)}. "
                "Expected list or numpy array."
            )

        # Build result dictionary for each label
        results = {}

        for i, label_name in enumerate(label_names):
            shap_values_label = shap_values_per_label[i]
            prediction_label = predictions_array[i]

            # Build contributions dictionary
            contributions = {
                feature: float(shap_values_label[j])
                for j, feature in enumerate(self.features)
            }

            results[label_name] = {
                "prediction": float(prediction_label),
                "contributions": contributions,
            }

        # Add processing values at the root level (not per label)
        results["values_before_processing"] = values_before
        results["values_after_processing"] = values_after

        return results
