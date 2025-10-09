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
            **kwargs (Any): Additional keyword arguments:
                - decimals (int): Number of decimal places to round SHAP contributions (default: 4)

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary with label names as keys, each containing:
                - 'prediction': Predicted probability for the label (float)
                - 'contributions': Dict mapping feature names to SHAP contributions (rounded)
                - 'values_before_processing': Dict with raw input values
                - 'values_after_processing': Dict with processed values

        Example:
            >>> result = predictor.predict_with_contributions({'age': 35, 'income': 50000})
            >>> {
            ...     'label_A': {
            ...         'prediction': 0.78,
            ...         'contributions': {'age': 0.1500, 'income': 0.2100}
            ...     },
            ...     'label_B': {
            ...         'prediction': 0.42,
            ...         'contributions': {'age': -0.0500, 'income': 0.1000}
            ...     },
            ...     'values_before_processing': {'age': 35, 'income': 50000},
            ...     'values_after_processing': {'age': 35, 'income': 50000}
            ... }
        """
        # Get rounding precision from kwargs (default: 4 decimals)
        decimals = kwargs.get("decimals", 4)

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
            # Ensure feature names are Python strings (not np.str_) and values are rounded
            contributions = {}
            for j, feature in enumerate(self.features):
                # Convert feature name to Python string
                feature_name = str(feature)
                # Extract and round SHAP value
                value = float(shap_values_label[j])
                contributions[feature_name] = round(value, decimals)

            results[label_name] = {
                "prediction": float(prediction_label),
                "contributions": contributions,
            }

        # Add processing values at the root level (not per label)
        results["values_before_processing"] = values_before
        results["values_after_processing"] = values_after

        return results

    def predict_with_text_explanation(
        self,
        observation: Union[DataFrame, Dict[str, Any]],
        mode: str = "llm",
        top_n: int = 3,
        language: str = "fr",
        feature_name_mapping: Optional[Dict[str, str]] = None,
        **kwargs: Any
    ) -> Dict[str, Dict[str, Any]]:
        """Make predictions with SHAP contributions and natural language explanations.

        This method extends predict_with_contributions() by adding natural language
        explanations for each label using either LLM-based or template-based generation.

        Args:
            observation (Union[DataFrame, Dict[str, Any]]): A single observation.
                Can be a dict (e.g., from JSON) or a single-row DataFrame.
            mode (str): Text generation mode ('llm' or 'template'). Default: 'llm'.
            top_n (int): Number of top contributing features per label. Default: 3.
            language (str): Language for explanations ('fr' or 'en'). Default: 'fr'.
            feature_name_mapping (Optional[Dict[str, str]]): Mapping from technical
                feature names to human-readable names (template mode only). Example:
                {'NumOfProducts': 'nombre de produits', 'Age': 'âge'}
            **kwargs (Any): Additional keyword arguments:
                - decimals (int): Number of decimal places for SHAP contributions (default: 4)
                - max_new_tokens (int): LLM max tokens (default: 300, LLM mode only)
                - temperature (float): LLM sampling temperature (default: 0.3, LLM mode only)

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary with label names as keys, each containing:
                - 'prediction': Predicted probability for the label (float)
                - 'contributions': Dict mapping feature names to SHAP contributions
                - 'explanation_text': Natural language explanation (str)
                - 'top_features': List of top N features (list of dicts)
                Plus shared keys at root level:
                - 'values_before_processing': Dict with raw input values
                - 'values_after_processing': Dict with processed values

        Example:
            >>> result = predictor.predict_with_text_explanation(
            ...     observation={'Age': 42, 'NumOfProducts': 1},
            ...     mode='llm',
            ...     top_n=3,
            ...     language='fr'
            ... )
            >>> print(result['label_A']['explanation_text'])
            La probabilité de label_A est de 75%. Cette prédiction s'explique
            principalement par...

        Raises:
            ValueError: If mode is not 'llm' or 'template'.
            ImportError: If LLM mode is used but transformers/torch are not installed.
        """
        # Validate mode
        if mode not in ["llm", "template"]:
            raise ValueError(f"mode must be 'llm' or 'template', got: {mode}")

        # Get base predictions with contributions
        result = self.predict_with_contributions(observation, **kwargs)

        # Initialize text explainer based on mode
        if mode == "llm":
            from mlexplainer.interpretation import TextExplainerLLM

            text_explainer = TextExplainerLLM(language=language)
        else:  # template mode
            from mlexplainer.interpretation import TextExplainerTemplate

            text_explainer = TextExplainerTemplate(
                language=language, feature_name_mapping=feature_name_mapping
            )

        # Extract values (shared across all labels)
        values_before = result.pop("values_before_processing")
        values_after = result.pop("values_after_processing")

        # Generate explanation for each label
        for label_name, label_data in result.items():
            # Extract top features for this label
            top_features = text_explainer._extract_top_features(
                label_data["contributions"], values_after, top_n
            )

            # Generate explanation text
            explanation_text = text_explainer.generate_explanation(
                prediction=label_data["prediction"],
                contributions=label_data["contributions"],
                values=values_after,
                top_n=top_n,
                target_name=label_name,
                **kwargs
            )

            # Add to label data
            label_data["explanation_text"] = explanation_text
            label_data["top_features"] = top_features

        # Add values back at root level
        result["values_before_processing"] = values_before
        result["values_after_processing"] = values_after

        return result
