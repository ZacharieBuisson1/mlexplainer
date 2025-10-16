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
            **kwargs (Any): Additional keyword arguments:
                - decimals (int): Number of decimal places to round SHAP contributions (default: 4)

        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'prediction': Predicted probability for positive class (float)
                - 'contributions': Dict mapping feature names to SHAP contributions (rounded)
                - 'values_before_processing': Dict with raw input values
                - 'values_after_processing': Dict with processed values

        Example:
            >>> result = predictor.predict_with_contributions({'age': 35, 'income': 50000})
            >>> {
            ...     'prediction': 0.78,
            ...     'contributions': {'age': 0.1500, 'income': 0.2100, ...},
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
        # Ensure feature names are Python strings (not np.str_) and values are rounded
        contributions = {}
        for i, feature in enumerate(self.features):
            # Convert feature name to Python string
            feature_name = str(feature)

            # Extract scalar value from SHAP values
            value = shap_values_single[i]
            if isinstance(value, np.ndarray):
                value = float(value.item()) if value.size == 1 else float(value.flat[0])
            else:
                value = float(value)

            # Round to specified decimals
            contributions[feature_name] = round(value, decimals)

        return {
            "prediction": float(prediction),
            "contributions": contributions,
            "values_before_processing": values_before,
            "values_after_processing": values_after,
        }

    def predict_with_text_explanation(
        self,
        observation: Union[DataFrame, Dict[str, Any]],
        mode: str = "llm",
        top_n: int = 3,
        language: str = "fr",
        target_name: Optional[str] = None,
        feature_name_mapping: Optional[Dict[str, str]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Make a prediction with SHAP contributions and natural language explanation.

        This method extends predict_with_contributions() by adding a natural language
        explanation of the prediction using either LLM-based or template-based generation.

        Args:
            observation (Union[DataFrame, Dict[str, Any]]): A single observation.
                Can be a dict (e.g., from JSON) or a single-row DataFrame.
            mode (str): Text generation mode ('llm' or 'template'). Default: 'llm'.
            top_n (int): Number of top contributing features to include in explanation. Default: 3.
            language (str): Language for explanation ('fr' or 'en'). Default: 'fr'.
            target_name (Optional[str]): Human-readable name for the target variable.
                Example: 'Exited', 'Churn', 'Default'. If None, uses 'positive class'.
            feature_name_mapping (Optional[Dict[str, str]]): Mapping from technical
                feature names to human-readable names (template mode only). Example:
                {'NumOfProducts': 'nombre de produits', 'Age': 'âge'}
            **kwargs (Any): Additional keyword arguments:
                - decimals (int): Number of decimal places for SHAP contributions (default: 4)
                - max_new_tokens (int): LLM max tokens (default: 300, LLM mode only)
                - temperature (float): LLM sampling temperature (default: 0.3, LLM mode only)

        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'prediction': Predicted probability for positive class (float)
                - 'contributions': Dict mapping feature names to SHAP contributions
                - 'values_before_processing': Dict with raw input values
                - 'values_after_processing': Dict with processed values
                - 'explanation_text': Natural language explanation (str)
                - 'top_features': List of top N features with their info (list of dicts)

        Example:
            >>> result = predictor.predict_with_text_explanation(
            ...     observation={'Age': 42, 'NumOfProducts': 1, 'IsActiveMember': 1},
            ...     mode='llm',
            ...     top_n=3,
            ...     language='fr',
            ...     target_name='Exited'
            ... )
            >>> print(result['explanation_text'])
            La probabilité de sortie (Exited) est de 38%. Cette prédiction s'explique
            principalement par le nombre de produits (valeur: 1, contribution: +53%),
            l'âge du client (valeur: 42 ans, contribution: +24%), et le statut de membre
            actif (valeur: Oui, contribution: -48%).

        Raises:
            ValueError: If mode is not 'llm' or 'template'.
            ImportError: If LLM mode is used but transformers/torch are not installed.
        """
        # Validate mode
        if mode not in ["llm", "template"]:
            raise ValueError(f"mode must be 'llm' or 'template', got: {mode}")

        # Get base prediction with contributions
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

        # Extract top features for metadata
        top_features = text_explainer._extract_top_features(
            result["contributions"], result["values_after_processing"], top_n
        )

        # Generate explanation text
        explanation_text = text_explainer.generate_explanation(
            prediction=result["prediction"],
            contributions=result["contributions"],
            values=result["values_after_processing"],
            top_n=top_n,
            target_name=target_name,
            **kwargs
        )

        # Add text explanation and top features to result
        result["explanation_text"] = explanation_text
        result["top_features"] = top_features

        return result
