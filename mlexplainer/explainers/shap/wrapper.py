"""Shap Wrapper for Models."""

from typing import Callable

from pandas import DataFrame
from shap import TreeExplainer


class ShapWrapper:
    """Shapley's values wrapper for models, based on TreeExplainer.
    This class is designed to calculate SHAP values for a given model and
    features in a DataFrame. It uses the TreeExplainer from the SHAP library
    to compute the SHAP values based on the model's predictions.

    Attributes:
        model (Callable): The model to be wrapped for SHAP value calculation.
        model_output (str): The type of output from the model, e.g.,
            "raw", "probability".
        shap_margin_explainer (TreeExplainer): The SHAP explainer instance.
    """

    def __init__(self, model: Callable, model_output: str = "raw"):
        """Initialize the ShapWrapper with a model.

        Args:
            model (Callable): The model to be wrapped for SHAP value calculation.
            model_output (str): The type of output from the model, e.g.,
                "raw", "probability".
        """
        self.model = model
        self.model_output = model_output

        self.shap_margin_explainer = TreeExplainer(
            model=self.model, model_output=self.model_output
        )

    def calculate(
        self, dataframe: DataFrame, features: list[str]
    ) -> DataFrame:
        """Calculate SHAP values for the given model and dataframe.

        Args:
            dataframe (DataFrame): The input DataFrame containing features.
            features (list[str]): List of feature names to calculate SHAP values for.

        Returns:
            DataFrame: A DataFrame containing SHAP values for the specified features.
        """
        # Example: calculate SHAP values for X using the model
        shap_values = self.shap_margin_explainer.shap_values(
            dataframe[features]
        )
        return shap_values

