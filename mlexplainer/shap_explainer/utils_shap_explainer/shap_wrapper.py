"""Shap Wrapper for Models."""

from pandas import DataFrame
from shap import TreeExplainer


def calculate_shap_values_tree_models(
    model: object,
    x_train: DataFrame,
    selected_features: list[str],
    model_output: str = "raw",
) -> tuple:
    """Calculate SHAP values for a given feature using the provided model.

    Args:
        model (object): The trained model to use for SHAP value calculation.
        x_train (DataFrame): Training feature values.
        selected_features (list[str]): List of selected feature names.
        feature (str): The feature name to calculate SHAP values for.

    Returns:
        tuple: A tuple containing SHAP values for training and test sets.
    """
    # Ensure the feature is present in both training and test sets
    shap_margin_explainer = TreeExplainer(
        model=model, model_output=model_output
    )
    shap_values = shap_margin_explainer.shap_values(x_train[selected_features])

    return shap_values
