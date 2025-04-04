from pandas import DataFrame, Series
import shap


def calculate_shap_values(
    model: object,
    x_train: DataFrame,
) -> tuple:
    """Calculate SHAP values for a given feature using the provided model.

    Args:
        model (object): The trained model to use for SHAP value calculation.
        x_train (DataFrame): Training feature values.
        feature (str): The feature name to calculate SHAP values for.

    Returns:
        tuple: A tuple containing SHAP values for training and test sets.
    """
    # Ensure the feature is present in both training and test sets

    explainer = shap.Explainer(model)
    shap_values = explainer(x_train)

    return shap_values
