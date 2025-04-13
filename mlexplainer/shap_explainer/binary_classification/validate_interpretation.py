"""Module to validate the interpretation of a feature based on the target rate and SHAP values."""

from pandas import DataFrame, Series
from numpy import ndarray

from ..utils_shap_explainer.utils import get_index_of_features


def is_interpretation_consistent(
    x_train: DataFrame,
    y_train: Series,
    feature: str,
    shap_values: ndarray = None,
    ymean_train: float = None,
    threshold_validation: float = 0.5,
    max_categories: int = 10,
) -> bool:
    """Validate the interpretation of a feature based on the target rate and SHAP values.

    Args:
        x_train (DataFrame): Training feature values.
        y_train (Series): Training target values.
        feature (str): The feature name to interpret.
        shap_values (ndarray, optional): SHAP values for the training features.
            Defaults to None.
        ymean_train (float, optional): Mean of the training target variable.
            Defaults to None.
        threshold_validation (float, optional): Threshold to validate the interpretation.
            Defaults to 0.5.
        max_categories (int, optional): Maximum number of categories to consider.
            Defaults to 10.

    Returns:
        bool: True if the interpretation is consistent, False otherwise.
    """

    observed_interpretation = None

    if ymean_train is None:
        ymean_train = y_train.mean()

    if x_train[feature].dtype != "category":

        if x_train[feature].nunique() < max_categories:

            # Fill missing values with a placeholder category
            x_train_filled = x_train[feature].fillna("missing")

            # Calculate the mean target rate for each modality
            modality_means = y_train.groupby(x_train_filled).mean()

            # Determine if the target rate for each modality is above or below the target mean
            observed_interpretation = {
                modality: "above" if mean > ymean_train else "below"
                for modality, mean in modality_means.items()
            }

    if shap_values is not None:

        if x_train[feature].nunique() < max_categories:

            # Fill missing values with a placeholder category
            x_train_filled = x_train[feature].fillna("missing")

            index_feature_train = get_index_of_features(x_train, feature)
            shap_means = {
                modality: shap_values[:, index_feature_train][
                    x_train_filled == modality
                ].mean()
                for modality in x_train_filled.unique()
            }
            shap_interpretation = {
                modality: "above" if mean > 0 else "below"
                for modality, mean in shap_means.items()
            }

    if observed_interpretation is not None and shap_values is not None:
        differences = sum(
            observed_interpretation[modality] != shap_interpretation[modality]
            for modality in observed_interpretation
        )
        total = len(observed_interpretation)
        return differences / total <= threshold_validation

    return False
