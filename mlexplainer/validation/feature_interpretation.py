"""Feature interpretation validation functions.

This module provides core validation functions to check interpretation consistency
between actual target rates and SHAP values for different feature types.
"""

from typing import Any, List, Optional, Tuple, Union

from numpy import inf, arange, isclose
from pandas import DataFrame, Series

from mlexplainer.utils.quantiles import group_values, is_in_quantile


def validate_single_feature_interpretation(
    x_train: DataFrame,
    y_binary: Series,
    feature: str,
    feature_shap_values: Series,
    numerical_features: List[str],
    ymean_binary: float,
    q: Optional[int] = None,
    threshold_nb_values: int = 15,
) -> List[Union[Tuple[str, str, bool], Tuple[Any, bool]]]:
    """Validate interpretation consistency between actual target rates and SHAP
    values for a single feature.

    This function compares feature impact by analyzing:
    - For continuous features: divides values into quantiles and compares target
        rates vs SHAP values
    - For discrete features: compares target rates by category vs SHAP values
    - Handles missing values as a separate category

    Args:
        x_train (DataFrame): Training feature values.
        y_binary (Series): Binary target values (0/1).
        feature (str): The feature name to validate.
        feature_shap_values (Series): SHAP values for the specific feature.
        numerical_features (List[str]): List of numerical feature names.
        ymean_binary (float): Mean of the binary target values.
        q (int): Number of quantiles for continuous features.
            If None, uses adaptive quantiles. Defaults to None.
        threshold_nb_values (int): Threshold for number of unique values to decide
            grouping method. Defaults to 15.

    Returns:
        List[Tuple]: List of tuples with validation results for each group/category.
            For continuous features: (start_interval, end_interval, is_consistent)
            For categorical features: (category_value, is_consistent)
    """
    # Determine if feature is continuous or discrete
    is_continuous = feature in numerical_features

    if is_continuous:
        # For continuous features, use quantile-based grouping
        grouped_data, _ = group_values(x_train[feature], y_binary, q)

        # Create interpretation dictionaries
        observed_interpretation = {}
        shap_interpretation = {}

        # First, process all groups to get interpretations
        quantiles_values: list[Union[int, float]] = []
        if not (
            q is None or x_train[feature].nunique() <= threshold_nb_values
        ):
            # Prepare quantile boundaries for later use
            quantiles = arange(1 / len(grouped_data), 1, 1 / len(grouped_data))
            quantiles = quantiles[
                [not isclose(quant, 1) for quant in quantiles]
            ].flatten()
            quantiles_values = list(x_train[feature].quantile(quantiles)) + [
                inf
            ]

        for _, row in grouped_data.iterrows():
            group_val = row["group"]
            target_rate = row["target"]

            # Determine observed interpretation (above/below global mean)
            observed_interpretation[group_val] = (
                "above"
                if target_rate > ymean_binary
                else ("below" if target_rate < ymean_binary else "neutral")
            )

            # Get SHAP values for this group
            if group_val != group_val:  # Check for NaN (missing values)
                group_mask = x_train[feature].isna()
            else:
                # Find which observations belong to this quantile group
                if (
                    q is None
                    or x_train[feature].nunique() <= threshold_nb_values
                ):
                    group_mask = x_train[feature] == group_val
                else:
                    group_mask = (
                        x_train[feature].apply(
                            lambda val: is_in_quantile(val, quantiles_values)
                        )
                        == group_val
                    )

            # Calculate mean SHAP value for this group
            if feature_shap_values[group_mask].shape[0] == 0:
                group_shap_mean = 0
            else:
                group_shap_mean = feature_shap_values[group_mask].mean()
            shap_interpretation[group_val] = (
                "above"
                if group_shap_mean > 0
                else "below" if group_shap_mean < 0 else "neutral"
            )

        # Now create interval mapping for all processed groups
        group_intervals = {}
        if q is None or x_train[feature].nunique() <= threshold_nb_values:
            # For small number of unique values, each value is its own group
            for key in observed_interpretation:
                if key != key:  # NaN case
                    group_intervals[key] = ("missing", "missing")
                else:
                    group_intervals[key] = (key, key)
        else:
            # For quantile-based grouping, create interval boundaries
            sorted_groups = sorted(
                [k for k in observed_interpretation if k == k]
            )  # Filter out NaN
            for i, group_val in enumerate(sorted_groups):
                if i == 0:
                    start_val = x_train[feature].min()
                else:
                    start_val = quantiles_values[i - 1]

                if group_val == inf:
                    end_val = x_train[feature].max()
                else:
                    end_val = group_val

                group_intervals[group_val] = (start_val, end_val)

            # Handle NaN groups
            for key in observed_interpretation:
                if key != key:  # NaN case
                    group_intervals[key] = ("missing", "missing")

        # Compare interpretations and return results with intervals for continuous features
        matches = []
        for key, obs_interp in observed_interpretation.items():
            start_val, end_val = group_intervals[key]
            if start_val == "missing" and end_val == "missing":
                matches.append(
                    (
                        "missing",
                        "missing",
                        obs_interp == shap_interpretation[key],
                    )
                )
            else:
                matches.append(
                    (
                        str(round(float(start_val), 3)),
                        str(round(float(end_val), 3)),
                        obs_interp == shap_interpretation[key],
                    )
                )

    else:
        # For discrete/categorical features, group by unique values
        feature_values = x_train[feature]
        unique_values = feature_values.unique()

        observed_interpretation = {}
        shap_interpretation = {}

        for value in unique_values:
            # Calculate target rate for this category
            mask = feature_values == value
            target_rate = y_binary[mask].mean()
            observed_interpretation[value] = (
                "above"
                if target_rate > ymean_binary
                else ("below" if target_rate < ymean_binary else "neutral")
            )

            # Calculate mean SHAP value for this category
            shap_mean = feature_shap_values[mask].mean()
            shap_interpretation[value] = (
                "above"
                if shap_mean > 0
                else "below" if shap_mean < 0 else "neutral"
            )

        # For discrete features, keep original format
        matches = [
            (key, observed_interpretation[key] == shap_interpretation[key])
            for key in observed_interpretation
        ]

    return matches
