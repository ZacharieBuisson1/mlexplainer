"""BinaryMLExplainer for binary classification tasks.
This module provides an implementation of the BaseMLExplainer for binary classification tasks,
including methods to explain numerical and categorical features using SHAP values.
"""

from typing import Callable, List

import matplotlib.pyplot as plt
from numpy import inf, arange, isclose, nan
from pandas import DataFrame, Series

from mlexplainer.core import BaseMLExplainer
from mlexplainer.explainers.shap.wrapper import ShapWrapper
from mlexplainer.visualization import (
    plot_feature_target_categorical_binary,
    plot_feature_target_numerical_binary,
    plot_shap_values_categorical_binary,
    plot_shap_values_numerical_binary,
)
from mlexplainer.utils.quantiles import group_values, is_in_quantile
from mlexplainer.utils.data_processing import (
    get_index_of_features,
    calculate_min_max_value,
)


class BinaryMLExplainer(BaseMLExplainer):
    """BinaryMLExplainer for binary classification tasks."""

    def __init__(
        self,
        x_train: DataFrame,
        y_train: Series,
        features: List[str],
        model: Callable,
        global_explainer: bool = True,
        local_explainer: bool = True,
    ):
        """
        Initialize the BinaryMLExplainer with training data, features, and model.
        Args:
            x_train (DataFrame): Training feature values.
            y_train (Series): Training target values.
            features (List[str]): List of feature names to interpret.
            model (Callable): The machine learning model to explain.
        Raises:
            ValueError: If x_train or y_train is None, or if features are not provided
                        or not present in x_train.
            ValueError: If any feature in features is not present in x_train.
            ValueError: If no features are provided.
        """
        if y_train.nunique() != 2:
            raise ValueError(
                "y_train must be a binary target variable with exactly two unique values."
            )

        super().__init__(
            x_train,
            y_train,
            features,
            model,
            global_explainer,
            local_explainer,
        )

        self.shap_values_train = ShapWrapper(self.model).calculate(
            dataframe=self.x_train, features=self.features
        )
        self.ymean_train = self.y_train.mean()

    def explain(self, **kwargs):
        """Explain the features for binary classification.
        This method interprets the features based on the training data and SHAP values.
        Args:
            **kwargs: Additional keyword arguments for customization, such as:
                - figsize: Tuple for figure size (default: (15, 8))
                - dpi: Dots per inch for the plot (default: 100)
                - q: Number of quantiles for plotting (default: 20)
        """

        if self.global_explainer:
            # plot a global features importance
            self._explain_global_features(**kwargs)

        if self.local_explainer:
            # check if num features are corrects
            self._explain_numerical(**kwargs)

            # check if cat features are corrects
            self._explain_categorical(**kwargs)

    def correctness_features(
        self,
        q=None,
    ) -> dict:
        """Analyze the correctness of the analysis for every feature.

        This method validates interpretation consistency between actual target rates
        and SHAP values for all features in the explainer.

        Args:
            q (int): Number of quantiles for continuous features. If None, uses adaptive quantiles.

        Returns:
            dict: Dictionary with feature names as keys and correctness results as values.
        """
        results = {}
        for feature in self.features:
            results[feature] = self._validate_feature_interpretation(
                feature, q
            )
        return results

    def _validate_feature_interpretation(
        self,
        feature: str,
        q=None,
    ) -> list:
        """Validate the interpretation consistency between actual target rates and SHAP values for a single feature.

        This method compares feature impact by analyzing:
        - For continuous features: divides values into quantiles and compares target rates vs SHAP values
        - For discrete features: compares target rates by category vs SHAP values
        - Handles missing values as a separate category

        Args:
            feature (str): The feature name to validate.
            q (int): Number of quantiles for continuous features. If None, uses adaptive quantiles.

        Returns:
            list: List of tuples with validation results for each group/category.
        """
        if feature not in self.features:
            raise ValueError(
                f"Feature '{feature}' not found in features list."
            )

        shap_values = self.shap_values_train

        if shap_values is None:
            return False

        # Get feature index for SHAP values
        feature_index = get_index_of_features(self.x_train, feature)
        feature_shap_values = shap_values[:, feature_index]

        # Determine if feature is continuous or discrete
        is_continuous = feature in self.numerical_features

        if is_continuous:
            # For continuous features, use quantile-based grouping
            grouped_data, _ = group_values(
                self.x_train[feature], self.y_train, q
            )

            # Create interpretation dictionaries
            observed_interpretation = {}
            shap_interpretation = {}

            # First, process all groups to get interpretations
            quantiles_values = None
            if not (q is None or self.x_train[feature].nunique() <= 15):
                # Prepare quantile boundaries for later use
                quantiles = arange(
                    1 / len(grouped_data), 1, 1 / len(grouped_data)
                )
                quantiles = [
                    quant for quant in quantiles if not isclose(quant, 1)
                ]
                quantiles_values = list(
                    self.x_train[feature].quantile(quantiles)
                ) + [inf]

            for _, row in grouped_data.iterrows():
                group_val = row["group"]
                target_rate = row["target"]

                # Determine observed interpretation (above/below global mean)
                observed_interpretation[group_val] = (
                    "above"
                    if target_rate > self.ymean_train
                    else (
                        "below"
                        if target_rate < self.ymean_train
                        else "neutral"
                    )
                )

                # Get SHAP values for this group
                if group_val != group_val:  # Check for NaN (missing values)
                    group_mask = self.x_train[feature].isna()
                else:
                    # Find which observations belong to this quantile group
                    if q is None or self.x_train[feature].nunique() <= 15:
                        group_mask = self.x_train[feature] == group_val
                    else:
                        group_mask = (
                            self.x_train[feature].apply(
                                lambda val: is_in_quantile(
                                    val, quantiles_values
                                )
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
            if q is None or self.x_train[feature].nunique() <= 15:
                # For small number of unique values, each value is its own group
                for key in observed_interpretation.keys():
                    if key != key:  # NaN case
                        group_intervals[key] = ("missing", "missing")
                    else:
                        group_intervals[key] = (key, key)
            else:
                # For quantile-based grouping, create interval boundaries
                sorted_groups = sorted(
                    [k for k in observed_interpretation.keys() if k == k]
                )  # Filter out NaN
                for i, group_val in enumerate(sorted_groups):
                    if i == 0:
                        start_val = self.x_train[feature].min()
                    else:
                        start_val = quantiles_values[i - 1]

                    if group_val == inf:
                        end_val = self.x_train[feature].max()
                    else:
                        end_val = group_val

                    group_intervals[group_val] = (start_val, end_val)

                # Handle NaN groups
                for key in observed_interpretation.keys():
                    if key != key:  # NaN case
                        group_intervals[key] = (nan, nan)

        else:
            # For discrete/categorical features, group by unique values
            feature_values = self.x_train[feature]
            unique_values = feature_values.unique()

            observed_interpretation = {}
            shap_interpretation = {}

            for value in unique_values:
                # Calculate target rate for this category
                mask = feature_values == value
                target_rate = self.y_train[mask].mean()
                observed_interpretation[value] = (
                    "above"
                    if target_rate > self.ymean_train
                    else (
                        "below"
                        if target_rate < self.ymean_train
                        else "neutral"
                    )
                )

                # Calculate mean SHAP value for this category
                shap_mean = feature_shap_values[mask].mean()
                shap_interpretation[value] = (
                    "above"
                    if shap_mean > 0
                    else "below" if shap_mean < 0 else "neutral"
                )

        # Compare interpretations and return results with intervals for continuous features
        if is_continuous:
            matches = [
                (
                    round(float(group_intervals[key][0]), 3),  # start_key
                    round(float(group_intervals[key][1]), 3),  # end_key
                    observed_interpretation[key]
                    == shap_interpretation[key],  # is_consistent
                )
                for key in observed_interpretation.keys()
            ]
        else:
            # For discrete features, keep original format
            matches = [
                (key, observed_interpretation[key] == shap_interpretation[key])
                for key in observed_interpretation.keys()
            ]

        return matches

    def _explain_global_features(self, **kwargs):
        """Interpret global features for binary classification."""
        # calculate the absolute value for each features
        absolute_shap_values = DataFrame(
            self.shap_values_train, columns=self.features
        ).apply(abs)

        # calculate the sum of mean of absolute SHAP values
        mean_absolute_shap_values = absolute_shap_values.mean().sum()

        relative_importance = (
            (
                absolute_shap_values.mean().divide(mean_absolute_shap_values)
                * 100
            )
            .reset_index(drop=False)
            .rename(columns={"index": "features", 0: "importances"})
            .sort_values(by="importances", ascending=True)
        )

        figsize = kwargs.get("figsize", (15, 8))
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # plot with horizontal bar chart
        ax.barh(
            relative_importance["features"], relative_importance["importances"]
        )

        # set the title and labels
        ax.set_title(
            "Global Feature Importance for Binary Classification (Mean of the absolute SHAP values)"
        )
        ax.set_xlabel("Relative Importance (%)")
        ax.set_ylabel("Features")

        for _, row in relative_importance.iterrows():
            ax.text(
                row.importances,
                row.features,
                s=" " + str(round(row.importances, 1)) + "%.",
                va="center",
            )

        plt.show()

    def _explain_numerical(self, **kwargs):
        """Interpret numerical features for binary classification."""
        # calculate ymean in train
        for feature in self.numerical_features:
            min_value_train, max_value_train = calculate_min_max_value(
                self.x_train, feature
            )

            # calculate delta
            delta = (max_value_train - min_value_train) / 10

            # Set default values for figsize and dpi if not provided
            figsize = kwargs.get("figsize", (15, 8))
            dpi = kwargs.get("dpi", 100)
            _, ax = plt.subplots(figsize=figsize, dpi=dpi)

            # plot feature target
            q = kwargs.get("q", 20)
            threshold_nb_values = kwargs.get("threshold_nb_values", 15)
            ax = plot_feature_target_numerical_binary(
                self.x_train,
                self.y_train,
                feature,
                q,
                ax,
                delta,
                threshold_nb_values=threshold_nb_values,
            )

            # plot SHAP values
            ax, _ = plot_shap_values_numerical_binary(
                x_train=self.x_train,
                feature=feature,
                shap_values_train=self.shap_values_train,
                delta=delta,
                ymean_train=self.ymean_train,
                ax=ax,
            )

            plt.show()
            plt.close()

    def _explain_categorical(self, **kwargs):
        """Interpret categorical features for binary classification."""
        for feature in self.categorical_features:
            # Set default values for figsize and dpi if not provided
            figsize = kwargs.get("figsize", (15, 8))
            dpi = kwargs.get("dpi", 100)
            _, ax = plt.subplots(figsize=figsize, dpi=dpi)

            color = kwargs.get("color", (0.28, 0.18, 0.71))
            ax = plot_feature_target_categorical_binary(
                self.x_train, self.y_train, feature, ax, color
            )

            ax, _ = plot_shap_values_categorical_binary(
                self.x_train, feature, self.shap_values_train, ax
            )

            plt.show()
            plt.close()
