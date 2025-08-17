"""BinaryMLExplainer for binary classification tasks.
This module provides an implementation of the BaseMLExplainer for binary classification tasks,
including methods to explain numerical and categorical features using SHAP values.
"""

from typing import Callable, List, Optional, Union

import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from streamlit import pyplot

from mlexplainer.core import BaseMLExplainer
from mlexplainer.explainers.shap.wrapper import ShapWrapper
from mlexplainer.visualization import (
    plot_feature_target_categorical_binary,
    plot_feature_target_numerical_binary,
    plot_shap_values_categorical_binary,
    plot_shap_values_numerical_binary,
)
from mlexplainer.utils.data_processing import (
    get_index_of_features,
    calculate_min_max_value,
)
from mlexplainer.validation.feature_interpretation import (
    validate_single_feature_interpretation,
)


class BinaryMLExplainer(BaseMLExplainer):
    """BinaryMLExplainer for binary classification tasks.
    This class extends BaseMLExplainer to provide specific methods for explaining
    features in binary classification tasks using SHAP values.
    It includes methods to interpret numerical and categorical features,
    validate feature interpretations, and visualize global feature importance.
    """

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
            global_explainer (bool): Whether to use a global explainer.
                                        Defaults to True.
            local_explainer (bool): Whether to use a local explainer.
                                        Defaults to True.

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

    def explain(
        self, features_to_explain: Union[list[str], None] = None, **kwargs
    ):
        """Explain the features for binary classification.
        This method interprets the features based on the training data and SHAP values.
        It visualizes global feature importance and interprets numerical
        and categorical features.

        Args:
            features_to_explain (Union[list[str], None]): List of feature names to explain.
            **kwargs: Additional keyword arguments for customization, such as:
                - figsize: Tuple for figure size (default: (15, 8))
                - dpi: Dots per inch for the plot (default: 100)
                - q: Number of quantiles for plotting (default: 20)
                - threshold_nb_values: Threshold for number of unique values
                        in numerical features (default: 15)
        """
        if features_to_explain is None:
            features_to_explain = self.features

        if self.global_explainer:
            # plot a global features importance
            self._explain_global_features(**kwargs)

        if self.local_explainer:
            # check if num features are corrects
            self._explain_numerical(features_to_explain, **kwargs)

            # check if cat features are corrects
            self._explain_categorical(features_to_explain, **kwargs)

    def correctness_features(
        self,
        q: Optional[int] = None,
    ) -> dict:
        """Analyze the correctness of the analysis for every feature.

        This method validates interpretation consistency between actual target rates
        and SHAP values for all features in the explainer.

        Args:
            q (int): Number of quantiles for continuous features.
                If None, uses adaptive quantiles. Defaults to None.

        Returns:
            dict: Dictionary with feature names as keys and correctness results as values.
        """
        if self.shap_values_train is None:
            return {}

        results = {}
        for feature in self.features:
            # Get feature index for SHAP values
            feature_index = get_index_of_features(self.x_train, feature)
            feature_shap_values = self.shap_values_train[:, feature_index]

            results[feature] = validate_single_feature_interpretation(
                x_train=self.x_train,
                y_binary=self.y_train,
                feature=feature,
                feature_shap_values=feature_shap_values,
                numerical_features=self.numerical_features,
                ymean_binary=self.ymean_train,
                q=q,
            )
        return results

    def _explain_global_features(self, **kwargs):
        """Interpret global features for binary classification.
        This method visualizes the global feature importance based on the mean of
        the absolute SHAP values for each feature.

        Args:
            **kwargs: Additional keyword arguments for customization, such as:
                - figsize: Tuple for figure size (default: (15, 8))
        """
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

        demo_mode = kwargs.get("demo_mode", False)
        if demo_mode:
            pyplot(fig)

        plt.show()

    def _explain_numerical(self, features_to_explain: list[str], **kwargs):
        """Interpret numerical features for binary classification.
        This method visualizes the relationship between numerical features and
        the target variable, and plots SHAP values for each numerical feature.

        Args:
            **kwargs: Additional keyword arguments for customization, such as:
                - figsize: Tuple for figure size (default: (15, 8))
                - dpi: Dots per inch for the plot (default: 100)
                - q: Number of quantiles for plotting (default: 20)
                - threshold_nb_values: Threshold for number of unique values
                        in numerical features (default: 15)
        """
        numerical_features_to_explain = [
            feature
            for feature in features_to_explain
            if feature in self.numerical_features
        ]
        # calculate ymean in train
        for feature in numerical_features_to_explain:
            min_value_train, max_value_train = calculate_min_max_value(
                self.x_train, feature
            )

            # calculate delta
            delta = (max_value_train - min_value_train) / 10

            # Set default values for figsize and dpi if not provided
            figsize = kwargs.get("figsize", (15, 8))
            dpi = kwargs.get("dpi", 100)
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

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

            demo_mode = kwargs.get("demo_mode", False)
            if demo_mode:
                pyplot(fig)

            plt.show()
            plt.close()

    def _explain_categorical(self, features_to_explain: list[str], **kwargs):
        """Interpret categorical features for binary classification.
        This method visualizes the relationship between categorical features and
        the target variable, and plots SHAP values for each categorical feature.

        Args:
            **kwargs: Additional keyword arguments for customization, such as:
                - figsize: Tuple for figure size (default: (15, 8))
                - dpi: Dots per inch for the plot (default: 100)
                - color: Color for the plot (default: (0.28, 0.18, 0.71))
        """
        categorical_features_to_explain = [
            feature
            for feature in features_to_explain
            if feature in self.categorical_features
        ]
        for feature in categorical_features_to_explain:
            # Set default values for figsize and dpi if not provided
            figsize = kwargs.get("figsize", (15, 8))
            dpi = kwargs.get("dpi", 100)
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            color = kwargs.get("color", (0.28, 0.18, 0.71))

            # little refactorization for missing values and interpretability
            self.x_train[feature] = self.x_train[feature].astype(str)
            self.x_train[feature] = self.x_train[feature].fillna("missing_value")

            ax = plot_feature_target_categorical_binary(
                self.x_train, self.y_train, feature, ax, color
            )

            ax, _ = plot_shap_values_categorical_binary(
                self.x_train, feature, self.shap_values_train, ax
            )

            demo_mode = kwargs.get("demo_mode", False)
            if demo_mode:
                pyplot(fig)

            plt.show()
            plt.close()
