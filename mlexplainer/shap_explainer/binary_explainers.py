from typing import Callable, List

import matplotlib.pyplot as plt
from pandas import DataFrame, Series

from mlexplainer.shap_explainer.base import BaseMLExplainer
from mlexplainer.shap_explainer.base.shap_wrapper import (
    ShapWrapper,
)
from mlexplainer.shap_explainer.plots.plot_targets import (
    plot_feature_target_categorical_binary,
    plot_feature_target_numerical_binary,
)
from mlexplainer.shap_explainer.plots.plot_shap import (
    plot_shap_values_categorical_binary,
    plot_shap_values_numerical_binary,
)


class BinaryMLExplainer(BaseMLExplainer):

    def __init__(
        self,
        x_train: DataFrame,
        y_train: Series,
        features: List[str],
        model: Callable,
    ):

        super().__init__(x_train, y_train, features, model)

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

        # check if num features are corrects
        self._explain_numerical(**kwargs)

        # check if cat features are corrects
        self._explain_categorical(**kwargs)

        return None

    def _explain_numerical(self, **kwargs):
        """Interpret features for binary classification.

        This method should be implemented to provide feature interpretation
        specific to binary classification tasks.
        """

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
            ax = plot_feature_target_numerical_binary(
                self.x_train, self.y_train, feature, q, ax, delta
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

    def _explain_categorical(self, **kwargs):
        """Interpret features for binary classification.

        This method should be implemented to provide feature interpretation
        specific to binary classification tasks.
        """

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


def calculate_min_max_value(dataframe: DataFrame, feature: str):
    """
    Calculate the minimum and maximum values of a feature in a DataFrame.
    Args:
        dataframe (DataFrame): The DataFrame containing the feature.
        feature (str): The name of the feature to calculate min and max values.
    Returns:
        tuple: A tuple containing the minimum and maximum values of the feature.
    """
    if dataframe[feature].dtype == "category":
        return 0, dataframe[feature].value_counts().shape[0] - 1
    else:
        return dataframe[feature].min(), dataframe[feature].max()
