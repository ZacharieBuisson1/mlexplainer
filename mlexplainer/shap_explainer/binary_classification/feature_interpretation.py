"""Module to interpret the impact of a feature on the target variable."""

from pandas import DataFrame, Series
from numpy import ndarray

import matplotlib.pyplot as plt

from ..utils_shap_explainer.plotting import (
    plot_feature_target,
    features_shap_plot,
)
from ..utils_shap_explainer.utils import (
    get_index_of_features,
    target_groupby_category,
)


def feature_interpretation(
    feature: str,
    x_train: DataFrame,
    y_train: Series,
    shap_values: ndarray = None,
    ymean_train: float = None,
    x_test: DataFrame = None,
    y_test: Series = None,
    shap_values_test: ndarray = None,
    q: int = 20,
    feature_label: str = None,
    figsize: tuple = (15, 8),
    dpi: int = 100,
) -> None:
    """Interpret the impact of a feature on the target variable.

    Args:
        feature (str): The feature name to interpret.
        x_train (DataFrame): Training feature values.
        y_train (Series): Training target values.
        shap_values (ndarray, optional): SHAP values for the training features. Defaults to None.
        ymean_train (float, optional): Mean of the training target variable. Defaults to None.
        x_test (DataFrame, optional): Test feature values. Defaults to None.
        y_test (Series, optional): Test target values. Defaults to None.
        shap_values_test (ndarray, optional): SHAP values for the test features. Defaults to None.
        q (int, optional): Number of quantiles. Defaults to 20.
        feature_label (str, optional): Label for the feature. Defaults to None.
        figsize (tuple, optional): Figure size for the plot. Defaults to (15, 8).
        dpi (int, optional): Dots per inch for the plot. Defaults to 100.
    """
    x_train_serie = x_train[feature].copy()

    if ymean_train is None:
        ymean_train = y_train.mean()

    if x_train[feature].dtype == "category":
        xmin, xmax = 0, x_train[feature].value_counts().shape[0] - 1
    else:
        xmin, xmax = x_train_serie.min(), x_train_serie.max()

    if x_test is not None:
        x_test_serie = x_test[feature].copy()
        if x_test[feature].dtype == "category":
            xmin_test, xmax_test = (
                0,
                x_test[feature].value_counts().shape[0] - 1,
            )
        else:
            xmin_test, xmax_test = x_test_serie.min(), x_test_serie.max()
        xmin_test, xmax_test = min(xmin, xmin_test), max(xmax, xmax_test)

    delta = (xmax - xmin) / 10

    _, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax, used_q = plot_feature_target(
        x_train_serie, y_train, q, ax, delta, ymean_train
    )

    if feature_label is None:
        feature_label = feature
    if used_q is not None:
        feature_label = f"{feature_label}, q={used_q}"

    ax.set_xlabel(feature_label, fontsize="large")

    if shap_values is not None:
        ax2 = ax.twinx()
        index_feature_train = get_index_of_features(x_train, feature)
        ax, ax2 = features_shap_plot(
            x_train_serie,
            shap_values[:, index_feature_train],
            ax,
            ax2,
            delta,
        )

        # Align and center the secondary y-axis (SHAP values) with the primary y-axis (real mean target)
        primary_ymin, primary_ymax = ax.get_ylim()  # Get primary y-axis limits
        shap_min, shap_max = (
            shap_values[:, index_feature_train].min(),
            shap_values[:, index_feature_train].max(),
        )

        # Determine the center points
        primary_center = ymean_train

        # Calculate the maximum range to ensure symmetry
        max_primary_offset = max(
            primary_center - primary_ymin, primary_ymax - primary_center
        )
        max_shap_offset = max(abs(shap_min), abs(shap_max))

        # Set the limits for the primary y-axis (centered around ymean_train)
        ax.set_ylim(
            primary_center - max_primary_offset,
            primary_center + max_primary_offset,
        )

        # Set the limits for the secondary y-axis (centered around 0)
        ax2.set_ylim(
            -max_shap_offset,
            max_shap_offset,
        )

        if (
            x_test is not None
            and y_test is not None
            and shap_values_test is not None
        ):
            index_feature_test = get_index_of_features(x_test, feature)
            ax, ax2 = features_shap_plot(
                x_test_serie,
                shap_values_test[:, index_feature_test],
                ax,
                ax2,
                delta,
                type_of_shap="test",
            )

    plt.show()


def feature_interpretation_category(
    feature: str,
    x_train: DataFrame,
    y_train: Series,
    target: str,
    shap_values: ndarray = None,
    figsize: tuple = (15, 8),
    dpi: int = 200,
) -> None:
    """Interpret the impact of a categorical feature on the target variable.

    Args:
        feature (str): The feature name to interpret.
        X_train (DataFrame): Training feature values.
        y_train (Series): Training target values.
        target (str): The target name to calculate statistics for.
        shap_values (ndarray, optional): SHAP values for the training features. Defaults to None.
        figsize (tuple, optional): Figure size for the plot. Defaults to (15, 8).
        dpi (int, optional): Dots per inch for the plot. Defaults to 200.
    """
    _, ax = plt.subplots(figsize=figsize, dpi=dpi)
    color = (0.28, 0.18, 0.71)
    index_feature = get_index_of_features(x_train, feature)

    feature_train = x_train[feature].copy()
    mean_target = y_train.mean()

    if shap_values is not None:
        shap_colors = [(0.12, 0.53, 0.9), (1.0, 0.5, 0.34)]
        colors = [
            shap_colors[int(u > 0)] for u in shap_values[:, index_feature]
        ]

    # First part - printing information for observed values
    stats_to_plot = target_groupby_category(x_train, feature, target)
    ax.plot(
        stats_to_plot["group"],
        stats_to_plot["mean_target"],
        "o",
        color=color,
    )
    ax.hlines(
        mean_target,
        0,
        feature_train.value_counts().shape[0] - 1,
        linestyle="--",
        color=color,
    )

    # Second part - print SHAP values
    if shap_values is not None:
        ax2 = ax.twinx()
        feature_values = x_train[feature].copy()
        ax2.scatter(
            feature_values,
            shap_values[:, index_feature],
            c=colors,
            s=2,
            alpha=1,
        )

        index_feature_train = get_index_of_features(x_train, feature)
        # Align and center the secondary y-axis (SHAP values) with the primary y-axis (real mean target)
        primary_ymin, primary_ymax = ax.get_ylim()  # Get primary y-axis limits
        shap_min, shap_max = (
            shap_values[:, index_feature_train].min(),
            shap_values[:, index_feature_train].max(),
        )

        # Determine the center points
        ymean_train = y_train.mean()
        primary_center = ymean_train

        # Calculate the maximum range to ensure symmetry
        max_primary_offset = max(
            primary_center - primary_ymin, primary_ymax - primary_center
        )
        max_shap_offset = max(abs(shap_min), abs(shap_max))

        # Set the limits for the primary y-axis (centered around ymean_train)
        ax.set_ylim(
            primary_center - max_primary_offset,
            primary_center + max_primary_offset,
        )

        # Set the limits for the secondary y-axis (centered around 0)
        ax2.set_ylim(
            -max_shap_offset,
            max_shap_offset,
        )

    plt.show()
