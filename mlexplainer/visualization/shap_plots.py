"""Module for plotting SHAP values in various formats.
This module provides functions to visualize SHAP values for both numerical and categorical features,
as well as for binary and multilabel classification tasks.
"""

from numpy import ndarray
from pandas import DataFrame, Series
from matplotlib.axes import Axes
import matplotlib.colors as mcolors

from mlexplainer.utils.data_processing import (
    get_index_of_features,
)


def plot_shap_scatter(
    feature_values: Series,
    shap_values: ndarray,
    ax: Axes,
    color_positive: tuple[float, float, float] = (1.0, 0.5, 0.34),
    color_negative: tuple[float, float, float] = (0.12, 0.53, 0.9),
    marker: str = "o",
    alpha: float = 1.0,
    s: float = 2.0,
    annotate: bool = True,
) -> Axes:
    """Plot a scatter plot of SHAP values.
    This function creates a scatter plot of SHAP values against feature values.
    The points are colored based on whether the SHAP value is positive or negative.

    Args:
        feature_values (Series): Values of the feature to plot.
        shap_values (ndarray): SHAP values to plot.
        ax (Axes): Matplotlib axis to plot on.
        color_positive (tuple[float, float, float], optional): Positive color.
            Defaults to (1.0, 0.5, 0.34).
        color_negative (tuple[float, float, float], optional): Negative color.
            Defaults to (0.12, 0.53, 0.9).
        marker (str, optional): Marker style for the scatter plot.
            Defaults to "o".
        alpha (int, optional): Alpha transparency of the points.
            Defaults to 1.
        s (int, optional): Size of the points in the scatter plot.
            Defaults to 2.
        annotate (bool, optional): Whether to annotate the plot with text labels.
            Defaults to True.

    Returns:
        Axes: Matplotlib axis with the scatter plot.
    """

    colors = [color_positive if v > 0 else color_negative for v in shap_values]
    ax.scatter(
        feature_values, shap_values, c=colors, s=s, alpha=alpha, marker=marker
    )
    ax.tick_params(axis="y", labelsize="large")

    if annotate:
        ax.text(
            x=1.05,
            y=0.8,
            s="Positive Impact",
            fontsize="large",
            rotation=90,
            ha="left",
            va="center",
            transform=ax.transAxes,
            color=color_positive,
        )
        ax.text(
            x=1.05,
            y=0.2,
            s="Negative Impact",
            fontsize="large",
            rotation=90,
            ha="left",
            va="center",
            transform=ax.transAxes,
            color=color_negative,
        )
        ax.text(
            x=1.1,
            y=0.5,
            s="Shapley Values",
            fontsize="large",
            rotation=90,
            ha="left",
            va="center",
            transform=ax.transAxes,
            color="black",
        )

        color_positive_str = mcolors.to_hex(color_positive)
        color_negative_str = mcolors.to_hex(color_negative)
        for tick in ax.get_yticklabels():
            tick_text = tick.get_text().replace(
                "−", "-"
            )  # Replace unicode minus with ASCII
            tick_value = float(tick_text)
            if tick_value == 0:
                color = "black"
            elif tick_value > 0:
                color = color_positive_str
            else:
                color = color_negative_str
            tick.set_color(color)

    return ax


def plot_shap_values_numerical_binary(
    x_train: DataFrame,
    feature: str,
    shap_values_train: ndarray,
    delta: float,
    ymean_train: float,
    ax: Axes,
    color_positive: tuple[float, float, float] = (1.0, 0.5, 0.34),
    color_negative: tuple[float, float, float] = (0.12, 0.53, 0.9),
    marker: str = "o",
    alpha: float = 1.0,
    s: float = 2.0,
    annotate: bool = True,
) -> tuple[Axes, Axes]:
    """
    Plot SHAP values for a binary classification feature.
    This function creates a scatter plot of SHAP values against feature values
    for a binary classification task. It adjusts the y-axis limits to center around
    the mean of the target variable in the training set, and aligns the secondary
    y-axis (SHAP values) with the primary y-axis (mean target).

    Args:
        x_train (DataFrame): Training feature values.
        feature (str): The feature name to plot.
        shap_values_train (ndarray): SHAP values for the training features.
        delta (float): Delta value for adjusting plot limits.
        ymean_train (float): Mean of the target variable in the training set.
        ax: Matplotlib axis to plot on.
        color_positive (tuple[float, float, float], optional): Color for positive SHAP values.
            Defaults to (1.0, 0.5, 0.34).
        color_negative (tuple[float, float, float], optional): Color for negative SHAP values.
            Defaults to (0.12, 0.53, 0.9).
        marker (str, optional): Marker style for the scatter plot. Defaults to "o".
        alpha (float, optional): Alpha transparency of the points. Defaults to 1.0.
        s (float, optional): Size of the points in the scatter plot. Defaults to 2.0.
        annotate (bool, optional): Whether to annotate the plot with text labels.
            Defaults to True.

    Returns:
        tuple: Matplotlib axes for the main plot and SHAP plot.
    """
    # plot shap values
    ax2 = ax.twinx()
    index_feature = get_index_of_features(x_train, feature)

    feature_values = x_train[feature].fillna(
        x_train[feature].min() - delta / 2
    )

    ax2 = plot_shap_scatter(
        feature_values,
        shap_values_train[:, index_feature],
        ax2,
        color_positive=color_positive,
        color_negative=color_negative,
        marker=marker,
        alpha=alpha,
        s=s,
        annotate=annotate,
    )

    # Align and center the secondary y-axis (SHAP values) with the primary y-axis (real mean target)
    primary_ymin, primary_ymax = ax.get_ylim()  # Get primary y-axis limits
    shap_min, shap_max = (
        shap_values_train[:, index_feature].min(),
        shap_values_train[:, index_feature].max(),
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

    # Set the color of the y-ticks based on the SHAP values
    colorize_yticklabel_shap(
        ax2, color_positive=color_positive, color_negative=color_negative
    )

    return ax, ax2


def plot_shap_values_categorical_binary(
    x_train: DataFrame,
    feature: str,
    shap_values_train: ndarray,
    ax: Axes,
    color_positive: tuple[float, float, float] = (1.0, 0.5, 0.34),
    color_negative: tuple[float, float, float] = (0.12, 0.53, 0.9),
    marker: str = "o",
    alpha: float = 1.0,
    s: float = 2.0,
    annotate: bool = True,
) -> tuple[Axes, Axes]:
    """Plot SHAP values for a categorical feature in a binary classification task.
    This function creates a scatter plot of SHAP values against feature values
    for a binary classification task. It adjusts the y-axis limits to center around
    the mean of the target variable in the training set, and aligns the secondary
    y-axis (SHAP values) with the primary y-axis (mean target).

    Args:
        x_train (DataFrame): Training feature values.
        feature (str): The feature name to plot.
        shap_values_train (ndarray): SHAP values for the training features.
        ax (Axes): Matplotlib axis to plot on.
        color_positive (tuple[float, float, float], optional): Color for positive SHAP values.
            Defaults to (1.0, 0.5, 0.34).
        color_negative (tuple[float, float, float], optional): Color for negative SHAP values.
            Defaults to (0.12, 0.53, 0.9).
        marker (str, optional): Marker style for the scatter plot. Defaults to "o".
        alpha (float, optional): Alpha transparency of the points. Defaults to 1.0.
        s (float, optional): Size of the points in the scatter plot. Defaults to 2.0.
        annotate (bool, optional): Whether to annotate the plot with text labels.
            Defaults to True.

    Returns:
        tuple: Matplotlib axes for the main plot and SHAP plot.
    """

    # calculate the index of the features in the dataframe, to cross with shap values
    index_feature = get_index_of_features(x_train, feature)

    # plot shap values
    ax2 = ax.twinx()
    feature_values = x_train[feature].copy()

    shap_min, shap_max = (
        shap_values_train[:, index_feature].min(),
        shap_values_train[:, index_feature].max(),
    )
    max_shap_offset = max(abs(shap_min), abs(shap_max))

    # Set the limits for the secondary y-axis (centered around 0)
    ax2.set_ylim(
        -max_shap_offset,
        max_shap_offset,
    )

    ax2 = plot_shap_scatter(
        feature_values,
        shap_values_train[:, index_feature],
        ax2,
        color_positive=color_positive,
        color_negative=color_negative,
        marker=marker,
        alpha=alpha,
        s=s,
        annotate=annotate,
    )

    # Set the color of the y-ticks based on the SHAP values
    colorize_yticklabel_shap(
        ax2, color_positive=color_positive, color_negative=color_negative
    )

    return ax, ax2


def plot_shap_values_numerical_multilabel(
    x_train: DataFrame,
    y_train: Series,
    feature: str,
    shap_values_train: ndarray,
    delta: float,
    axes: ndarray,
    color_positive: tuple[float, float, float] = (1.0, 0.5, 0.34),
    color_negative: tuple[float, float, float] = (0.12, 0.53, 0.9),
    marker: str = "o",
    alpha: float = 1.0,
    s: float = 2.0,
    annotate: bool = True,
) -> ndarray:
    """
    Plot SHAP values for a numerical feature in a multilabel classification task.
    This function creates a scatter plot of SHAP values against feature values
    for each modality in the multilabel target. It adjusts the y-axis limits to center
    around the mean of the target variable in the training set for each modality,
    and aligns the secondary y-axis (SHAP values) with the primary y-axis (mean target).

    Args:
        x_train (DataFrame): Training feature values.
        y_train (Series): Training target values (multilabel).
        feature (str): The feature name to plot.
        shap_values_train (ndarray): SHAP values for the training features.
        delta (float): Delta value for adjusting plot limits.
        axes (ndarray): Array of Matplotlib axes to plot on, one for each modality.
        color_positive (tuple[float, float, float], optional): Color for positive SHAP values.
        color_negative (tuple[float, float, float], optional): Color for negative SHAP values.
        marker (str, optional): Marker style for the scatter plot. Defaults to "o".
        alpha (float, optional): Alpha transparency of the points. Defaults to 1.0.
        s (float, optional): Size of the points in the scatter plot. Defaults to 2.0.
        annotate (bool, optional): Whether to annotate the plot with text labels.

    Returns:
        ndarray: Array of Matplotlib axes with the scatter plots for each modality.
    """

    modalities = y_train.unique()

    for i, modality in enumerate(modalities):
        ax = axes[i]

        # Create a temporary binary target for the current modality
        y_binary = (y_train == modality).astype(int)
        ymean_binary = y_binary.mean()

        ax, ax2 = plot_shap_values_numerical_binary(
            x_train=x_train,
            feature=feature,
            shap_values_train=shap_values_train[:, :, i],
            delta=delta,
            ymean_train=ymean_binary,
            ax=ax,
            color_positive=color_positive,
            color_negative=color_negative,
            marker=marker,
            alpha=alpha,
            s=s,
            annotate=annotate,
        )

        # Set the color of the y-ticks based on the SHAP values
        colorize_yticklabel_shap(
            ax2, color_positive=color_positive, color_negative=color_negative
        )

        axes[i] = ax2

    return axes


def plot_shap_values_categorical_multilabel(
    x_train: DataFrame,
    y_train: Series,
    feature: str,
    shap_values_train: ndarray,
    axes: ndarray,
    color_positive: tuple[float, float, float] = (1.0, 0.5, 0.34),
    color_negative: tuple[float, float, float] = (0.12, 0.53, 0.9),
    marker: str = "o",
    alpha: float = 1.0,
    s: float = 2.0,
    annotate: bool = True,
) -> ndarray:
    """
    Plot SHAP values for a categorical feature in a multilabel classification task.
    This function creates a scatter plot of SHAP values against feature values
    for each modality in the multilabel target. It adjusts the y-axis limits to center
    around the mean of the target variable in the training set for each modality,
    and aligns the secondary y-axis (SHAP values) with the primary y-axis (mean target).

    Args:
        x_train (DataFrame): Training feature values.
        y_train (Series): Training target values (multilabel).
        feature (str): The feature name to plot.
        shap_values_train (ndarray): SHAP values for the training features.
        axes (ndarray): Array of Matplotlib axes to plot on, one for each modality.
        color_positive (tuple[float, float, float], optional): Color for positive SHAP values.
            Defaults to (1.0, 0.5, 0.34).
        color_negative (tuple[float, float, float], optional): Color for negative SHAP values.
            Defaults to (0.12, 0.53, 0.9).
        marker (str, optional): Marker style for the scatter plot. Defaults to "o".
        alpha (float, optional): Alpha transparency of the points. Defaults to 1.0.
        s (float, optional): Size of the points in the scatter plot. Defaults to 2.0.
        annotate (bool, optional): Whether to annotate the plot with text labels.

    Returns:
        ndarray: Array of Matplotlib axes with the scatter plots for each modality.
    """

    modalities = y_train.unique()

    for i, _ in enumerate(modalities):
        ax = axes[i]

        ax, ax2 = plot_shap_values_categorical_binary(
            x_train=x_train,
            feature=feature,
            shap_values_train=shap_values_train[:, :, i],
            ax=ax,
            color_positive=color_positive,
            color_negative=color_negative,
            marker=marker,
            alpha=alpha,
            s=s,
            annotate=annotate,
        )

        colorize_yticklabel_shap(
            ax2, color_positive=color_positive, color_negative=color_negative
        )

        axes[i] = ax2

    return axes


def colorize_yticklabel_shap(
    ax: Axes,
    color_positive: tuple[float, float, float] = (1.0, 0.5, 0.34),
    color_negative: tuple[float, float, float] = (0.12, 0.53, 0.9),
) -> Axes:
    """Colorize the y-tick labels of a SHAP plot based on their values.

    Args:
        ax (Axes): Matplotlib axis to modify.
        color_positive (tuple[float, float, float], optional): Color for positive SHAP values.
            Defaults to (1.0, 0.5, 0.34).
        color_negative (tuple[float, float, float], optional): Color for negative SHAP values.
            Defaults to (0.12, 0.53, 0.9).

    Returns:
        Axes: Matplotlib axis with colored y-tick labels.
    """
    for tick in ax.get_yticklabels():
        tick_text = tick.get_text().replace("−", "-")
        tick_value = float(tick_text)
        if tick_value == 0:
            color = "black"
        elif tick_value > 0:
            color = mcolors.to_hex(color_positive)
        else:
            color = mcolors.to_hex(color_negative)
        tick.set_color(color)

    return ax
