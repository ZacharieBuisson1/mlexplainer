"""Plotting functions for SHAP explanations in binary classification tasks.
This module provides functions to visualize the relationship between features and target variables,
including numerical and categorical features, as well as handling missing values.
"""

from typing import Union

from pandas import concat, DataFrame, Series
from numpy import nan
import matplotlib.pyplot as plt

from matplotlib import ticker
from matplotlib.axes import Axes

from mlexplainer.utils import group_values, target_groupby_category


def creneau(x: DataFrame, xmin: float, xmax: float) -> DataFrame:
    """Create a square wave with every group's mean.

    Args:
        x (pandas.DataFrame): Input DataFrame with a 'group' column.
        xmin (float): Minimum value to replace NaNs after shifting.
        xmax (float): Maximum value to replace the max group value.

    Returns:
        pandas.DataFrame: Transformed DataFrame with square wave pattern.
    """
    # Work on explicit copies to avoid warnings
    x_a = x.copy()
    x_b = x.copy()

    x_b["group"] = x_b["group"].shift(1)
    x_b["group"] = x_b["group"].replace(nan, xmin)

    x_a["ranking"] = "a"
    x_b["ranking"] = "b"

    # Concatenation et tri
    double_x = concat([x_a, x_b], axis=0)
    double_x = double_x.sort_values(by=["group", "ranking"])
    double_x = double_x.drop(columns=["ranking"])

    # Applying min and max
    double_x["group"] = double_x["group"].replace(
        double_x["group"].min(), xmin
    )
    double_x["group"] = double_x["group"].replace(
        double_x["group"].max(), xmax
    )

    return double_x


def add_nans(
    nan_observation: DataFrame, xmin: float, delta: float, ax, color: tuple
) -> Axes:
    """Plot missing values on the given axis.

    Args:
        nan_observation (DataFrame): DataFrame containing missing values.
        xmin (float): Minimum value to replace NaNs after shifting.
        delta (float): Delta value for adjusting plot limits.
        ax: Matplotlib axis to plot on.
        color (tuple): Color for plotting missing values.
    Returns:
        Axes: Matplotlib axis with missing values plotted.
    """
    if nan_observation.shape[0] > 0:
        # Fill NaN values with a value to plot
        nan_observation = nan_observation.fillna(xmin - delta / 2)

        # Plot missing values impact
        ax.scatter(
            nan_observation["group"], nan_observation["target"], color=color
        )

        # Separation with rest of the graph
        ax.axvline(xmin - delta / 4, color="black", lw=1)

        # Define limits
        ax.set_xlim(xmin - delta * 3 / 4)

    return ax


def set_centered_ylim(ax: Axes, center: float) -> Axes:
    """Set the y-axis limits centered around a specified value.

    Args:
        ax (Axes): Matplotlib axis to set limits for.
        center (float): Center value around which to set the limits.
    Returns:
        Axes: Matplotlib axis with updated y-axis limits.
    """

    ymin, ymax = ax.get_ylim()
    max_offset = max(center - ymin, ymax - center)
    ax.set_ylim(center - max_offset, center + max_offset)

    return ax


def reformat_y_axis(
    ax: Axes,
    color: tuple[float, float, float] = (0.28, 0.18, 0.71),
) -> Axes:
    """Refactor the y-axis of a plot.
    Args:
        ax (Axes): Matplotlib axis to refactor.
        color (tuple[float, float, float]): Color for the y-axis label and ticks.
    Returns:
        Axes: Matplotlib axis with refactored y-axis.
    """

    # Transform tick to percent
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))

    # Change the label
    ax.set_ylabel("Target Rate", fontsize="large", color=color)

    # Change size of ticks
    ax.tick_params(axis="y", labelsize="large")

    # Change color of ticks
    for tick in ax.get_yticklabels():
        tick.set_color(color)

    return ax


def plot_feature_numerical_target(
    x: Series,
    y: Series,
    q: int,
    ax,
    delta: float,
    ymean: float,
    threshold_nb_values: float = 15,
    color: tuple[float, float, float] = (0.28, 0.18, 0.71),
) -> tuple[Axes, int | None]:
    """Plot the relationship between a feature and the target variable.

    Args:
        x (Series): Feature values.
        y (Series): Target values.
        q (int): Number of quantiles.
        ax: Matplotlib axis to plot on.
        delta (float): Delta value for adjusting plot limits.
        ymean (float): Mean of the target variable.

    Returns:
        tuple: Matplotlib axis and used quantiles.
    """

    xmin, xmax = x.min(), x.max()  # Define observed min and max

    stats, used_q = group_values(
        x, y, q, threshold_nb_values
    )  # Creation of stats
    nan_observation = stats.query("group != group")  # Gather missing values
    stats = stats.query("group == group")  # Select only non-missing values

    # If it's discrete features
    if not q or x.nunique() < threshold_nb_values:
        stats_to_plot = stats
    # If it is continuous values
    else:
        stats_to_plot = creneau(
            stats, xmin - delta / 4, xmax + delta / 4
        )  # Create square wave with them

    # Plot purple curves
    ax.plot(
        stats_to_plot["group"], stats_to_plot["target"], color=color, alpha=0.8
    )

    # Plot mean of the observed target
    ax.hlines(
        ymean,
        xmin - 3 * delta / 4,
        xmax + delta / 4,
        linestyle="--",
        color=color,
    )

    # Define limits of this graph
    ax.set_xlim(xmin - delta / 4, xmax + delta / 4)

    # Add missing values
    ax = add_nans(nan_observation, xmin, delta, ax, color)

    # Transform tick to percent
    ax = reformat_y_axis(ax, color)

    return ax, used_q


def plot_feature_target_numerical_binary(
    dataframe: DataFrame,
    target_serie: Series,
    feature: str,
    q: int,
    ax: Axes,
    delta: float,
    threshold_nb_values: float = 15,
    target_modality: Union[str, None] = None,
) -> Axes:
    """Plot the relationship between a feature and the target variable for binary
    classification.

    Args:
        dataframe (DataFrame): DataFrame containing the feature and target variable.
        target_serie (Series): Series representing the target variable.
        feature (str): The feature name to plot.
        q (int): Number of quantiles.
        ax (Axes): Matplotlib axis to plot on.
        delta (float): Delta value for adjusting plot limits.
        threshold_nb_values (float): Threshold for number of unique values to
        decide grouping method.

    Returns:
        Axes: Matplotlib axis with the feature-target plot.
    """

    ax, used_q = plot_feature_numerical_target(
        dataframe[feature],
        target_serie,
        q,
        ax,
        delta,
        target_serie.mean(),
        threshold_nb_values=threshold_nb_values,
    )

    # set up a label for the feature
    feature_label = feature
    if used_q is not None:
        feature_label = f"{feature_label}, q={used_q}"

    if target_modality is None:
        ax.set_xlabel(feature_label, fontsize="large")
    else:
        ax.set_xlabel(
            f"{feature_label}, binary target={target_modality} VS all.",
            fontsize="large",
        )

    return ax


def plot_feature_target_categorical_binary(
    dataframe: DataFrame,
    target: Series,
    feature: str,
    ax: Axes,
    color: tuple[float, float, float] = (0.28, 0.18, 0.71),
    target_modality: Union[str, None] = None,
):
    """
    Plot the relationship between a categorical feature and the
    target variable for binary classification.

    Args:
        dataframe (DataFrame): DataFrame containing the feature and target variable.
        target (Series): Series representing the target variable.
        feature (str): The feature name to plot.
        ax (Axes): Matplotlib axis to plot on.
        color (tuple[float, float, float]): Color for the plot.

    Returns:
        Axes: Matplotlib axis with the feature-target plot.
    """

    feature_train = dataframe[feature].copy()
    mean_target = target.mean()

    # First part - printing information for observed values
    stats_to_plot = target_groupby_category(dataframe, feature, target)
    ax.plot(
        stats_to_plot["group"],
        stats_to_plot["mean_target"],
        "o",
        color=color,
    )
    num_categories = feature_train.value_counts().shape[0]
    ax.hlines(
        mean_target,
        -0.5,
        num_categories - 0.5,
        linestyle="--",
        color=color,
    )

    # Determine the center points
    y_center = target.mean()
    ax = set_centered_ylim(ax, y_center)

    # Transform tick to percent
    ax = reformat_y_axis(ax, color)

    # set up a label for the feature
    feature_label = feature
    if target_modality is not None:
        ax.set_xlabel(
            f"{feature_label}, binary target={target_modality} VS all.",
            fontsize="large",
        )
    else:
        ax.set_xlabel(feature_label, fontsize="large")

    return ax


def plot_feature_target_numerical_multilabel(
    dataframe: DataFrame,
    target_serie: Series,
    feature: str,
    q: int = 20,
    delta: float = 0.1,
    figsize: tuple = (15, 8),
    dpi: int = 100,
    threshold_nb_values: float = 15,
):
    """Plot the relationship between a numerical feature and all target modalities with SHAP values.

    Args:
        dataframe (DataFrame): DataFrame containing the feature and target variable.
        target_serie (Series): Series representing the target variable.
        feature (str): The feature name to plot.
        modalities (list): List of unique target modalities.
        shap_values (ndarray, optional): SHAP values for the training features.
        q (int, optional): Number of quantiles. Defaults to 20.
        figsize (tuple, optional): Figure size for the plot. Defaults to (15, 8).
        dpi (int, optional): Dots per inch for the plot. Defaults to 100.
    """
    # Get unique modalities in the target variable
    modalities = target_serie.unique()

    # Calculate subplot layout
    rows = (len(modalities) + 2) // 3  # 3 plots per row
    adjusted_figsize = (figsize[0], figsize[1] * rows / 2)
    fig, axes = plt.subplots(rows, 3, figsize=adjusted_figsize, dpi=dpi)
    axes = axes.flatten()

    if len(modalities) == 1:
        axes = [axes]

    for i, modality in enumerate(modalities):

        ax = axes[i]

        # Create binary target for this modality
        y_binary = Series((target_serie == modality).astype(int))

        # Plot feature-target relationship
        ax = plot_feature_target_numerical_binary(
            dataframe,
            y_binary,
            feature,
            q,
            ax,
            delta,
            threshold_nb_values=threshold_nb_values,
            target_modality=modality,
        )

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    return fig, axes


def plot_feature_target_categorical_multilabel(
    dataframe: DataFrame,
    target_serie: Series,
    feature: str,
    modalities: list,
    figsize: tuple = (15, 8),
    dpi: int = 200,
    color: tuple[float, float, float] = (0.28, 0.18, 0.71),
):
    """Plot the relationship between a categorical feature and all target modalities with SHAP values.

    Args:
        dataframe (DataFrame): DataFrame containing the feature and target variable.
        target_serie (Series): Series representing the target variable.
        feature (str): The feature name to plot.
        modalities (list): List of unique target modalities.
        shap_values (ndarray, optional): SHAP values for the training features.
        figsize (tuple, optional): Figure size for the plot. Defaults to (15, 8).
        dpi (int, optional): Dots per inch for the plot. Defaults to 200.
    """
    # Calculate subplot layout
    rows = (len(modalities) + 2) // 3  # 3 plots per row
    adjusted_figsize = (figsize[0], figsize[1] * rows / 2)
    fig, axes = plt.subplots(
        rows, 3, figsize=adjusted_figsize, dpi=dpi, sharex=True
    )
    axes = axes.flatten()

    if len(modalities) == 1:
        axes = [axes]

    for i, modality in enumerate(modalities):
        ax = axes[i]

        # Create binary target for this modality
        y_binary = (target_serie == modality).astype(int)

        ax = plot_feature_target_categorical_binary(
            dataframe,
            y_binary,
            feature,
            ax,
            color=color,
            target_modality=modality,
        )

    return fig, axes
