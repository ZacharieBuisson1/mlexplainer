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


def feature_interpretation_multimodal(
    feature: str,
    x_train: DataFrame,
    y_train: Series,
    shap_values: ndarray = None,
    q: int = 20,
    feature_label: str = None,
    figsize: tuple = (15, 8),
    dpi: int = 100,
) -> None:
    """Interpret the impact of a feature on each modality of the target variable.

    Args:
        feature (str): The feature name to interpret.
        x_train (DataFrame): Training feature values.
        y_train (Series): Training target values (categorical or multi-modal).
        shap_values (ndarray, optional): SHAP values for the training features. Defaults to None.
        q (int, optional): Number of quantiles. Defaults to 20.
        feature_label (str, optional): Label for the feature. Defaults to None.
        figsize (tuple, optional): Figure size for the plot. Defaults to (15, 8).
        dpi (int, optional): Dots per inch for the plot. Defaults to 100.
    """
    # Get unique modalities in the target variable
    modalities = y_train.unique()

    # Calculate the number of rows and columns for subplots
    rows = (len(modalities) + 2) // 3  # 3 plots per row
    adjusted_figsize = (
        figsize[0],
        figsize[1] * rows / 2,
    )  # Adjust height based on rows
    fig, axes = plt.subplots(
        rows, 3, figsize=adjusted_figsize, dpi=dpi, sharex=True
    )
    axes = axes.flatten()  # Flatten the axes array for easier indexing

    if len(modalities) == 1:
        axes = [axes]  # Ensure axes is iterable for a single modality

    x_train_serie = x_train[feature].copy()

    if x_train[feature].dtype == "category":
        xmin, xmax = 0, x_train[feature].value_counts().shape[0] - 1
    else:
        xmin, xmax = x_train_serie.min(), x_train_serie.max()

    delta = (xmax - xmin) / 10

    for i, modality in enumerate(modalities):
        ax = axes[i]

        # Create a temporary binary target for the current modality
        y_temp = (y_train == modality).astype(int)

        # Plot feature-target relationship for this binary target
        ax, used_q = plot_feature_target(
            x_train_serie, y_temp, q, ax, delta, y_temp.mean()
        )

        if feature_label is None:
            feature_label = feature
        if used_q is not None:
            feature_label = f"{feature_label}, q={used_q}"

        ax.set_title(f"Target Modality: {modality}", fontsize="large")
        ax.set_ylabel("Probability", fontsize="large")

        # Add SHAP values if provided
        if shap_values is not None:
            ax2 = ax.twinx()
            index_feature_train = get_index_of_features(x_train, feature)
            ax, ax2 = features_shap_plot(
                x_train_serie,
                shap_values[:, index_feature_train][:, i],
                ax,
                ax2,
                delta,
            )

            # Align and center the secondary y-axis (SHAP values) with the primary y-axis (real mean target)
            primary_ymin, primary_ymax = (
                ax.get_ylim()
            )  # Get primary y-axis limits
            shap_min, shap_max = (
                shap_values[:, index_feature_train][:, i].min(),
                shap_values[:, index_feature_train][:, i].max(),
            )

            # Determine the center points
            primary_center = y_temp.mean()

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

    # Hide unused subplots
    for j in range(len(modalities), len(axes)):
        fig.delaxes(axes[j])

    plt.xlabel(feature_label, fontsize="large")
    plt.tight_layout()
    plt.show()


def feature_interpretation_multimodal_category(
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
        x_train (DataFrame): Training feature values.
        y_train (Series): Training target values.
        shap_values (ndarray, optional): SHAP values for the training features. Defaults to None.
        figsize (tuple, optional): Figure size for the plot. Defaults to (15, 8).
        dpi (int, optional): Dots per inch for the plot. Defaults to 200.
    """
    # Prepare the plot
    # Get unique modalities in the target variable
    modalities = y_train.unique()

    # Calculate the number of rows and columns for subplots
    rows = (len(modalities) + 2) // 3  # 3 plots per row
    adjusted_figsize = (
        figsize[0],
        figsize[1] * rows / 2,
    )  # Adjust height based on rows
    _, axes = plt.subplots(
        rows, 3, figsize=adjusted_figsize, dpi=dpi, sharex=True
    )
    axes = axes.flatten()  # Flatten the axes array for easier indexing

    color = (0.28, 0.18, 0.71)
    feature_train = x_train[feature].copy()
    index_feature = get_index_of_features(x_train, feature)

    for i, modality in enumerate(modalities):
        ax = axes[i]

        # Create a temporary binary target for the current modality
        y_temp = (y_train == modality).astype(int)

        if shap_values is not None:
            shap_colors = [(0.12, 0.53, 0.9), (1.0, 0.5, 0.34)]
            colors = [
                shap_colors[int(u > 0)]
                for u in shap_values[:, index_feature][:, i]
            ]

        tmp_x_train = x_train.copy()
        tmp_x_train[target] = tmp_x_train[target].apply(
            lambda u: int(u == modality)
        )
        stats_to_plot = target_groupby_category(tmp_x_train, feature, target)

        ax.plot(
            stats_to_plot["group"],
            stats_to_plot["mean_target"],
            "o",
            color=color,
        )
        ax.hlines(
            y_temp.mean(),
            0,
            feature_train.value_counts().shape[0] - 1,
            linestyle="--",
            color=color,
        )

        ax.set_title(f"Target Modality: {modality}", fontsize="large")
        ax.set_ylabel("Probability", fontsize="large")

        if shap_values is not None:
            ax2 = ax.twinx()
            feature_values = x_train[feature].copy()
            ax2.scatter(
                feature_values,
                shap_values[:, index_feature][:, i],
                c=colors,
                s=2,
                alpha=1,
            )

            index_feature_train = get_index_of_features(x_train, feature)

            # Align and center the secondary y-axis (SHAP values) with the primary y-axis (real mean target)
            primary_ymin, primary_ymax = (
                ax.get_ylim()
            )  # Get primary y-axis limits
            shap_min, shap_max = (
                shap_values[:, index_feature_train][:, i].min(),
                shap_values[:, index_feature_train][:, i].max(),
            )

            # Determine the center points
            ymean_train = y_temp.mean()
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
