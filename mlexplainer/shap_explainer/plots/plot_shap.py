from pandas import DataFrame, Series

from mlexplainer.shap_explainer.plots.utils import (
    get_index_of_features,
)


def features_shap_plot(
    dataframe: DataFrame,
    feature_shap_values: Series,
    ax1,
    ax2,
    delta: float,
    type_of_shap: str = "train",
):
    """Plot SHAP values for features.

    Args:
        dataframe (DataFrame): Feature values.
        feature_shap_values (Series): SHAP values for the features.
        ax1 (Axes): Matplotlib axis for the main plot.
        ax2 (Axes): Matplotlib axis for the SHAP plot.
        delta (float): Delta value for adjusting plot limits.
        type_of_shap (str): Type of SHAP values ("train" or "test").

    Returns:
        tuple: Matplotlib axes for the main plot and SHAP plot.
    """
    # Fill NaN values to plot
    feature_values = dataframe.fillna(dataframe.min() - delta / 2)

    if type_of_shap == "train":
        shap_color = [(0.12, 0.53, 0.9), (1.0, 0.5, 0.34)]
    elif type_of_shap == "test":
        shap_color = [(0, 0, 0), (0, 0, 0)]
    else:
        raise ValueError("Invalid type_of_shap value. Use 'train' or 'test'.")

    colors = [shap_color[int(u > 0)] for u in feature_shap_values]

    ax2.scatter(
        feature_values,
        feature_shap_values,
        c=colors,
        s=2,
        alpha=1,
        marker="x" if type_of_shap == "test" else "o",
    )

    ax2.tick_params(axis="y", labelsize="large")
    ax2.text(
        x=1.05,
        y=0.8,
        s="Impact à la hausse",
        fontsize="large",
        rotation=90,
        ha="left",
        va="center",
        transform=ax2.transAxes,
        color=shap_color[1],
    )
    ax2.text(
        x=1.05,
        y=0.2,
        s="Impact à la baisse",
        fontsize="large",
        rotation=90,
        ha="left",
        va="center",
        transform=ax2.transAxes,
        color=shap_color[0],
    )
    ax2.text(
        x=1.1,
        y=0.5,
        s="Valeurs de Shapley",
        fontsize="large",
        rotation=90,
        ha="left",
        va="center",
        transform=ax2.transAxes,
        color="black",
    )

    return ax1, ax2


def plot_shap_values_numerical_binary(
    x_train: DataFrame,
    feature: str,
    shap_values_train,
    delta: float,
    ymean_train: float,
    ax,
) -> tuple:
    """
    Plot SHAP values for a binary classification feature.
    Args:
        x_train (DataFrame): Training feature values.
        feature (str): The feature name to plot.
        shap_values_train (ndarray): SHAP values for the training features.
        delta (float): Delta value for adjusting plot limits.
        ymean_train (float): Mean of the target variable in the training set.
        ax: Matplotlib axis to plot on.
    Returns:
        tuple: Matplotlib axes for the main plot and SHAP plot.
    """
    # plot shap values
    ax2 = ax.twinx()
    index_feature_train = get_index_of_features(x_train, feature)
    ax, ax2 = features_shap_plot(
        x_train[feature],
        shap_values_train[:, index_feature_train],
        ax,
        ax2,
        delta,
    )

    # Align and center the secondary y-axis (SHAP values) with the primary y-axis (real mean target)
    primary_ymin, primary_ymax = ax.get_ylim()  # Get primary y-axis limits
    shap_min, shap_max = (
        shap_values_train[:, index_feature_train].min(),
        shap_values_train[:, index_feature_train].max(),
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

    return ax, ax2


def plot_shap_values_categorical_binary(
    x_train: DataFrame,
    feature: str,
    shap_values_train,
    ax,
) -> tuple:

    # calculate the index of the features in the dataframe, to cross with shap values
    index_feature = get_index_of_features(x_train, feature)

    if shap_values_train is not None:
        shap_colors = [(0.12, 0.53, 0.9), (1.0, 0.5, 0.34)]
        colors = [
            shap_colors[int(u > 0)]
            for u in shap_values_train[:, index_feature]
        ]

        ax2 = ax.twinx()
        feature_values = x_train[feature].copy()
        ax2.scatter(
            feature_values,
            shap_values_train[:, index_feature],
            c=colors,
            s=2,
            alpha=1,
        )

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

        ax2.text(
            x=1.05,
            y=0.8,
            s="Impact à la hausse",
            fontsize="large",
            rotation=90,
            ha="left",
            va="center",
            transform=ax2.transAxes,
            color=shap_colors[1],
        )
        ax2.text(
            x=1.05,
            y=0.2,
            s="Impact à la baisse",
            fontsize="large",
            rotation=90,
            ha="left",
            va="center",
            transform=ax2.transAxes,
            color=shap_colors[0],
        )
        ax2.text(
            x=1.1,
            y=0.5,
            s="Valeurs de Shapley",
            fontsize="large",
            rotation=90,
            ha="left",
            va="center",
            transform=ax2.transAxes,
            color="black",
        )

    return ax, ax2
