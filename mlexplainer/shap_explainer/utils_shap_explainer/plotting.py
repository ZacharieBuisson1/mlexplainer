from pandas import concat, DataFrame, Series
from numpy import nan

from matplotlib import ticker

from .utils import group_values


def creneau(x: DataFrame, xmin: float, xmax: float) -> DataFrame:
    """Create a square wave with every group's mean.

    Args:
        x (pandas.DataFrame): Input DataFrame with a 'group' column.
        xmin (float): Minimum value to replace NaNs after shifting.
        xmax (float): Maximum value to replace the max group value.

    Returns:
        pandas.DataFrame: Transformed DataFrame with square wave pattern.
    """
    x_copy = x.copy()  # Copy of x
    x_copy["group"] = x_copy["group"].shift(1)
    x_copy["group"] = x_copy["group"].replace(nan, xmin)

    # Create column to sort dataframe at the end
    x_copy["ranking"] = "b"
    x["ranking"] = "a"

    # Concatenation of duplicated dataframes
    double_x = concat([x, x_copy], axis=0)  # Concatenation
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
) -> None:
    """Plot missing values on the given axis.

    Args:
        nan_observation (DataFrame): DataFrame containing missing values.
        xmin (float): Minimum value to replace NaNs after shifting.
        delta (float): Delta value for adjusting plot limits.
        ax: Matplotlib axis to plot on.
        color (tuple): Color for plotting missing values.
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


def plot_feature_target(
    x: Series, y: Series, q: int, ax, delta: float, ymean: float
) -> tuple:
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
    color = (0.28, 0.18, 0.71)  # Define purple color

    xmin, xmax = x.min(), x.max()  # Define observed min and max

    stats, used_q = group_values(x, y, q)  # Creation of stats
    nan_observation = stats.query("group != group")  # Gather missing values
    stats = stats.query("group == group")  # Select only non-missing values

    # If it's discrete features
    if not q or x.nunique() < 15:
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
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))

    # Change the label
    ax.set_ylabel("Taux de cible", fontsize="large", color=color)

    # Change size of ticks
    ax.tick_params(axis="y", labelsize="large")

    # Change color of ticks
    for tick in ax.get_yticklabels():
        tick.set_color(color)

    return ax, used_q


def features_shap_plot(
    X: DataFrame,
    feature_shap_values: Series,
    ax1,
    ax2,
    delta: float,
    type_of_shap: str = "train",
):
    """Plot SHAP values for features.

    Args:
        X (DataFrame): Feature values.
        feature_shap_values (Series): SHAP values for the features.
        ax1 (Axes): Matplotlib axis for the main plot.
        ax2 (Axes): Matplotlib axis for the SHAP plot.
        delta (float): Delta value for adjusting plot limits.
        type_of_shap (str): Type of SHAP values ("train" or "test").

    Returns:
        tuple: Matplotlib axes for the main plot and SHAP plot.
    """
    # Fill NaN values to plot
    feature_values = X.fillna(X.min() - delta / 2)

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
