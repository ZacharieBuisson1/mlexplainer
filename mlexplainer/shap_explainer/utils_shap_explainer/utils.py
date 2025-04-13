from pandas import DataFrame, isna, merge, notna, qcut, Series
from numpy import arange, inf, isclose, nan


def get_index(column_name: str, dataframe: DataFrame) -> int:
    """Extract the index of a column in a DataFrame.

    Args:
        column_name (str): Column name to extract the index from.
        dataframe (DataFrame): DataFrame to extract the index from.

    Returns:
        int: Index of the column in the DataFrame.
    """

    ind_index = list(dataframe.columns).index(column_name)
    return ind_index


def is_in_quantile(value: int, quantile_values: list[int]) -> int:
    """Return the quantile of a value, given a list of quantiles.

    Args:
        value (int): Search value.
        quantile_values (list[int]): List of quantiles.

    Returns:
        int: Upper bound of the quantile.
    """
    if isna(value):
        return -1

    for quantile in quantile_values:
        if value <= quantile:
            return quantile

    # return the last quantile if value exceeds all
    return inf


def nb_min_quantiles(x: DataFrame, q: int = None) -> int:
    """Calculate the number of quantiles to use for a feature.

    Args:
        x (DataFrame): DataFrame to calculate the number of quantiles for.
        q (int, optional): Number of quantiles. Defaults to None.

    Returns:
        int: Final number of quantiles to use.
    """
    if q is None:
        nb_quantile = 20  # min 5% of observations in each quantile
        min_size = 0  # min size after cutting

        while min_size < 0.01:
            cutting = qcut(x, nb_quantile, duplicates="drop").value_counts(
                dropna=False, normalize=True
            )

            # using only not NaN cut: cannot cut according to these type of observations
            not_nan = [u for u in cutting.index if notna(u)]
            min_size = cutting[not_nan].min()

            # using a smaller value of quantile to respect constraint
            nb_quantile -= 1
    else:
        nb_quantile = q

    # correction for NaN value: if it is relevant to consider it as a class or not
    pct_nan = x.isna().mean()
    nb_quantile = min(int(nb_quantile * (1 - pct_nan)) + 1, nb_quantile)

    return nb_quantile


def group_values(x: Series, y: Series, q: int) -> tuple[DataFrame, int]:
    """Create a new DataFrame of cut values.

    Args:
        x (DataFrame): Feature values.
        y (DataFrame): Target values.
        q (int): Number of quantiles.

    Returns:
        DataFrame: Grouped values with statistics.
        int: Used quantiles.
    """
    # Check if the series is full of missing values
    if x.isna().all():
        results = DataFrame(
            {"group": [nan], "target": [y.mean()], "volume": [len(x)]}
        )
        return results, 1

    df = DataFrame({"value": x, "target": y})

    if q is None:
        df["group"] = df["value"]
    elif x.nunique() <= 15:
        df["group"] = df["value"]
    else:
        q = nb_min_quantiles(x, q)
        quantiles = arange(1 / q, 1, 1 / q)
        quantiles = [quant for quant in quantiles if not isclose(quant, 1)]
        quantiles_values = list(df["value"].quantile(quantiles)) + [inf]

        df["group"] = df["value"].apply(
            lambda u: is_in_quantile(u, quantiles_values)
        )

    # handle missing value
    replaced_nan_value = df["group"].min() - 10
    df.loc[df["value"].isna(), "group"] = replaced_nan_value

    # compute statistics
    stats_groupby = df.groupby("group").mean().sort_index().reset_index()
    volume = (
        df["group"]
        .value_counts()
        .to_frame()
        .sort_index()
        .reset_index()
        .sort_index()
    )
    results = merge(volume, stats_groupby, how="left", on="group")

    # if there is at least one missing value in the dataset
    if df["value"].isna().any():
        results["group"] = results["group"].replace(
            results["group"].min(), nan
        )

    return results.sort_index(), q


def get_index_of_features(X: DataFrame, feature: str) -> int:
    """Get the index of a feature in the DataFrame columns.

    Args:
        X (DataFrame): DataFrame containing the features.
        feature (str): The feature name to find the index of.

    Returns:
        int: Index of the feature in the DataFrame columns.
    """
    try:
        return X.columns.tolist().index(feature)
    except ValueError as exc:
        raise ValueError("Feature is not in X.") from exc


def target_groupby_category(
    dataframe: DataFrame, feature: str, target: str
) -> DataFrame:
    """Group by a categorical feature and calculate mean and volume of the target.

    Args:
        dataframe (DataFrame): Input DataFrame containing the feature and target.
        feature (str): The feature name to group by.
        target (str): The target name to calculate statistics for.

    Returns:
        DataFrame: DataFrame with mean and volume of the target for each group.
    """
    df_feat_target = dataframe[[feature, target]].copy()
    df_feat_target["group"] = dataframe[feature]

    df_feat_target_group_mean = (
        df_feat_target.groupby("group", dropna=False)[target]
        .mean()
        .sort_index()
        .reset_index()
        .rename(columns={target: "mean_target"})
    )

    df_feat_target_group_volume = (
        df_feat_target.groupby("group", dropna=False)[target]
        .count()
        .sort_index()
        .reset_index()
        .rename(columns={target: "volume_target"})
    )

    results = merge(
        df_feat_target_group_mean,
        df_feat_target_group_volume,
        how="left",
        on="group",
    )

    return results
