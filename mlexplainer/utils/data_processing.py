"""Utility functions for data processing in ML Explainer."""

from pandas import concat, DataFrame, Series, merge


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

    return dataframe[feature].min(), dataframe[feature].max()


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


def get_index_of_features(dataframe: DataFrame, feature: str) -> int:
    """Get the index of a feature in the DataFrame columns.

    Args:
        dataframe (DataFrame): DataFrame containing the features.
        feature (str): The feature name to find the index of.

    Returns:
        int: Index of the feature in the DataFrame columns.
    """
    try:
        return dataframe.columns.tolist().index(feature)
    except ValueError as exc:
        raise ValueError("Feature is not in dataframe.") from exc


def target_groupby_category(
    dataframe: DataFrame,
    feature: str,
    target_serie: Series,
) -> DataFrame:
    """Group by a categorical feature and calculate mean and volume of the target.

    Args:
        dataframe (DataFrame): Input DataFrame containing the feature and target.
        feature (str): The feature name to group by.
        target_serie (Series): The target series to calculate statistics for.

    Returns:
        DataFrame: DataFrame with mean and volume of the target for each group.
    """
    target = target_serie.name
    df_feat_target = concat(
        [dataframe[[feature]], target_serie], axis=1
    ).copy()
    df_feat_target["group"] = dataframe[feature]

    df_feat_target_group_mean = (
        df_feat_target.groupby("group", dropna=False, observed=False)[target]
        .mean()
        .sort_index()
        .reset_index()
        .rename(columns={target: "mean_target"})
    )

    df_feat_target_group_volume = (
        df_feat_target.groupby("group", dropna=False, observed=False)[target]
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
