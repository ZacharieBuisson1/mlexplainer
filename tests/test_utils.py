"""Tests for utility modules."""

import unittest
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from mlexplainer.utils.data_processing import (
    calculate_min_max_value,
    get_index,
    get_index_of_features,
    target_groupby_category,
)
from mlexplainer.utils.quantiles import (
    group_values,
    is_in_quantile,
    nb_min_quantiles,
)


class TestDataProcessing(unittest.TestCase):
    """Test cases for data_processing module."""

    def setUp(self):
        """Set up test data."""
        self.df = DataFrame(
            {
                "numerical": [1.0, 2.0, 3.0, 4.0, 5.0],
                "categorical": pd.Categorical(["A", "B", "A", "C", "B"]),
                "string": ["text1", "text2", "text3", "text4", "text5"],
            }
        )
        self.target = Series([0, 1, 0, 1, 0], name="target")

    def test_calculate_min_max_value_numerical(self):
        """Test calculate_min_max_value for numerical feature."""
        min_val, max_val = calculate_min_max_value(self.df, "numerical")
        self.assertEqual(min_val, 1.0)
        self.assertEqual(max_val, 5.0)

    def test_calculate_min_max_value_categorical(self):
        """Test calculate_min_max_value for categorical feature."""
        min_val, max_val = calculate_min_max_value(self.df, "categorical")
        self.assertEqual(min_val, 0)
        self.assertEqual(max_val, 2)  # 3 categories - 1

    def test_get_index_valid_column(self):
        """Test get_index with valid column."""
        index = get_index("numerical", self.df)
        self.assertEqual(index, 0)

        index = get_index("categorical", self.df)
        self.assertEqual(index, 1)

    def test_get_index_invalid_column(self):
        """Test get_index with invalid column."""
        with self.assertRaises(ValueError):
            get_index("nonexistent", self.df)

    def test_get_index_of_features_valid(self):
        """Test get_index_of_features with valid feature."""
        index = get_index_of_features(self.df, "numerical")
        self.assertEqual(index, 0)

        index = get_index_of_features(self.df, "string")
        self.assertEqual(index, 2)

    def test_get_index_of_features_invalid(self):
        """Test get_index_of_features with invalid feature."""
        with self.assertRaises(ValueError) as context:
            get_index_of_features(self.df, "nonexistent")

        self.assertIn("Feature is not in dataframe.", str(context.exception))

    def test_target_groupby_category(self):
        """Test target_groupby_category function."""
        result = target_groupby_category(self.df, "categorical", self.target)

        self.assertIsInstance(result, DataFrame)
        self.assertIn("group", result.columns)
        self.assertIn("mean_target", result.columns)
        self.assertIn("volume_target", result.columns)

        # Test that all categories are present
        expected_groups = ["A", "B", "C"]
        self.assertEqual(sorted(result["group"].tolist()), expected_groups)

    def test_target_groupby_category_with_na(self):
        """Test target_groupby_category with NaN values."""
        df_with_na = self.df.copy()
        df_with_na.loc[0, "categorical"] = None

        result = target_groupby_category(
            df_with_na, "categorical", self.target
        )

        self.assertIsInstance(result, DataFrame)
        # Should include NaN group
        self.assertTrue(pd.isna(result["group"]).any() or len(result) >= 3)


class TestQuantiles(unittest.TestCase):
    """Test cases for quantiles module."""

    def setUp(self):
        """Set up test data."""
        self.series = Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.target = Series([0, 0, 1, 1, 0, 1, 1, 0, 1, 1])
        self.quantiles = [2.5, 5.0, 7.5]

    def test_is_in_quantile_normal_values(self):
        """Test is_in_quantile with normal values."""
        result = is_in_quantile(1, self.quantiles)
        self.assertEqual(result, 2.5)

        result = is_in_quantile(4, self.quantiles)
        self.assertEqual(result, 5.0)

        result = is_in_quantile(6, self.quantiles)
        self.assertEqual(result, 7.5)

    def test_is_in_quantile_exceeds_all(self):
        """Test is_in_quantile with value exceeding all quantiles."""
        result = is_in_quantile(10, self.quantiles)
        self.assertEqual(result, np.inf)

    def test_is_in_quantile_nan(self):
        """Test is_in_quantile with NaN value."""
        result = is_in_quantile(np.nan, self.quantiles)
        self.assertEqual(result, -1)

    def test_is_in_quantile_empty_list(self):
        """Test is_in_quantile with empty quantiles list."""
        result = is_in_quantile(5, [])
        self.assertEqual(result, np.inf)

    def test_nb_min_quantiles_default(self):
        """Test nb_min_quantiles with default parameters."""
        result = nb_min_quantiles(self.series)
        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)
        self.assertLessEqual(result, len(self.series.unique()))

    def test_nb_min_quantiles_with_q(self):
        """Test nb_min_quantiles with specific q."""
        result = nb_min_quantiles(self.series, q=5)
        self.assertEqual(result, 5)

    def test_nb_min_quantiles_with_nan(self):
        """Test nb_min_quantiles with NaN values."""
        series_with_nan = self.series.copy()
        series_with_nan.iloc[:3] = np.nan

        result = nb_min_quantiles(series_with_nan)
        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)

    def test_nb_min_quantiles_all_nan(self):
        """Test nb_min_quantiles with all NaN values."""
        series_all_nan = Series([np.nan] * 10)
        result = nb_min_quantiles(series_all_nan)
        self.assertEqual(result, 1)

    def test_group_values_default(self):
        """Test group_values with default parameters."""
        result_df, used_q = group_values(self.series, self.target, q=None)

        self.assertIsInstance(result_df, DataFrame)
        self.assertIn("group", result_df.columns)
        self.assertIn("target", result_df.columns)
        self.assertIsNone(used_q)

    def test_group_values_with_q(self):
        """Test group_values with specific q."""
        result_df, used_q = group_values(self.series, self.target, q=3)

        self.assertIsInstance(result_df, DataFrame)
        self.assertEqual(used_q, 3)

    def test_group_values_with_nan(self):
        """Test group_values with NaN values."""
        series_with_nan = self.series.copy()
        series_with_nan.iloc[:2] = np.nan

        result_df, used_q = group_values(series_with_nan, self.target, q=None)

        self.assertIsInstance(result_df, DataFrame)
        # Should handle NaN group
        self.assertTrue(len(result_df) > 0)

    def test_group_values_all_nan(self):
        """Test group_values with all NaN values."""
        series_all_nan = Series([np.nan] * 10)

        result_df, used_q = group_values(series_all_nan, self.target, q=None)

        self.assertIsInstance(result_df, DataFrame)
        self.assertEqual(used_q, 1)
        self.assertEqual(len(result_df), 1)


if __name__ == "__main__":
    unittest.main()
