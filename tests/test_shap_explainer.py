from unittest import main, TestCase

from numpy import array, inf, nan
from pandas import DataFrame, Series

from mlexplainer.utils.data_processing import get_index
from mlexplainer.utils.quantiles import (
    group_values,
    is_in_quantile,
    nb_min_quantiles,
)


class TestShapExplainer(TestCase):
    """Class to evaluate all the functions in the shap_explainer module."""

    def setUp(self):
        self.data = {"A": [1, 2], "B": [3, 4], "C": [5, 6]}
        self.df = DataFrame(self.data)

    def test_get_index_valid_column(self):
        """TEST - Test the get_index function with a valid column."""
        self.assertEqual(get_index("B", self.df), 1)

    def test_get_index_invalid_column(self):
        """TEST - Test the get_index function with an invalid column."""
        with self.assertRaises(ValueError):
            get_index("D", self.df)

    def test_get_index_first_column(self):
        """TEST - Test the get_index function with the first column."""
        self.assertEqual(get_index("A", self.df), 0)

    def test_get_index_last_column(self):
        """TEST - Test the get_index function with the last column."""
        self.assertEqual(get_index("C", self.df), 2)

    def test_value_in_quantile(self):
        """TEST - Test the is_in_quantile function with a value in the quantile."""
        self.assertEqual(is_in_quantile(5, [10, 20, 30]), 10)
        self.assertEqual(is_in_quantile(15, [10, 20, 30]), 20)
        self.assertEqual(is_in_quantile(25, [10, 20, 30]), 30)

    def test_value_exceeds_all_quantiles(self):
        """TEST - Test the is_in_quantile function with a value that exceeds all"""
        self.assertEqual(is_in_quantile(35, [10, 20, 30]), inf)

    def test_value_is_na(self):
        """TEST - Test the is_in_quantile function with a value that is NA."""
        self.assertEqual(is_in_quantile(nan, [10, 20, 30]), -1)

    def test_empty_quantile_list(self):
        """TEST - Test the is_in_quantile function with an empty quantile list."""
        self.assertEqual(is_in_quantile(5, []), inf)


class TestNbMinQuantiles(TestCase):
    """Class to evaluate the nb_min_quantiles function."""

    def setUp(self):
        # Data with 20 random values
        self.data = DataFrame(
            {
                "feature": [
                    15,
                    12,
                    18,
                    4,
                    9,
                    15,
                    12,
                    11,
                    4,
                    11,
                    8,
                    10,
                    18,
                    13,
                    20,
                    5,
                    4,
                    11,
                    14,
                    20,
                ]
            }
        )

    def test_nb_min_quantiles_default(self):
        """TEST - Test the nb_min_quantiles function with the default parameters."""
        result = nb_min_quantiles(self.data["feature"])
        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)
        self.assertEqual(result, 10)

    def test_nb_min_quantiles_with_q(self):
        """TEST - Test the nb_min_quantiles function with a specific number of quantiles."""
        result = nb_min_quantiles(self.data["feature"], q=5)
        self.assertEqual(result, 5)

    def test_nb_min_quantiles_with_nan(self):
        """TEST - Test the nb_min_quantiles function with nan values."""
        # add nan values
        data_with_nan = self.data.copy()
        data_with_nan.loc[0:4, "feature"] = float("nan")

        result = nb_min_quantiles(data_with_nan["feature"])
        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)
        self.assertEqual(result, 9)

    def test_nb_min_quantiles_with_all_nan(self):
        """TEST - Test the nb_min_quantiles function with all nan values."""
        data_all_nan = DataFrame({"feature": [float("nan")] * 20})
        result = nb_min_quantiles(data_all_nan["feature"])
        self.assertEqual(result, 1)


class TestGroupValues(TestCase):

    def setUp(self):
        # Create a DataFrame with 20 rows of random values between 1 and 20 for testing
        self.dataframe = Series(
            [
                1,
                2,
                2,
                2,
                1,
                3,
                1,
                3,
                3,
                1,
                3,
                1,
                1,
                1,
                2,
                2,
                3,
                1,
                1,
                2,
            ]
        )
        self.target = Series(
            [
                1,
                0,
                1,
                0,
                0,
                1,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                1,
                1,
                0,
                0,
                1,
                0,
            ]
        )

    def test_group_values_default(self):
        """TEST - Test the group_values function with the default parameters."""
        results, used_q = group_values(self.dataframe, self.target, q=None)
        self.assertIsInstance(results, DataFrame)

        # validate values
        self.assertEqual(sorted(results["group"]), [1, 2, 3])
        self.assertEqual(
            [round(target, 2) for target in results["target"].tolist()],
            [0.56, 0.5, 0.8],
        )
        self.assertEqual(used_q, None)

    def test_group_values_with_q(self):
        """TEST - Test the group_values function with a specific number of quantiles."""
        results, used_q = group_values(self.dataframe, self.target, q=5)
        self.assertIsInstance(results, DataFrame)

        # validate values
        self.assertEqual(sorted(results["group"]), [1, 2, 3])
        self.assertEqual(
            [round(target, 2) for target in results["target"].tolist()],
            [0.56, 0.5, 0.8],
        )
        self.assertEqual(used_q, 5)

    def test_group_values_with_nan(self):
        """TEST - Test the group_values function with nan values."""
        x_with_nan = self.dataframe.copy()
        x_with_nan.loc[0:4] = float("nan")
        results, used_q = group_values(x_with_nan, self.target, q=None)

        # validate values
        self.assertTrue(
            sorted(results["group"])[0] != sorted(results["group"])[0]
        )
        self.assertEqual(sorted(results["group"])[1:], [1, 2, 3])
        self.assertEqual(
            [round(target, 2) for target in results["target"].tolist()],
            [0.4, 0.57, 0.67, 0.8],
        )
        self.assertEqual(used_q, None)

    def test_group_values_with_all_nan(self):
        """TEST - Test the group_values function with all nan values."""
        x_all_nan = Series([float("nan")] * 20)
        results, used_q = group_values(x_all_nan, self.target, q=None)

        # validate values
        self.assertTrue(
            sorted(results["group"])[0] != sorted(results["group"])[0]
        )
        self.assertEqual(
            [round(target, 2) for target in results["target"].tolist()],
            [0.6],
        )
        self.assertEqual(used_q, 1)


class TestBinaryMLExplainerImports(TestCase):
    def test_import_binary_explainer(self):
        """Test that BinaryMLExplainer can be imported from the main package."""
        from mlexplainer import BinaryMLExplainer
        self.assertTrue(BinaryMLExplainer)

    def test_import_shap_wrapper(self):
        """Test that ShapWrapper can be imported from the main package."""
        from mlexplainer import ShapWrapper  
        self.assertTrue(ShapWrapper)

    def test_import_base_explainer(self):
        """Test that BaseMLExplainer can be imported from the main package."""
        from mlexplainer import BaseMLExplainer
        self.assertTrue(BaseMLExplainer)

    def test_import_multilabel_functions(self):
        """Test that multilabel explainer can be imported."""
        from mlexplainer.explainers.shap import MultilabelMLExplainer
        self.assertTrue(MultilabelMLExplainer)

    def test_import_visualization_functions(self):
        """Test that visualization functions can be imported."""
        from mlexplainer.visualization import (
            plot_shap_values_numerical_binary,
            plot_feature_target_categorical_binary,
        )
        self.assertTrue(plot_shap_values_numerical_binary)
        self.assertTrue(plot_feature_target_categorical_binary)


if __name__ == "__main__":
    main()
