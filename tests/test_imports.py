"""Tests for package imports and basic functionality."""

import unittest


class TestPackageImports(unittest.TestCase):
    """Test cases for package imports."""
    
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

    def test_import_multilabel_explainer(self):
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


if __name__ == '__main__':
    unittest.main()