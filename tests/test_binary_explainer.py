"""Tests for Binary ML Explainer functionality."""

import unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from mlexplainer.explainers.shap.binary import BinaryMLExplainer


class TestBinaryMLExplainer(unittest.TestCase):
    """Test cases for BinaryMLExplainer."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.x_train = DataFrame({
            'numerical_feature': np.random.randn(100),
            'categorical_feature': pd.Categorical(np.random.choice(['A', 'B', 'C'], 100)),
            'string_feature': np.random.choice(['text1', 'text2', 'text3'], 100)
        })
        self.y_train = Series(np.random.choice([0, 1], 100))
        self.features = ['numerical_feature', 'categorical_feature']
        self.model = Mock()
        
        # Mock SHAP values
        self.mock_shap_values = np.random.randn(100, len(self.features))
    
    @patch('mlexplainer.explainers.shap.binary.ShapWrapper')
    def test_initialization_valid_binary(self, mock_shap_wrapper):
        """Test valid initialization with binary target."""
        mock_wrapper_instance = Mock()
        mock_wrapper_instance.calculate.return_value = self.mock_shap_values
        mock_shap_wrapper.return_value = mock_wrapper_instance
        
        explainer = BinaryMLExplainer(
            self.x_train, self.y_train, self.features, self.model
        )
        
        self.assertIsInstance(explainer, BinaryMLExplainer)
        self.assertEqual(explainer.features, self.features)
        self.assertEqual(explainer.model, self.model)
        self.assertIsNotNone(explainer.shap_values_train)
        self.assertIsNotNone(explainer.ymean_train)
    
    def test_initialization_non_binary_target(self):
        """Test error when target is not binary."""
        y_non_binary = Series([0, 1, 2])  # Three classes
        
        with self.assertRaises(ValueError) as context:
            BinaryMLExplainer(self.x_train, y_non_binary, self.features, self.model)
        
        self.assertIn("binary target variable with exactly two unique values", str(context.exception))
    
    def test_initialization_single_class_target(self):
        """Test error when target has only one class."""
        y_single_class = Series([0, 0, 0, 0])
        
        with self.assertRaises(ValueError) as context:
            BinaryMLExplainer(self.x_train, y_single_class, self.features, self.model)
        
        self.assertIn("binary target variable with exactly two unique values", str(context.exception))
    
    @patch('mlexplainer.explainers.shap.binary.ShapWrapper')
    @patch('mlexplainer.explainers.shap.binary.plt')
    def test_explain_global_only(self, mock_plt, mock_shap_wrapper):
        """Test explain method with global explainer only."""
        mock_wrapper_instance = Mock()
        mock_wrapper_instance.calculate.return_value = self.mock_shap_values
        mock_shap_wrapper.return_value = mock_wrapper_instance
        
        # Mock subplots to return proper fig, ax tuple
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        explainer = BinaryMLExplainer(
            self.x_train, self.y_train, self.features, self.model,
            global_explainer=True, local_explainer=False
        )
        
        explainer.explain()
        
        # Should call plotting functions
        self.assertTrue(mock_plt.show.called)
        self.assertTrue(mock_plt.subplots.called)
    
    @patch('mlexplainer.explainers.shap.binary.ShapWrapper')
    @patch('mlexplainer.explainers.shap.binary.plt')
    @patch('mlexplainer.explainers.shap.binary.plot_feature_target_numerical_binary')
    @patch('mlexplainer.explainers.shap.binary.plot_shap_values_numerical_binary')
    @patch('mlexplainer.explainers.shap.binary.plot_feature_target_categorical_binary')
    @patch('mlexplainer.explainers.shap.binary.plot_shap_values_categorical_binary')
    def test_explain_local_numerical(self, mock_cat_shap_plot, mock_cat_target_plot, 
                                   mock_shap_plot, mock_target_plot, mock_plt, mock_shap_wrapper):
        """Test explain method with numerical features."""
        mock_wrapper_instance = Mock()
        mock_wrapper_instance.calculate.return_value = self.mock_shap_values
        mock_shap_wrapper.return_value = mock_wrapper_instance
        
        # Mock matplotlib components
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_target_plot.return_value = mock_ax
        mock_shap_plot.return_value = (mock_ax, Mock())  # Returns tuple (ax, other)
        mock_cat_target_plot.return_value = mock_ax
        mock_cat_shap_plot.return_value = (mock_ax, Mock())
        
        explainer = BinaryMLExplainer(
            self.x_train, self.y_train, self.features, self.model,
            global_explainer=False, local_explainer=True
        )
        
        explainer.explain()
        
        # Should call numerical plotting functions
        self.assertTrue(mock_target_plot.called)
        self.assertTrue(mock_shap_plot.called)
    
    @patch('mlexplainer.explainers.shap.binary.ShapWrapper')
    @patch('mlexplainer.explainers.shap.binary.plt')
    @patch('mlexplainer.explainers.shap.binary.plot_feature_target_categorical_binary')
    @patch('mlexplainer.explainers.shap.binary.plot_shap_values_categorical_binary')
    @patch('mlexplainer.explainers.shap.binary.plot_feature_target_numerical_binary')
    @patch('mlexplainer.explainers.shap.binary.plot_shap_values_numerical_binary')
    def test_explain_local_categorical(self, mock_num_shap_plot, mock_num_target_plot,
                                     mock_shap_plot, mock_target_plot, mock_plt, mock_shap_wrapper):
        """Test explain method with categorical features."""
        mock_wrapper_instance = Mock()
        mock_wrapper_instance.calculate.return_value = self.mock_shap_values
        mock_shap_wrapper.return_value = mock_wrapper_instance
        
        # Mock matplotlib components
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_target_plot.return_value = mock_ax
        mock_shap_plot.return_value = (mock_ax, Mock())  # Returns tuple (ax, other)
        mock_num_target_plot.return_value = mock_ax
        mock_num_shap_plot.return_value = (mock_ax, Mock())
        
        explainer = BinaryMLExplainer(
            self.x_train, self.y_train, self.features, self.model,
            global_explainer=False, local_explainer=True
        )
        
        explainer.explain()
        
        # Should call categorical plotting functions
        self.assertTrue(mock_target_plot.called)
        self.assertTrue(mock_shap_plot.called)
    
    @patch('mlexplainer.explainers.shap.binary.ShapWrapper')
    def test_correctness_features(self, mock_shap_wrapper):
        """Test correctness_features method."""
        mock_wrapper_instance = Mock()
        mock_wrapper_instance.calculate.return_value = self.mock_shap_values
        mock_shap_wrapper.return_value = mock_wrapper_instance
        
        explainer = BinaryMLExplainer(
            self.x_train, self.y_train, self.features, self.model
        )
        
        result = explainer.correctness_features()
        
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), set(self.features))
        
        # Each feature should have validation results
        for feature, validation_result in result.items():
            self.assertIsInstance(validation_result, list)
    
    @patch('mlexplainer.explainers.shap.binary.ShapWrapper')
    def test_validate_feature_interpretation_invalid_feature(self, mock_shap_wrapper):
        """Test _validate_feature_interpretation with invalid feature."""
        mock_wrapper_instance = Mock()
        mock_wrapper_instance.calculate.return_value = self.mock_shap_values
        mock_shap_wrapper.return_value = mock_wrapper_instance
        
        explainer = BinaryMLExplainer(
            self.x_train, self.y_train, self.features, self.model
        )
        
        with self.assertRaises(ValueError) as context:
            explainer._validate_feature_interpretation('nonexistent_feature')
        
        self.assertIn("not found in features list", str(context.exception))
    
    @patch('mlexplainer.explainers.shap.binary.ShapWrapper')
    def test_validate_feature_interpretation_numerical(self, mock_shap_wrapper):
        """Test _validate_feature_interpretation with numerical feature."""
        mock_wrapper_instance = Mock()
        mock_wrapper_instance.calculate.return_value = self.mock_shap_values
        mock_shap_wrapper.return_value = mock_wrapper_instance
        
        explainer = BinaryMLExplainer(
            self.x_train, self.y_train, ['numerical_feature'], self.model
        )
        
        result = explainer._validate_feature_interpretation('numerical_feature')
        
        self.assertIsInstance(result, list)
        # Should have validation results for quantile groups
        self.assertGreater(len(result), 0)
    
    @patch('mlexplainer.explainers.shap.binary.ShapWrapper')
    def test_validate_feature_interpretation_categorical(self, mock_shap_wrapper):
        """Test _validate_feature_interpretation with categorical feature."""
        mock_wrapper_instance = Mock()
        mock_wrapper_instance.calculate.return_value = self.mock_shap_values
        mock_shap_wrapper.return_value = mock_wrapper_instance
        
        explainer = BinaryMLExplainer(
            self.x_train, self.y_train, ['categorical_feature'], self.model
        )
        
        result = explainer._validate_feature_interpretation('categorical_feature')
        
        self.assertIsInstance(result, list)
        # Should have validation results for each category
        self.assertGreater(len(result), 0)
    
    @patch('mlexplainer.explainers.shap.binary.ShapWrapper')
    def test_feature_categorization(self, mock_shap_wrapper):
        """Test that features are correctly categorized."""
        mock_wrapper_instance = Mock()
        mock_wrapper_instance.calculate.return_value = self.mock_shap_values
        mock_shap_wrapper.return_value = mock_wrapper_instance
        
        explainer = BinaryMLExplainer(
            self.x_train, self.y_train, self.features, self.model
        )
        
        self.assertIn('numerical_feature', explainer.numerical_features)
        self.assertIn('categorical_feature', explainer.categorical_features)
        self.assertIn('string_feature', explainer.string_features)


if __name__ == '__main__':
    unittest.main()