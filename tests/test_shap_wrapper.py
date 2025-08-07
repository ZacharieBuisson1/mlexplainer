"""Tests for SHAP wrapper functionality."""

import unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
from pandas import DataFrame

from mlexplainer.explainers.shap.wrapper import ShapWrapper


class TestShapWrapper(unittest.TestCase):
    """Test cases for ShapWrapper."""
    
    def setUp(self):
        """Set up test data."""
        self.df = DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [2.0, 4.0, 6.0, 8.0, 10.0],
            'feature3': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        self.features = ['feature1', 'feature2']
        self.model = Mock()
    
    @patch('mlexplainer.explainers.shap.wrapper.TreeExplainer')
    def test_initialization_default(self, mock_tree_explainer):
        """Test ShapWrapper initialization with default parameters."""
        mock_explainer_instance = Mock()
        mock_tree_explainer.return_value = mock_explainer_instance
        
        wrapper = ShapWrapper(self.model)
        
        self.assertEqual(wrapper.model, self.model)
        self.assertEqual(wrapper.model_output, "raw")
        self.assertIsNotNone(wrapper.shap_margin_explainer)
    
    @patch('mlexplainer.explainers.shap.wrapper.TreeExplainer')
    def test_initialization_custom_output(self, mock_tree_explainer):
        """Test ShapWrapper initialization with custom model output."""
        mock_explainer_instance = Mock()
        mock_tree_explainer.return_value = mock_explainer_instance
        
        wrapper = ShapWrapper(self.model, model_output="probability")
        
        self.assertEqual(wrapper.model, self.model)
        self.assertEqual(wrapper.model_output, "probability")
        self.assertIsNotNone(wrapper.shap_margin_explainer)
    
    @patch('mlexplainer.explainers.shap.wrapper.TreeExplainer')
    def test_tree_explainer_initialization(self, mock_tree_explainer):
        """Test that TreeExplainer is properly initialized."""
        mock_explainer_instance = Mock()
        mock_tree_explainer.return_value = mock_explainer_instance
        
        wrapper = ShapWrapper(self.model, model_output="probability")
        
        mock_tree_explainer.assert_called_once_with(
            model=self.model, 
            model_output="probability"
        )
        self.assertEqual(wrapper.shap_margin_explainer, mock_explainer_instance)
    
    @patch('mlexplainer.explainers.shap.wrapper.TreeExplainer')
    def test_calculate_method(self, mock_tree_explainer):
        """Test the calculate method."""
        # Mock the SHAP explainer
        mock_explainer_instance = Mock()
        mock_shap_values = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]])
        mock_explainer_instance.shap_values.return_value = mock_shap_values
        mock_tree_explainer.return_value = mock_explainer_instance
        
        wrapper = ShapWrapper(self.model)
        result = wrapper.calculate(self.df, self.features)
        
        # Verify that shap_values was called with the correct data
        mock_explainer_instance.shap_values.assert_called_once()
        call_args = mock_explainer_instance.shap_values.call_args[0][0]
        pd.testing.assert_frame_equal(call_args, self.df[self.features])
        
        # Verify the result
        np.testing.assert_array_equal(result, mock_shap_values)
    
    @patch('mlexplainer.explainers.shap.wrapper.TreeExplainer')
    def test_calculate_with_subset_features(self, mock_tree_explainer):
        """Test calculate method with a subset of features."""
        mock_explainer_instance = Mock()
        mock_shap_values = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])
        mock_explainer_instance.shap_values.return_value = mock_shap_values
        mock_tree_explainer.return_value = mock_explainer_instance
        
        wrapper = ShapWrapper(self.model)
        single_feature = ['feature1']
        result = wrapper.calculate(self.df, single_feature)
        
        # Verify that only the specified feature was used
        call_args = mock_explainer_instance.shap_values.call_args[0][0]
        pd.testing.assert_frame_equal(call_args, self.df[single_feature])
        
        np.testing.assert_array_equal(result, mock_shap_values)
    
    @patch('mlexplainer.explainers.shap.wrapper.TreeExplainer')
    def test_calculate_with_empty_features(self, mock_tree_explainer):
        """Test calculate method with empty features list."""
        mock_explainer_instance = Mock()
        mock_shap_values = np.array([])
        mock_explainer_instance.shap_values.return_value = mock_shap_values
        mock_tree_explainer.return_value = mock_explainer_instance
        
        wrapper = ShapWrapper(self.model)
        result = wrapper.calculate(self.df, [])
        
        # Verify that an empty dataframe was passed
        call_args = mock_explainer_instance.shap_values.call_args[0][0]
        self.assertEqual(len(call_args.columns), 0)
        
        np.testing.assert_array_equal(result, mock_shap_values)
    
    @patch('mlexplainer.explainers.shap.wrapper.TreeExplainer')
    def test_calculate_with_invalid_features(self, mock_tree_explainer):
        """Test calculate method with invalid features."""
        mock_explainer_instance = Mock()
        mock_tree_explainer.return_value = mock_explainer_instance
        
        wrapper = ShapWrapper(self.model)
        
        with self.assertRaises(KeyError):
            wrapper.calculate(self.df, ['nonexistent_feature'])


if __name__ == '__main__':
    unittest.main()