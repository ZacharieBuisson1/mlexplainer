"""Tests for Multilabel ML Explainer functionality."""

import unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from mlexplainer.explainers.shap.multilabel import MultilabelMLExplainer


class TestMultilabelMLExplainer(unittest.TestCase):
    """Test cases for MultilabelMLExplainer."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.x_train = DataFrame({
            'numerical_feature': np.random.randn(100),
            'categorical_feature': pd.Categorical(np.random.choice(['A', 'B', 'C'], 100)),
            'string_feature': np.random.choice(['text1', 'text2', 'text3'], 100)
        })
        # Multilabel target with multiple numeric classes
        self.y_train = Series(np.random.choice([0, 1, 2], 100))
        self.features = ['numerical_feature', 'categorical_feature']
        self.model = Mock()
        
        # Mock SHAP values for multilabel (3D array for multi-output)
        self.mock_shap_values = np.random.randn(100, len(self.features), 3)
    
    @patch('mlexplainer.explainers.shap.multilabel.ShapWrapper')
    def test_initialization_valid_multilabel(self, mock_shap_wrapper):
        """Test valid initialization with multilabel target."""
        mock_wrapper_instance = Mock()
        mock_wrapper_instance.calculate.return_value = self.mock_shap_values
        mock_shap_wrapper.return_value = mock_wrapper_instance
        
        explainer = MultilabelMLExplainer(
            self.x_train, self.y_train, self.features, self.model
        )
        
        self.assertIsInstance(explainer, MultilabelMLExplainer)
        self.assertEqual(explainer.features, self.features)
        self.assertEqual(explainer.model, self.model)
        self.assertIsNotNone(explainer.shap_values_train)
        self.assertIsNotNone(explainer.ymean_train)
        self.assertIsNotNone(explainer.modalities)
        self.assertEqual(len(explainer.modalities), 3)
    
    def test_initialization_single_class_target(self):
        """Test error when target has only one class."""
        y_single_class = Series([0, 0, 0, 0])
        
        with self.assertRaises(ValueError) as context:
            MultilabelMLExplainer(self.x_train, y_single_class, self.features, self.model)
        
        self.assertIn("at least two unique values", str(context.exception))
    
    @patch('mlexplainer.explainers.shap.multilabel.ShapWrapper')
    @patch('mlexplainer.explainers.shap.multilabel.plt')
    def test_explain_global_only(self, mock_plt, mock_shap_wrapper):
        """Test explain method with global explainer only."""
        mock_wrapper_instance = Mock()
        mock_wrapper_instance.calculate.return_value = self.mock_shap_values
        mock_shap_wrapper.return_value = mock_wrapper_instance
        
        # Mock subplots to return proper fig, ax tuple
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        explainer = MultilabelMLExplainer(
            self.x_train, self.y_train, self.features, self.model,
            global_explainer=True, local_explainer=False
        )
        
        explainer.explain()
        
        # Should call plotting functions
        self.assertTrue(mock_plt.show.called)
        self.assertTrue(mock_plt.subplots.called)
    
    @patch('mlexplainer.explainers.shap.multilabel.ShapWrapper')
    @patch('mlexplainer.explainers.shap.multilabel.plt')
    @patch('mlexplainer.explainers.shap.multilabel.plot_feature_target_numerical_multilabel')
    @patch('mlexplainer.explainers.shap.multilabel.plot_shap_values_numerical_multilabel')
    @patch('mlexplainer.explainers.shap.multilabel.plot_feature_target_categorical_multilabel')
    @patch('mlexplainer.explainers.shap.multilabel.plot_shap_values_categorical_multilabel')
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
        mock_shap_plot.return_value = mock_ax
        mock_cat_target_plot.return_value = mock_ax
        mock_cat_shap_plot.return_value = mock_ax
        
        explainer = MultilabelMLExplainer(
            self.x_train, self.y_train, self.features, self.model,
            global_explainer=False, local_explainer=True
        )
        
        explainer.explain()
        
        # Should call numerical plotting functions
        self.assertTrue(mock_target_plot.called)
        self.assertTrue(mock_shap_plot.called)
    
    @patch('mlexplainer.explainers.shap.multilabel.ShapWrapper')
    @patch('mlexplainer.explainers.shap.multilabel.plt')
    @patch('mlexplainer.explainers.shap.multilabel.plot_feature_target_categorical_multilabel')
    @patch('mlexplainer.explainers.shap.multilabel.plot_shap_values_categorical_multilabel')
    @patch('mlexplainer.explainers.shap.multilabel.plot_feature_target_numerical_multilabel')
    @patch('mlexplainer.explainers.shap.multilabel.plot_shap_values_numerical_multilabel')
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
        mock_shap_plot.return_value = mock_ax
        mock_num_target_plot.return_value = mock_ax
        mock_num_shap_plot.return_value = mock_ax
        
        explainer = MultilabelMLExplainer(
            self.x_train, self.y_train, self.features, self.model,
            global_explainer=False, local_explainer=True
        )
        
        explainer.explain()
        
        # Should call categorical plotting functions
        self.assertTrue(mock_target_plot.called)
        self.assertTrue(mock_shap_plot.called)
    
    @patch('mlexplainer.explainers.shap.multilabel.ShapWrapper')
    def test_correctness_features(self, mock_shap_wrapper):
        """Test correctness_features method."""
        mock_wrapper_instance = Mock()
        mock_wrapper_instance.calculate.return_value = self.mock_shap_values
        mock_shap_wrapper.return_value = mock_wrapper_instance
        
        explainer = MultilabelMLExplainer(
            self.x_train, self.y_train, self.features, self.model
        )
        
        result = explainer.correctness_features()
        
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), set(self.features))
        
        # Each feature should have validation results for each modality
        for feature, validation_result in result.items():
            self.assertIsInstance(validation_result, dict)
            # Should have results for each modality
            for modality in explainer.modalities:
                self.assertIn(modality, validation_result)
    
    @patch('mlexplainer.explainers.shap.multilabel.ShapWrapper')
    def test_correctness_features_valid(self, mock_shap_wrapper):
        """Test correctness_features with valid features."""
        mock_wrapper_instance = Mock()
        mock_wrapper_instance.calculate.return_value = self.mock_shap_values
        mock_shap_wrapper.return_value = mock_wrapper_instance
        
        explainer = MultilabelMLExplainer(
            self.x_train, self.y_train, self.features, self.model
        )
        
        result = explainer.correctness_features()
        self.assertIsInstance(result, dict)
        # Should have results for all features
        for feature in self.features:
            self.assertIn(feature, result)
    
    @patch('mlexplainer.explainers.shap.multilabel.ShapWrapper')
    def test_correctness_features_numerical(self, mock_shap_wrapper):
        """Test correctness_features with numerical feature."""
        mock_wrapper_instance = Mock()
        mock_wrapper_instance.calculate.return_value = self.mock_shap_values
        mock_shap_wrapper.return_value = mock_wrapper_instance
        
        explainer = MultilabelMLExplainer(
            self.x_train, self.y_train, ['numerical_feature'], self.model
        )
        
        result = explainer.correctness_features()
        
        self.assertIsInstance(result, dict)
        self.assertIn('numerical_feature', result)
        # Should have results for each modality
        for modality in explainer.modalities:
            self.assertIn(modality, result['numerical_feature'])
            self.assertIsInstance(result['numerical_feature'][modality], list)
    
    @patch('mlexplainer.explainers.shap.multilabel.ShapWrapper')
    def test_correctness_features_categorical(self, mock_shap_wrapper):
        """Test correctness_features with categorical feature."""
        mock_wrapper_instance = Mock()
        mock_wrapper_instance.calculate.return_value = self.mock_shap_values
        mock_shap_wrapper.return_value = mock_wrapper_instance
        
        explainer = MultilabelMLExplainer(
            self.x_train, self.y_train, ['categorical_feature'], self.model
        )
        
        result = explainer.correctness_features()
        
        self.assertIsInstance(result, dict)
        self.assertIn('categorical_feature', result)
        # Should have results for each modality
        for modality in explainer.modalities:
            self.assertIn(modality, result['categorical_feature'])
            self.assertIsInstance(result['categorical_feature'][modality], list)
    
    @patch('mlexplainer.explainers.shap.multilabel.ShapWrapper')
    def test_correctness_features_no_shap_values(self, mock_shap_wrapper):
        """Test correctness_features when SHAP values are None."""
        mock_wrapper_instance = Mock()
        mock_wrapper_instance.calculate.return_value = None
        mock_shap_wrapper.return_value = mock_wrapper_instance
        
        explainer = MultilabelMLExplainer(
            self.x_train, self.y_train, self.features, self.model
        )
        # Override shap_values_train to None
        explainer.shap_values_train = None
        
        result = explainer.correctness_features()
        
        self.assertIsInstance(result, dict)
        # Should return False for all features and modalities
        for feature in self.features:
            self.assertIn(feature, result)
            for modality in explainer.modalities:
                self.assertFalse(result[feature][modality])
    
    @patch('mlexplainer.explainers.shap.multilabel.ShapWrapper')
    def test_modalities_extraction(self, mock_shap_wrapper):
        """Test that modalities are correctly extracted from y_train."""
        mock_wrapper_instance = Mock()
        mock_wrapper_instance.calculate.return_value = self.mock_shap_values
        mock_shap_wrapper.return_value = mock_wrapper_instance
        
        explainer = MultilabelMLExplainer(
            self.x_train, self.y_train, self.features, self.model
        )
        
        expected_modalities = self.y_train.unique()
        np.testing.assert_array_equal(
            sorted(explainer.modalities), 
            sorted(expected_modalities)
        )
    
    @patch('mlexplainer.explainers.shap.multilabel.ShapWrapper')
    def test_feature_categorization(self, mock_shap_wrapper):
        """Test that features are correctly categorized."""
        mock_wrapper_instance = Mock()
        mock_wrapper_instance.calculate.return_value = self.mock_shap_values
        mock_shap_wrapper.return_value = mock_wrapper_instance
        
        explainer = MultilabelMLExplainer(
            self.x_train, self.y_train, self.features, self.model
        )
        
        self.assertIn('numerical_feature', explainer.numerical_features)
        self.assertIn('categorical_feature', explainer.categorical_features)
        self.assertIn('string_feature', explainer.string_features)


if __name__ == '__main__':
    unittest.main()