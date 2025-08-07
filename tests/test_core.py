"""Tests for core module functionality."""

import unittest
from unittest.mock import Mock

import pandas as pd
from pandas import DataFrame, Series

from mlexplainer.core.base_explainer import BaseMLExplainer


class ConcreteMLExplainer(BaseMLExplainer):
    """Concrete implementation of BaseMLExplainer for testing."""
    
    def explain(self, **kwargs):
        """Concrete implementation of explain method."""
        return {"explained": True}
    
    def correctness_features(self, q=None):
        """Concrete implementation of correctness_features method."""
        return {feature: True for feature in self.features}


class TestBaseMLExplainer(unittest.TestCase):
    """Test cases for BaseMLExplainer."""
    
    def setUp(self):
        """Set up test data."""
        self.x_train = DataFrame({
            'numerical_feature': [1.0, 2.0, 3.0, 4.0, 5.0],
            'categorical_feature': pd.Categorical(['A', 'B', 'A', 'C', 'B']),
            'string_feature': ['text1', 'text2', 'text3', 'text4', 'text5']
        })
        self.y_train = Series([0, 1, 0, 1, 0])
        self.features = ['numerical_feature', 'categorical_feature']
        self.model = Mock()
    
    def test_initialization_valid(self):
        """Test valid initialization of BaseMLExplainer."""
        explainer = ConcreteMLExplainer(
            self.x_train, self.y_train, self.features, self.model
        )
        
        self.assertIsInstance(explainer, BaseMLExplainer)
        self.assertEqual(explainer.features, self.features)
        self.assertEqual(explainer.model, self.model)
        self.assertTrue(explainer.global_explainer)
        self.assertTrue(explainer.local_explainer)
    
    def test_initialization_with_flags(self):
        """Test initialization with custom flags."""
        explainer = ConcreteMLExplainer(
            self.x_train, self.y_train, self.features, self.model,
            global_explainer=False, local_explainer=False
        )
        
        self.assertFalse(explainer.global_explainer)
        self.assertFalse(explainer.local_explainer)
    
    def test_feature_categorization(self):
        """Test automatic feature categorization."""
        explainer = ConcreteMLExplainer(
            self.x_train, self.y_train, self.features, self.model
        )
        
        self.assertEqual(explainer.numerical_features, ['numerical_feature'])
        self.assertEqual(explainer.categorical_features, ['categorical_feature'])
        self.assertEqual(explainer.string_features, ['string_feature'])
    
    def test_invalid_x_train_none(self):
        """Test error when x_train is None."""
        with self.assertRaises(AttributeError):
            # This will fail because we try to access None.columns
            ConcreteMLExplainer(None, self.y_train, self.features, self.model)
    
    def test_invalid_y_train_none(self):
        """Test error when y_train is None."""
        with self.assertRaises(ValueError) as context:
            ConcreteMLExplainer(self.x_train, None, self.features, self.model)
        
        self.assertIn("X_train and y_train must be provided", str(context.exception))
    
    def test_invalid_no_features(self):
        """Test error when no features provided."""
        with self.assertRaises(ValueError) as context:
            ConcreteMLExplainer(self.x_train, self.y_train, [], self.model)
        
        self.assertIn("At least one feature must be provided", str(context.exception))
    
    def test_invalid_missing_features(self):
        """Test error when features not in x_train."""
        invalid_features = ['nonexistent_feature']
        with self.assertRaises(ValueError) as context:
            ConcreteMLExplainer(self.x_train, self.y_train, invalid_features, self.model)
        
        self.assertIn("All features must be present in x_train", str(context.exception))
        self.assertIn("nonexistent_feature", str(context.exception))
    
    def test_mixed_valid_invalid_features(self):
        """Test error when some features are missing."""
        mixed_features = ['numerical_feature', 'nonexistent_feature']
        with self.assertRaises(ValueError) as context:
            ConcreteMLExplainer(self.x_train, self.y_train, mixed_features, self.model)
        
        self.assertIn("All features must be present in x_train", str(context.exception))
        self.assertIn("nonexistent_feature", str(context.exception))
    
    def test_explain_method_exists(self):
        """Test that explain method is implemented."""
        explainer = ConcreteMLExplainer(
            self.x_train, self.y_train, self.features, self.model
        )
        result = explainer.explain()
        self.assertEqual(result, {"explained": True})
    
    def test_correctness_features_method_exists(self):
        """Test that correctness_features method is implemented."""
        explainer = ConcreteMLExplainer(
            self.x_train, self.y_train, self.features, self.model
        )
        result = explainer.correctness_features()
        expected = {feature: True for feature in self.features}
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()