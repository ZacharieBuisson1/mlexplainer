"""Tests for validation module functions."""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch

from mlexplainer.validation.feature_interpretation import validate_single_feature_interpretation


class TestValidationFeatureInterpretation(unittest.TestCase):
    """Test the validate_single_feature_interpretation function."""

    def setUp(self):
        """Set up test data."""
        # Create sample data
        self.x_train = pd.DataFrame({
            'numerical_feature': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'categorical_feature': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
        })
        self.y_binary = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        self.feature_shap_values_num = pd.Series([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0])
        self.feature_shap_values_cat = pd.Series([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0])
        self.numerical_features = ['numerical_feature']
        self.ymean_binary = self.y_binary.mean()

    def test_validate_numerical_feature(self):
        """Test validation with numerical feature."""
        result = validate_single_feature_interpretation(
            x_train=self.x_train,
            y_binary=self.y_binary,
            feature='numerical_feature',
            feature_shap_values=self.feature_shap_values_num,
            numerical_features=self.numerical_features,
            ymean_binary=self.ymean_binary,
            q=None
        )
        
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
        # Each element should be a tuple with 3 elements for numerical features
        for item in result:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 3)  # (start, end, is_consistent)

    def test_validate_categorical_feature(self):
        """Test validation with categorical feature."""
        result = validate_single_feature_interpretation(
            x_train=self.x_train,
            y_binary=self.y_binary,
            feature='categorical_feature',
            feature_shap_values=self.feature_shap_values_cat,
            numerical_features=self.numerical_features,
            ymean_binary=self.ymean_binary,
            q=None
        )
        
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
        # Each element should be a tuple with 2 elements for categorical features
        for item in result:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)  # (category, is_consistent)

    def test_validate_with_missing_values(self):
        """Test validation with missing values."""
        # Add missing values
        x_train_with_nan = self.x_train.copy()
        x_train_with_nan.loc[0, 'numerical_feature'] = np.nan
        
        result = validate_single_feature_interpretation(
            x_train=x_train_with_nan,
            y_binary=self.y_binary,
            feature='numerical_feature',
            feature_shap_values=self.feature_shap_values_num,
            numerical_features=self.numerical_features,
            ymean_binary=self.ymean_binary,
            q=None
        )
        
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_validate_with_quantiles(self):
        """Test validation with specified number of quantiles."""
        result = validate_single_feature_interpretation(
            x_train=self.x_train,
            y_binary=self.y_binary,
            feature='numerical_feature',
            feature_shap_values=self.feature_shap_values_num,
            numerical_features=self.numerical_features,
            ymean_binary=self.ymean_binary,
            q=5
        )
        
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)


if __name__ == '__main__':
    unittest.main()