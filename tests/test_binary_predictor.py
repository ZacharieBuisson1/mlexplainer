"""Tests for BinaryMLPredictor class with different ML frameworks."""

import unittest

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from mlexplainer.predictors import BinaryMLPredictor


class TestBinaryPredictorXGBoost(unittest.TestCase):
    """Test suite for BinaryMLPredictor with XGBoost."""

    def setUp(self):
        """Set up test fixtures with XGBoost model."""
        np.random.seed(42)
        self.x_train = pd.DataFrame(
            {
                "age": np.random.randint(18, 80, 100),
                "income": np.random.randint(20000, 100000, 100),
                "education": pd.Categorical(
                    np.random.choice(["High School", "Bachelor", "Master"], 100)
                ),
            }
        )

        y_train = np.random.randint(0, 2, 100)
        self.model = XGBClassifier(
            random_state=42,
            n_estimators=10,
            enable_categorical=True,
            max_depth=3,
        )
        self.model.fit(self.x_train, y_train)

    def test_xgboost_initialization(self):
        """Test initialization with XGBoost model."""
        predictor = BinaryMLPredictor(self.model, self.x_train)

        self.assertIsNotNone(predictor.model)
        self.assertIsNotNone(predictor.features)
        self.assertEqual(len(predictor.features), 3)
        self.assertIn("age", predictor.features)
        self.assertIn("income", predictor.features)
        self.assertIn("education", predictor.features)

    def test_xgboost_predict_with_contributions(self):
        """Test prediction and contributions with XGBoost."""
        predictor = BinaryMLPredictor(self.model, self.x_train)

        observation = {"age": 35, "income": 50000, "education": "Bachelor"}
        result = predictor.predict_with_contributions(observation)

        # Check structure
        self.assertIn("prediction", result)
        self.assertIn("contributions", result)
        self.assertIn("values_before_processing", result)
        self.assertIn("values_after_processing", result)

        # Check prediction bounds
        self.assertGreaterEqual(result["prediction"], 0.0)
        self.assertLessEqual(result["prediction"], 1.0)

        # Check contributions for all features
        self.assertEqual(set(result["contributions"].keys()), set(predictor.features))


class TestBinaryPredictorLightGBM(unittest.TestCase):
    """Test suite for BinaryMLPredictor with LightGBM."""

    def setUp(self):
        """Set up test fixtures with LightGBM model."""
        np.random.seed(42)
        self.x_train = pd.DataFrame(
            {
                "age": np.random.randint(18, 80, 100),
                "income": np.random.randint(20000, 100000, 100),
                "education": pd.Categorical(
                    np.random.choice(["High School", "Bachelor", "Master"], 100)
                ),
            }
        )

        y_train = np.random.randint(0, 2, 100)
        self.model = LGBMClassifier(
            random_state=42,
            n_estimators=10,
            max_depth=3,
            verbose=-1,
        )
        self.model.fit(self.x_train, y_train, categorical_feature=["education"])

    def test_lightgbm_initialization(self):
        """Test initialization with LightGBM model."""
        predictor = BinaryMLPredictor(self.model, self.x_train)

        self.assertIsNotNone(predictor.model)
        self.assertIsNotNone(predictor.features)
        self.assertEqual(len(predictor.features), 3)
        self.assertIn("education", predictor.features)

    def test_lightgbm_predict_with_contributions(self):
        """Test prediction and contributions with LightGBM."""
        predictor = BinaryMLPredictor(self.model, self.x_train)

        observation = {"age": 35, "income": 50000, "education": "Bachelor"}
        result = predictor.predict_with_contributions(observation)

        # Check structure
        self.assertIn("prediction", result)
        self.assertIn("contributions", result)
        self.assertIn("values_before_processing", result)
        self.assertIn("values_after_processing", result)

        # Check prediction bounds
        self.assertGreaterEqual(result["prediction"], 0.0)
        self.assertLessEqual(result["prediction"], 1.0)

        # Check contributions for all features
        self.assertEqual(set(result["contributions"].keys()), set(predictor.features))


class TestBinaryPredictorCatBoost(unittest.TestCase):
    """Test suite for BinaryMLPredictor with CatBoost."""

    def setUp(self):
        """Set up test fixtures with CatBoost model."""
        np.random.seed(42)
        self.x_train = pd.DataFrame(
            {
                "age": np.random.randint(18, 80, 100),
                "income": np.random.randint(20000, 100000, 100),
                "education": np.random.choice(
                    ["High School", "Bachelor", "Master"], 100
                ),
            }
        )

        y_train = np.random.randint(0, 2, 100)
        self.model = CatBoostClassifier(
            random_state=42,
            iterations=10,
            depth=3,
            verbose=0,
            cat_features=["education"],
        )
        self.model.fit(self.x_train, y_train)

    def test_catboost_initialization(self):
        """Test initialization with CatBoost model."""
        predictor = BinaryMLPredictor(self.model, self.x_train)

        self.assertIsNotNone(predictor.model)
        self.assertIsNotNone(predictor.features)
        self.assertEqual(len(predictor.features), 3)
        self.assertIn("education", predictor.features)

    def test_catboost_predict_with_contributions(self):
        """Test prediction and contributions with CatBoost."""
        predictor = BinaryMLPredictor(self.model, self.x_train)

        observation = {"age": 35, "income": 50000, "education": "Bachelor"}
        result = predictor.predict_with_contributions(observation)

        # Check structure
        self.assertIn("prediction", result)
        self.assertIn("contributions", result)
        self.assertIn("values_before_processing", result)
        self.assertIn("values_after_processing", result)

        # Check prediction bounds
        self.assertGreaterEqual(result["prediction"], 0.0)
        self.assertLessEqual(result["prediction"], 1.0)

        # Check contributions for all features
        self.assertEqual(set(result["contributions"].keys()), set(predictor.features))


class TestBinaryPredictorRandomForest(unittest.TestCase):
    """Test suite for BinaryMLPredictor with Random Forest."""

    def setUp(self):
        """Set up test fixtures with Random Forest model."""
        np.random.seed(42)
        self.x_train = pd.DataFrame(
            {
                "age": np.random.randint(18, 80, 100),
                "income": np.random.randint(20000, 100000, 100),
                "score": np.random.uniform(0, 1, 100),
            }
        )

        y_train = np.random.randint(0, 2, 100)
        self.model = RandomForestClassifier(
            random_state=42, n_estimators=10, max_depth=3
        )
        self.model.fit(self.x_train, y_train)

    def test_randomforest_initialization(self):
        """Test initialization with Random Forest model."""
        predictor = BinaryMLPredictor(self.model, self.x_train)

        self.assertIsNotNone(predictor.model)
        self.assertEqual(len(predictor.features), 3)

    def test_randomforest_predict_with_contributions(self):
        """Test prediction and contributions with Random Forest."""
        predictor = BinaryMLPredictor(self.model, self.x_train)

        observation = {"age": 35, "income": 50000, "score": 0.75}
        result = predictor.predict_with_contributions(observation)

        # Check structure
        self.assertIn("prediction", result)
        self.assertIn("contributions", result)
        self.assertIn("values_before_processing", result)
        self.assertIn("values_after_processing", result)

        # Check prediction bounds
        self.assertGreaterEqual(result["prediction"], 0.0)
        self.assertLessEqual(result["prediction"], 1.0)

        # Check all features have contributions
        self.assertEqual(len(result["contributions"]), len(predictor.features))


class TestBinaryPredictorGradientBoosting(unittest.TestCase):
    """Test suite for BinaryMLPredictor with Gradient Boosting."""

    def setUp(self):
        """Set up test fixtures with Gradient Boosting model."""
        np.random.seed(42)
        self.x_train = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "feature3": np.random.randn(100),
            }
        )

        y_train = np.random.randint(0, 2, 100)
        self.model = GradientBoostingClassifier(
            random_state=42, n_estimators=10, max_depth=3
        )
        self.model.fit(self.x_train, y_train)

    def test_gradientboosting_initialization(self):
        """Test initialization with Gradient Boosting model."""
        predictor = BinaryMLPredictor(self.model, self.x_train)

        self.assertIsNotNone(predictor.model)
        self.assertEqual(len(predictor.features), 3)

    def test_gradientboosting_predict_with_contributions(self):
        """Test prediction and contributions with Gradient Boosting."""
        predictor = BinaryMLPredictor(self.model, self.x_train)

        observation = {"feature1": 0.5, "feature2": -0.3, "feature3": 1.2}
        result = predictor.predict_with_contributions(observation)

        # Check structure
        self.assertIn("prediction", result)
        self.assertIn("contributions", result)
        self.assertIn("values_before_processing", result)
        self.assertIn("values_after_processing", result)

        # Check prediction bounds
        self.assertGreaterEqual(result["prediction"], 0.0)
        self.assertLessEqual(result["prediction"], 1.0)


class TestBinaryPredictorInputValidation(unittest.TestCase):
    """Test input validation for BinaryMLPredictor."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.x_train = pd.DataFrame(
            {
                "age": np.random.randint(18, 80, 100),
                "income": np.random.randint(20000, 100000, 100),
            }
        )

        y_train = np.random.randint(0, 2, 100)
        self.model = XGBClassifier(random_state=42, n_estimators=10)
        self.model.fit(self.x_train, y_train)

    def test_dict_input(self):
        """Test prediction with dictionary input."""
        predictor = BinaryMLPredictor(self.model, self.x_train)

        observation = {"age": 35, "income": 50000}
        result = predictor.predict_with_contributions(observation)

        self.assertIsNotNone(result)

    def test_dataframe_input(self):
        """Test prediction with DataFrame input."""
        predictor = BinaryMLPredictor(self.model, self.x_train)

        observation = pd.DataFrame([{"age": 35, "income": 50000}])
        result = predictor.predict_with_contributions(observation)

        self.assertIsNotNone(result)

    def test_multiple_rows_error(self):
        """Test that multiple rows raise ValueError."""
        predictor = BinaryMLPredictor(self.model, self.x_train)

        observation = pd.DataFrame(
            [{"age": 35, "income": 50000}, {"age": 40, "income": 60000}]
        )

        with self.assertRaises(ValueError) as context:
            predictor.predict_with_contributions(observation)

        self.assertIn("exactly 1 row", str(context.exception))

    def test_missing_features_error(self):
        """Test that missing features raise ValueError."""
        predictor = BinaryMLPredictor(self.model, self.x_train)

        observation = {"age": 35}  # Missing 'income'

        with self.assertRaises(ValueError) as context:
            predictor.predict_with_contributions(observation)

        self.assertIn("missing required features", str(context.exception))


class TestBinaryPredictorTextExplanation(unittest.TestCase):
    """Test suite for predict_with_text_explanation method."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.x_train = pd.DataFrame(
            {
                "age": np.random.randint(18, 80, 100),
                "income": np.random.randint(20000, 100000, 100),
                "NumOfProducts": np.random.randint(1, 5, 100),
            }
        )

        y_train = np.random.randint(0, 2, 100)
        self.model = XGBClassifier(random_state=42, n_estimators=10)
        self.model.fit(self.x_train, y_train)

    def test_predict_with_text_explanation_template_mode(self):
        """Test prediction with template-based text explanation."""
        predictor = BinaryMLPredictor(self.model, self.x_train)

        observation = {"age": 35, "income": 50000, "NumOfProducts": 2}
        result = predictor.predict_with_text_explanation(
            observation=observation,
            mode="template",
            top_n=2,
            language="fr",
            target_name="churn",
        )

        # Check all expected keys
        self.assertIn("prediction", result)
        self.assertIn("contributions", result)
        self.assertIn("values_before_processing", result)
        self.assertIn("values_after_processing", result)
        self.assertIn("explanation_text", result)
        self.assertIn("top_features", result)

        # Check explanation text is not empty
        self.assertIsInstance(result["explanation_text"], str)
        self.assertGreater(len(result["explanation_text"]), 0)

        # Check top_features structure
        self.assertIsInstance(result["top_features"], list)
        self.assertLessEqual(len(result["top_features"]), 2)
        if len(result["top_features"]) > 0:
            self.assertIn("name", result["top_features"][0])
            self.assertIn("value", result["top_features"][0])
            self.assertIn("contribution", result["top_features"][0])
            self.assertIn("impact", result["top_features"][0])

    def test_predict_with_text_explanation_english(self):
        """Test prediction with English template explanation."""
        predictor = BinaryMLPredictor(self.model, self.x_train)

        observation = {"age": 45, "income": 70000, "NumOfProducts": 3}
        result = predictor.predict_with_text_explanation(
            observation=observation,
            mode="template",
            language="en",
            target_name="retention",
        )

        # Check explanation is in English
        explanation = result["explanation_text"]
        self.assertIn("probability", explanation.lower())

    def test_predict_with_text_explanation_with_feature_mapping(self):
        """Test prediction with feature name mapping."""
        predictor = BinaryMLPredictor(self.model, self.x_train)

        observation = {"age": 35, "income": 50000, "NumOfProducts": 2}
        feature_mapping = {"NumOfProducts": "nombre de produits", "age": "âge"}

        result = predictor.predict_with_text_explanation(
            observation=observation,
            mode="template",
            language="fr",
            target_name="churn",
            feature_name_mapping=feature_mapping,
        )

        # Check mapped names appear in explanation (if those features are in top_n)
        explanation = result["explanation_text"]
        self.assertIsInstance(explanation, str)

    def test_predict_with_text_explanation_invalid_mode(self):
        """Test that invalid mode raises ValueError."""
        predictor = BinaryMLPredictor(self.model, self.x_train)

        observation = {"age": 35, "income": 50000, "NumOfProducts": 2}

        with self.assertRaises(ValueError) as context:
            predictor.predict_with_text_explanation(
                observation=observation,
                mode="invalid_mode",
            )

        self.assertIn("mode must be 'llm' or 'template'", str(context.exception))

    def test_predict_with_text_explanation_default_parameters(self):
        """Test prediction with default parameters."""
        predictor = BinaryMLPredictor(self.model, self.x_train)

        observation = {"age": 35, "income": 50000, "NumOfProducts": 2}
        result = predictor.predict_with_text_explanation(observation=observation)

        # Should use default mode='llm', language='fr', top_n=3
        self.assertIn("explanation_text", result)
        self.assertIn("top_features", result)
        self.assertLessEqual(len(result["top_features"]), 3)

    def test_predict_with_text_explanation_top_n_variation(self):
        """Test prediction with different top_n values."""
        predictor = BinaryMLPredictor(self.model, self.x_train)

        observation = {"age": 35, "income": 50000, "NumOfProducts": 2}

        # Test top_n=1
        result_1 = predictor.predict_with_text_explanation(
            observation=observation,
            mode="template",
            top_n=1,
        )
        self.assertEqual(len(result_1["top_features"]), 1)

        # Test top_n=3
        result_3 = predictor.predict_with_text_explanation(
            observation=observation,
            mode="template",
            top_n=3,
        )
        self.assertLessEqual(len(result_3["top_features"]), 3)

    def test_predict_with_text_explanation_preserves_base_result(self):
        """Test that text explanation preserves base prediction result."""
        predictor = BinaryMLPredictor(self.model, self.x_train)

        observation = {"age": 35, "income": 50000, "NumOfProducts": 2}

        base_result = predictor.predict_with_contributions(observation)
        text_result = predictor.predict_with_text_explanation(
            observation=observation,
            mode="template",
        )

        # Check base fields are identical
        self.assertEqual(base_result["prediction"], text_result["prediction"])
        self.assertEqual(base_result["contributions"], text_result["contributions"])
        self.assertEqual(
            base_result["values_before_processing"],
            text_result["values_before_processing"],
        )


if __name__ == "__main__":
    unittest.main()
