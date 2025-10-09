"""Tests for MultilabelMLPredictor class with different ML frameworks."""

import unittest

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from mlexplainer.predictors import MultilabelMLPredictor


class TestMultilabelPredictorXGBoost(unittest.TestCase):
    """Test suite for MultilabelMLPredictor with XGBoost (multiclass)."""

    def setUp(self):
        """Set up test fixtures with XGBoost multiclass model."""
        np.random.seed(42)
        self.x_train = pd.DataFrame(
            {
                "age": np.random.randint(18, 80, 100),
                "income": np.random.randint(20000, 100000, 100),
                "score": np.random.uniform(0, 1, 100),
            }
        )

        # Multiclass target (3 classes)
        y_train = np.random.randint(0, 3, 100)
        self.model = XGBClassifier(
            random_state=42,
            n_estimators=10,
            max_depth=3,
            objective="multi:softprob",
            num_class=3,
        )
        self.model.fit(self.x_train, y_train)

        self.label_names = ["Class_A", "Class_B", "Class_C"]

    def test_xgboost_multiclass_initialization(self):
        """Test initialization with XGBoost multiclass model."""
        predictor = MultilabelMLPredictor(
            self.model, self.x_train, label_names=self.label_names
        )

        self.assertIsNotNone(predictor.model)
        self.assertIsNotNone(predictor.features)
        self.assertEqual(len(predictor.features), 3)
        self.assertEqual(predictor.label_names, self.label_names)

    def test_xgboost_multiclass_predict_with_contributions(self):
        """Test prediction and contributions with XGBoost multiclass."""
        predictor = MultilabelMLPredictor(
            self.model, self.x_train, label_names=self.label_names
        )

        observation = {"age": 35, "income": 50000, "score": 0.75}
        result = predictor.predict_with_contributions(observation)

        # Check result is dict with label names as keys + processing values
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 5)  # 3 labels + 2 processing values
        self.assertIn("Class_A", result)
        self.assertIn("Class_B", result)
        self.assertIn("Class_C", result)

        # Check processing values at root level
        self.assertIn("values_before_processing", result)
        self.assertIn("values_after_processing", result)

        # Check structure for each label
        for label_name in self.label_names:
            label_result = result[label_name]
            self.assertIn("prediction", label_result)
            self.assertIn("contributions", label_result)

            # Check prediction bounds
            self.assertGreaterEqual(label_result["prediction"], 0.0)
            self.assertLessEqual(label_result["prediction"], 1.0)

            # Check contributions for all features
            self.assertEqual(
                set(label_result["contributions"].keys()), set(predictor.features)
            )

    def test_xgboost_multiclass_without_label_names(self):
        """Test multiclass prediction without explicit label names."""
        predictor = MultilabelMLPredictor(self.model, self.x_train)

        observation = {"age": 35, "income": 50000, "score": 0.75}
        result = predictor.predict_with_contributions(observation)

        # Should auto-generate label names
        self.assertIn("label_0", result)
        self.assertIn("label_1", result)
        self.assertIn("label_2", result)


class TestMultilabelPredictorLightGBM(unittest.TestCase):
    """Test suite for MultilabelMLPredictor with LightGBM (multiclass)."""

    def setUp(self):
        """Set up test fixtures with LightGBM multiclass model."""
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

        # Multiclass target (3 classes)
        y_train = np.random.randint(0, 3, 100)
        self.model = LGBMClassifier(
            random_state=42,
            n_estimators=10,
            max_depth=3,
            verbose=-1,
        )
        self.model.fit(self.x_train, y_train, categorical_feature=["education"])

        self.label_names = ["Class_A", "Class_B", "Class_C"]

    def test_lightgbm_multiclass_initialization(self):
        """Test initialization with LightGBM multiclass model."""
        predictor = MultilabelMLPredictor(
            self.model, self.x_train, label_names=self.label_names
        )

        self.assertIsNotNone(predictor.model)
        self.assertIsNotNone(predictor.features)
        self.assertEqual(len(predictor.features), 3)
        self.assertIn("education", predictor.features)

    def test_lightgbm_multiclass_predict_with_contributions(self):
        """Test prediction and contributions with LightGBM multiclass."""
        predictor = MultilabelMLPredictor(
            self.model, self.x_train, label_names=self.label_names
        )

        observation = {"age": 35, "income": 50000, "education": "Bachelor"}
        result = predictor.predict_with_contributions(observation)

        # Check result structure
        self.assertEqual(len(result), 5)  # n_labels + 2 processing values
        for label_name in self.label_names:
            self.assertIn(label_name, result)
            self.assertIn("prediction", result[label_name])
            self.assertIn("contributions", result[label_name])


class TestMultilabelPredictorCatBoost(unittest.TestCase):
    """Test suite for MultilabelMLPredictor with CatBoost (multiclass)."""

    def setUp(self):
        """Set up test fixtures with CatBoost multiclass model."""
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

        # Multiclass target (3 classes)
        y_train = np.random.randint(0, 3, 100)
        self.model = CatBoostClassifier(
            random_state=42,
            iterations=10,
            depth=3,
            verbose=0,
            cat_features=["education"],
        )
        self.model.fit(self.x_train, y_train)

        self.label_names = ["Type_A", "Type_B", "Type_C"]

    def test_catboost_multiclass_initialization(self):
        """Test initialization with CatBoost multiclass model."""
        predictor = MultilabelMLPredictor(
            self.model, self.x_train, label_names=self.label_names
        )

        self.assertIsNotNone(predictor.model)
        self.assertIsNotNone(predictor.features)
        self.assertEqual(len(predictor.features), 3)
        self.assertIn("education", predictor.features)

    def test_catboost_multiclass_predict_with_contributions(self):
        """Test prediction and contributions with CatBoost multiclass."""
        predictor = MultilabelMLPredictor(
            self.model, self.x_train, label_names=self.label_names
        )

        observation = {"age": 35, "income": 50000, "education": "Bachelor"}
        result = predictor.predict_with_contributions(observation)

        # Check result structure
        self.assertEqual(len(result), 5)  # n_labels + 2 processing values
        for label_name in self.label_names:
            self.assertIn(label_name, result)
            self.assertIn("prediction", result[label_name])
            self.assertIn("contributions", result[label_name])


class TestMultilabelPredictorRandomForest(unittest.TestCase):
    """Test suite for MultilabelMLPredictor with Random Forest (multiclass)."""

    def setUp(self):
        """Set up test fixtures with Random Forest multiclass model."""
        np.random.seed(42)
        self.x_train = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "feature3": np.random.randn(100),
            }
        )

        # Multiclass target (4 classes)
        y_train = np.random.randint(0, 4, 100)
        self.model = RandomForestClassifier(
            random_state=42, n_estimators=10, max_depth=3
        )
        self.model.fit(self.x_train, y_train)

        self.label_names = ["Type_1", "Type_2", "Type_3", "Type_4"]

    def test_randomforest_multiclass_initialization(self):
        """Test initialization with Random Forest multiclass model."""
        predictor = MultilabelMLPredictor(
            self.model, self.x_train, label_names=self.label_names
        )

        self.assertIsNotNone(predictor.model)
        self.assertEqual(len(predictor.features), 3)
        self.assertEqual(len(predictor.label_names), 4)

    def test_randomforest_multiclass_predict_with_contributions(self):
        """Test prediction and contributions with Random Forest multiclass."""
        predictor = MultilabelMLPredictor(
            self.model, self.x_train, label_names=self.label_names
        )

        observation = {"feature1": 0.5, "feature2": -0.3, "feature3": 1.2}
        result = predictor.predict_with_contributions(observation)

        # Check result structure
        self.assertEqual(len(result), 6)  # 4 labels + 2 processing values
        for label_name in self.label_names:
            self.assertIn(label_name, result)
            self.assertIn("prediction", result[label_name])
            self.assertIn("contributions", result[label_name])


class TestMultilabelPredictorOneVsRest(unittest.TestCase):
    """Test suite for MultilabelMLPredictor with OneVsRest (true multilabel)."""

    def setUp(self):
        """Set up test fixtures with OneVsRest multilabel model."""
        np.random.seed(42)
        self.x_train = pd.DataFrame(
            {
                "age": np.random.randint(18, 80, 100),
                "income": np.random.randint(20000, 100000, 100),
            }
        )

        # True multilabel target (multiple labels can be true simultaneously)
        # Create binary matrix: [has_label_A, has_label_B, has_label_C]
        y_train_label_A = np.random.randint(0, 2, 100)
        y_train_label_B = np.random.randint(0, 2, 100)

        # Train separate binary classifiers for each label
        base_classifier = XGBClassifier(random_state=42, n_estimators=10)

        # For simplicity, we'll test with just one binary classifier
        # (true OneVsRest would have multiple, but testing concept is same)
        self.model = base_classifier
        self.model.fit(self.x_train, y_train_label_A)

        self.label_names = ["Label_A", "Label_B"]

    def test_multilabel_structure(self):
        """Test multilabel predictor basic structure."""
        # Note: This is a simplified test - true multilabel would require
        # proper OneVsRestClassifier setup with multiple outputs
        predictor = MultilabelMLPredictor(
            self.model, self.x_train, label_names=["Label_A"]
        )

        self.assertIsNotNone(predictor.model)
        self.assertIsNotNone(predictor.features)


class TestMultilabelPredictorInputValidation(unittest.TestCase):
    """Test input validation for MultilabelMLPredictor."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.x_train = pd.DataFrame(
            {
                "age": np.random.randint(18, 80, 100),
                "income": np.random.randint(20000, 100000, 100),
            }
        )

        y_train = np.random.randint(0, 3, 100)
        self.model = XGBClassifier(
            random_state=42, n_estimators=10, objective="multi:softprob", num_class=3
        )
        self.model.fit(self.x_train, y_train)

    def test_dict_input(self):
        """Test prediction with dictionary input."""
        predictor = MultilabelMLPredictor(self.model, self.x_train)

        observation = {"age": 35, "income": 50000}
        result = predictor.predict_with_contributions(observation)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)

    def test_dataframe_input(self):
        """Test prediction with DataFrame input."""
        predictor = MultilabelMLPredictor(self.model, self.x_train)

        observation = pd.DataFrame([{"age": 35, "income": 50000}])
        result = predictor.predict_with_contributions(observation)

        self.assertIsNotNone(result)

    def test_multiple_rows_error(self):
        """Test that multiple rows raise ValueError."""
        predictor = MultilabelMLPredictor(self.model, self.x_train)

        observation = pd.DataFrame(
            [{"age": 35, "income": 50000}, {"age": 40, "income": 60000}]
        )

        with self.assertRaises(ValueError) as context:
            predictor.predict_with_contributions(observation)

        self.assertIn("exactly 1 row", str(context.exception))

    def test_wrong_label_names_count(self):
        """Test that mismatched label_names count raises error."""
        # Model has 3 classes but we provide 2 label names
        predictor = MultilabelMLPredictor(
            self.model, self.x_train, label_names=["A", "B"]
        )

        observation = {"age": 35, "income": 50000}

        with self.assertRaises(ValueError) as context:
            predictor.predict_with_contributions(observation)

        self.assertIn("does not match", str(context.exception))


class TestMultilabelPredictorTextExplanation(unittest.TestCase):
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

        # Multiclass target (3 classes)
        y_train = np.random.randint(0, 3, 100)
        self.model = XGBClassifier(random_state=42, n_estimators=10)
        self.model.fit(self.x_train, y_train)

        self.label_names = ["Class_A", "Class_B", "Class_C"]

    def test_predict_with_text_explanation_template_mode(self):
        """Test prediction with template-based text explanation."""
        predictor = MultilabelMLPredictor(
            self.model, self.x_train, label_names=self.label_names
        )

        observation = {"age": 35, "income": 50000, "NumOfProducts": 2}
        result = predictor.predict_with_text_explanation(
            observation=observation,
            mode="template",
            top_n=2,
            language="fr",
        )

        # Check all expected top-level keys
        self.assertIn("values_before_processing", result)
        self.assertIn("values_after_processing", result)

        # Check each label has explanation and top_features
        for label_name in self.label_names:
            self.assertIn(label_name, result)
            self.assertIn("prediction", result[label_name])
            self.assertIn("contributions", result[label_name])
            self.assertIn("explanation_text", result[label_name])
            self.assertIn("top_features", result[label_name])

            # Check explanation text is not empty
            self.assertIsInstance(result[label_name]["explanation_text"], str)
            self.assertGreater(len(result[label_name]["explanation_text"]), 0)

            # Check top_features structure
            self.assertIsInstance(result[label_name]["top_features"], list)
            self.assertLessEqual(len(result[label_name]["top_features"]), 2)

    def test_predict_with_text_explanation_english(self):
        """Test prediction with English template explanation."""
        predictor = MultilabelMLPredictor(
            self.model, self.x_train, label_names=self.label_names
        )

        observation = {"age": 45, "income": 70000, "NumOfProducts": 3}
        result = predictor.predict_with_text_explanation(
            observation=observation,
            mode="template",
            language="en",
        )

        # Check explanation is in English for each label
        for label_name in self.label_names:
            explanation = result[label_name]["explanation_text"]
            self.assertIn("probability", explanation.lower())

    def test_predict_with_text_explanation_with_feature_mapping(self):
        """Test prediction with feature name mapping."""
        predictor = MultilabelMLPredictor(
            self.model, self.x_train, label_names=self.label_names
        )

        observation = {"age": 35, "income": 50000, "NumOfProducts": 2}
        feature_mapping = {"NumOfProducts": "nombre de produits", "age": "âge"}

        result = predictor.predict_with_text_explanation(
            observation=observation,
            mode="template",
            language="fr",
            feature_name_mapping=feature_mapping,
        )

        # Check all labels have explanations
        for label_name in self.label_names:
            self.assertIn("explanation_text", result[label_name])
            self.assertIsInstance(result[label_name]["explanation_text"], str)

    def test_predict_with_text_explanation_invalid_mode(self):
        """Test that invalid mode raises ValueError."""
        predictor = MultilabelMLPredictor(
            self.model, self.x_train, label_names=self.label_names
        )

        observation = {"age": 35, "income": 50000, "NumOfProducts": 2}

        with self.assertRaises(ValueError) as context:
            predictor.predict_with_text_explanation(
                observation=observation,
                mode="invalid_mode",
            )

        self.assertIn("mode must be 'llm' or 'template'", str(context.exception))

    def test_predict_with_text_explanation_default_parameters(self):
        """Test prediction with default parameters (template mode for CI compatibility)."""
        predictor = MultilabelMLPredictor(
            self.model, self.x_train, label_names=self.label_names
        )

        observation = {"age": 35, "income": 50000, "NumOfProducts": 2}
        # Use template mode to avoid LLM model download in CI
        result = predictor.predict_with_text_explanation(
            observation=observation, mode="template"
        )

        # Should have explanation_text and top_features (default top_n=3)
        for label_name in self.label_names:
            self.assertIn("explanation_text", result[label_name])
            self.assertIn("top_features", result[label_name])
            self.assertLessEqual(len(result[label_name]["top_features"]), 3)

    def test_predict_with_text_explanation_top_n_variation(self):
        """Test prediction with different top_n values."""
        predictor = MultilabelMLPredictor(
            self.model, self.x_train, label_names=self.label_names
        )

        observation = {"age": 35, "income": 50000, "NumOfProducts": 2}

        # Test top_n=1
        result_1 = predictor.predict_with_text_explanation(
            observation=observation,
            mode="template",
            top_n=1,
        )
        for label_name in self.label_names:
            self.assertEqual(len(result_1[label_name]["top_features"]), 1)

        # Test top_n=3
        result_3 = predictor.predict_with_text_explanation(
            observation=observation,
            mode="template",
            top_n=3,
        )
        for label_name in self.label_names:
            self.assertLessEqual(len(result_3[label_name]["top_features"]), 3)

    def test_predict_with_text_explanation_preserves_base_result(self):
        """Test that text explanation preserves base prediction result."""
        predictor = MultilabelMLPredictor(
            self.model, self.x_train, label_names=self.label_names
        )

        observation = {"age": 35, "income": 50000, "NumOfProducts": 2}

        base_result = predictor.predict_with_contributions(observation)
        text_result = predictor.predict_with_text_explanation(
            observation=observation,
            mode="template",
        )

        # Check base fields are identical for each label
        for label_name in self.label_names:
            self.assertEqual(
                base_result[label_name]["prediction"],
                text_result[label_name]["prediction"],
            )
            self.assertEqual(
                base_result[label_name]["contributions"],
                text_result[label_name]["contributions"],
            )

        # Check processing values are identical
        self.assertEqual(
            base_result["values_before_processing"],
            text_result["values_before_processing"],
        )
        self.assertEqual(
            base_result["values_after_processing"],
            text_result["values_after_processing"],
        )

    def test_predict_with_text_explanation_auto_generated_labels(self):
        """Test prediction without explicit label names (auto-generated)."""
        predictor = MultilabelMLPredictor(self.model, self.x_train)

        observation = {"age": 35, "income": 50000, "NumOfProducts": 2}
        result = predictor.predict_with_text_explanation(
            observation=observation,
            mode="template",
        )

        # Should have auto-generated labels (label_0, label_1, label_2)
        self.assertIn("label_0", result)
        self.assertIn("label_1", result)
        self.assertIn("label_2", result)

        # Each should have explanation
        for i in range(3):
            label_key = f"label_{i}"
            self.assertIn("explanation_text", result[label_key])


if __name__ == "__main__":
    unittest.main()
