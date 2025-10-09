"""Tests for text explanation classes."""

import unittest
from unittest.mock import MagicMock, patch

from mlexplainer.core import BaseTextExplainer
from mlexplainer.interpretation import TextExplainerTemplate, TextExplainerLLM
from mlexplainer.interpretation.model_cache import LLMModelCache


class ConcreteTextExplainer(BaseTextExplainer):
    """Concrete implementation for testing abstract base class."""

    def generate_explanation(
        self, prediction, contributions, values, top_n=3, target_name=None, **kwargs
    ):
        """Simple implementation for testing."""
        return f"Prediction: {prediction:.1%}"


class TestBaseTextExplainer(unittest.TestCase):
    """Test suite for BaseTextExplainer abstract base class."""

    def test_initialization_valid_language_fr(self):
        """Test initialization with valid French language."""
        explainer = ConcreteTextExplainer(language="fr")
        self.assertEqual(explainer.language, "fr")

    def test_initialization_valid_language_en(self):
        """Test initialization with valid English language."""
        explainer = ConcreteTextExplainer(language="en")
        self.assertEqual(explainer.language, "en")

    def test_initialization_invalid_language(self):
        """Test initialization with invalid language raises ValueError."""
        with self.assertRaises(ValueError) as context:
            ConcreteTextExplainer(language="es")

        self.assertIn("must be 'fr' or 'en'", str(context.exception))

    def test_rank_features_by_contribution(self):
        """Test feature ranking by absolute contribution."""
        explainer = ConcreteTextExplainer()
        contributions = {"age": 0.15, "income": -0.21, "score": 0.03}

        ranked = explainer._rank_features_by_contribution(contributions)

        # Should be sorted by absolute value (descending)
        self.assertEqual(len(ranked), 3)
        self.assertEqual(ranked[0]["name"], "income")
        self.assertAlmostEqual(ranked[0]["abs_contribution"], 0.21)
        self.assertEqual(ranked[0]["impact"], "negative")

        self.assertEqual(ranked[1]["name"], "age")
        self.assertAlmostEqual(ranked[1]["abs_contribution"], 0.15)
        self.assertEqual(ranked[1]["impact"], "positive")

        self.assertEqual(ranked[2]["name"], "score")
        self.assertAlmostEqual(ranked[2]["abs_contribution"], 0.03)
        self.assertEqual(ranked[2]["impact"], "positive")

    def test_extract_top_features(self):
        """Test extraction of top N features."""
        explainer = ConcreteTextExplainer()
        contributions = {"age": 0.15, "income": -0.21, "score": 0.03, "status": 0.05}
        values = {"age": 35, "income": 50000, "score": 0.75, "status": "active"}

        top_features = explainer._extract_top_features(contributions, values, top_n=2)

        # Should return top 2 by absolute contribution
        self.assertEqual(len(top_features), 2)

        # First should be income
        self.assertEqual(top_features[0]["name"], "income")
        self.assertEqual(top_features[0]["value"], 50000)
        self.assertAlmostEqual(top_features[0]["contribution"], -0.21)
        self.assertEqual(top_features[0]["impact"], "negative")

        # Second should be age
        self.assertEqual(top_features[1]["name"], "age")
        self.assertEqual(top_features[1]["value"], 35)
        self.assertAlmostEqual(top_features[1]["contribution"], 0.15)
        self.assertEqual(top_features[1]["impact"], "positive")

    def test_extract_top_features_default_top_n(self):
        """Test extraction with default top_n=3."""
        explainer = ConcreteTextExplainer()
        contributions = {
            "a": 0.1,
            "b": -0.2,
            "c": 0.3,
            "d": -0.05,
            "e": 0.15,
        }
        values = {k: f"val_{k}" for k in contributions.keys()}

        top_features = explainer._extract_top_features(contributions, values)

        # Should return top 3
        self.assertEqual(len(top_features), 3)
        self.assertEqual(top_features[0]["name"], "c")  # 0.3
        self.assertEqual(top_features[1]["name"], "b")  # -0.2
        self.assertEqual(top_features[2]["name"], "e")  # 0.15


class TestTextExplainerTemplate(unittest.TestCase):
    """Test suite for TextExplainerTemplate."""

    def test_initialization_default(self):
        """Test initialization with defaults."""
        explainer = TextExplainerTemplate()
        self.assertEqual(explainer.language, "fr")
        self.assertEqual(explainer.feature_name_mapping, {})

    def test_initialization_with_mapping(self):
        """Test initialization with feature name mapping."""
        mapping = {"NumOfProducts": "nombre de produits"}
        explainer = TextExplainerTemplate(feature_name_mapping=mapping)
        self.assertEqual(explainer.feature_name_mapping, mapping)

    def test_generate_explanation_french_positive_contribution(self):
        """Test French explanation with positive contribution."""
        explainer = TextExplainerTemplate(language="fr")
        prediction = 0.78
        contributions = {"age": 0.15, "income": 0.21, "score": 0.03}
        values = {"age": 35, "income": 50000, "score": 0.75}

        explanation = explainer.generate_explanation(
            prediction=prediction,
            contributions=contributions,
            values=values,
            top_n=2,
            target_name="churn",
        )

        # Check structure
        self.assertIn("probabilité de churn est de 78.0%", explanation)
        self.assertIn("income", explanation)
        self.assertIn("50000", explanation)
        self.assertIn("+21%", explanation)

    def test_generate_explanation_french_negative_contribution(self):
        """Test French explanation with negative contribution."""
        explainer = TextExplainerTemplate(language="fr")
        prediction = 0.42
        contributions = {"age": -0.15, "income": -0.21}
        values = {"age": 65, "income": 30000}

        explanation = explainer.generate_explanation(
            prediction=prediction,
            contributions=contributions,
            values=values,
            top_n=2,
            target_name="approval",
        )

        # Check negative sign is explicitly shown
        self.assertIn("probabilité de approval est de 42.0%", explanation)
        self.assertIn("income", explanation)
        self.assertIn("-21%", explanation)

    def test_generate_explanation_english(self):
        """Test English explanation."""
        explainer = TextExplainerTemplate(language="en")
        prediction = 0.65
        contributions = {"age": 0.10, "score": 0.05}
        values = {"age": 40, "score": 0.8}

        explanation = explainer.generate_explanation(
            prediction=prediction,
            contributions=contributions,
            values=values,
            top_n=2,
            target_name="success",
        )

        # Check English structure
        self.assertIn("probability of success is 65.0%", explanation)
        self.assertIn("age", explanation)
        self.assertIn("+10%", explanation)

    def test_generate_explanation_with_feature_mapping(self):
        """Test explanation with feature name mapping."""
        mapping = {"NumOfProducts": "nombre de produits", "IsActiveMember": "statut actif"}
        explainer = TextExplainerTemplate(language="fr", feature_name_mapping=mapping)

        prediction = 0.55
        contributions = {"NumOfProducts": 0.20, "IsActiveMember": 0.15}
        values = {"NumOfProducts": 3, "IsActiveMember": 1}

        explanation = explainer.generate_explanation(
            prediction=prediction,
            contributions=contributions,
            values=values,
            top_n=2,
            target_name="rétention",
        )

        # Check mapped names are used
        self.assertIn("nombre de produits", explanation)
        self.assertIn("statut actif", explanation)
        self.assertNotIn("NumOfProducts", explanation)
        self.assertNotIn("IsActiveMember", explanation)

    def test_generate_explanation_without_target_name(self):
        """Test explanation generation without target name."""
        explainer = TextExplainerTemplate(language="fr")
        prediction = 0.60
        contributions = {"age": 0.10}
        values = {"age": 30}

        explanation = explainer.generate_explanation(
            prediction=prediction,
            contributions=contributions,
            values=values,
            top_n=1,
        )

        # Should use "classe positive" as default
        self.assertIn("probabilité de classe positive est de 60.0%", explanation)

    def test_generate_explanation_top_n_larger_than_features(self):
        """Test when top_n exceeds number of features."""
        explainer = TextExplainerTemplate(language="fr")
        prediction = 0.50
        contributions = {"age": 0.10, "income": 0.05}
        values = {"age": 30, "income": 40000}

        explanation = explainer.generate_explanation(
            prediction=prediction,
            contributions=contributions,
            values=values,
            top_n=10,  # More than available features
            target_name="test",
        )

        # Should include all available features
        self.assertIn("age", explanation)
        self.assertIn("income", explanation)


class TestTextExplainerLLM(unittest.TestCase):
    """Test suite for TextExplainerLLM."""

    def test_initialization_default(self):
        """Test initialization with defaults."""
        explainer = TextExplainerLLM(language="fr")
        self.assertEqual(explainer.language, "fr")
        self.assertEqual(explainer.model_name, "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertIsNone(explainer.quantization)
        self.assertEqual(explainer.max_new_tokens, 300)
        self.assertEqual(explainer.temperature, 0.5)

    def test_initialization_custom_parameters(self):
        """Test initialization with custom parameters."""
        explainer = TextExplainerLLM(
            language="en",
            model_name="custom/model",
            quantization="4bit",
            max_new_tokens=200,
            temperature=0.7,
        )
        self.assertEqual(explainer.language, "en")
        self.assertEqual(explainer.model_name, "custom/model")
        self.assertEqual(explainer.quantization, "4bit")
        self.assertEqual(explainer.max_new_tokens, 200)
        self.assertEqual(explainer.temperature, 0.7)

    def test_generate_explanation_structure(self):
        """Test that generate_explanation returns proper structure without actual model loading."""
        # This test verifies the basic initialization without loading the model
        # Actual generation tests are skipped in CI due to disk space constraints
        explainer = TextExplainerLLM(language="fr")

        # Verify initialization worked
        self.assertEqual(explainer.language, "fr")
        self.assertEqual(explainer.model_name, "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertIsNone(explainer.quantization)

        # Note: Actual model generation test removed to avoid CI disk space issues
        # with downloading 3GB+ model files


class TestLLMModelCache(unittest.TestCase):
    """Test suite for LLMModelCache singleton."""

    def setUp(self):
        """Reset singleton instance before each test."""
        LLMModelCache._instance = None

    def test_singleton_pattern(self):
        """Test that LLMModelCache implements singleton pattern."""
        cache1 = LLMModelCache()
        cache2 = LLMModelCache()

        # Should be the same instance
        self.assertIs(cache1, cache2)

    def test_cache_initialization(self):
        """Test cache initializes with empty models dict."""
        cache = LLMModelCache()
        self.assertIsNotNone(cache._models)
        self.assertIsInstance(cache._models, dict)

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    def test_get_model_caches_result(self, mock_tokenizer_class, mock_model_class):
        """Test that get_model caches the loaded model."""
        # Setup mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"

        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Test
        cache = LLMModelCache()
        model1, tokenizer1 = cache.get_model("test/model", quantization=None)
        model2, tokenizer2 = cache.get_model("test/model", quantization=None)

        # Should return same instances (cached)
        self.assertIs(model1, model2)
        self.assertIs(tokenizer1, tokenizer2)

        # Should only load once
        self.assertEqual(mock_model_class.from_pretrained.call_count, 1)
        self.assertEqual(mock_tokenizer_class.from_pretrained.call_count, 1)

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    def test_get_model_different_quantization(self, mock_tokenizer_class, mock_model_class):
        """Test that different quantization settings create separate cache entries."""
        # Setup mocks
        mock_model_class.from_pretrained.return_value = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Test
        cache = LLMModelCache()
        cache.get_model("test/model", quantization=None)
        cache.get_model("test/model", quantization="4bit")

        # Should load twice (different cache keys)
        self.assertEqual(mock_model_class.from_pretrained.call_count, 2)


if __name__ == "__main__":
    unittest.main()
