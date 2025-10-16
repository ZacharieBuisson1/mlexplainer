"""Template-based text explanation generator using f-strings."""

from typing import Any, Dict, Optional

from mlexplainer.core.base_text_explainer import BaseTextExplainer


class TextExplainerTemplate(BaseTextExplainer):
    """Template-based text explanation generator.

    Uses pre-defined f-string templates to generate fast, deterministic explanations.
    Ideal for production environments where speed is critical (<50ms).

    Features:
    - Very fast generation (<50ms)
    - No external dependencies
    - Deterministic output
    - Optional feature name mapping for human-readable names

    Attributes:
        feature_name_mapping (Optional[Dict[str, str]]): Mapping from technical
            feature names to human-readable names.
    """

    # Template strings per language
    TEMPLATES = {
        "fr": """La probabilité de {target_name} est de {prediction:.1%}. Cette prédiction s'explique principalement par {features_list}.""",
        "en": """The probability of {target_name} is {prediction:.1%}. This prediction is mainly explained by {features_list}.""",
    }

    FEATURE_TEMPLATES = {
        "fr": {
            "positive": "{feature_name} (valeur: {value}, contribution: +{contrib:.0%})",
            "negative": "{feature_name} (valeur: {value}, contribution: -{contrib:.0%})",
        },
        "en": {
            "positive": "{feature_name} (value: {value}, contribution: +{contrib:.0%})",
            "negative": "{feature_name} (value: {value}, contribution: -{contrib:.0%})",
        },
    }

    def __init__(
        self, language: str = "fr", feature_name_mapping: Optional[Dict[str, str]] = None
    ):
        """Initialize the template-based text explainer.

        Args:
            language (str): Language for explanations ('fr' or 'en'). Default: 'fr'.
            feature_name_mapping (Optional[Dict[str, str]]): Mapping from technical
                feature names to human-readable names. Example:
                {'NumOfProducts': 'nombre de produits', 'Age': 'âge'}

        Raises:
            ValueError: If language is invalid.
        """
        super().__init__(language)
        self.feature_name_mapping = feature_name_mapping or {}

    def generate_explanation(
        self,
        prediction: float,
        contributions: Dict[str, float],
        values: Dict[str, Any],
        top_n: int = 3,
        target_name: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """Generate a natural language explanation using templates.

        Args:
            prediction (float): The model's prediction (probability).
            contributions (Dict[str, float]): Feature SHAP contributions.
            values (Dict[str, Any]): Feature values (after preprocessing).
            top_n (int): Number of top contributing features to include. Default: 3.
            target_name (Optional[str]): Human-readable name for the target variable.
            **kwargs (Any): Additional parameters (unused for template mode).

        Returns:
            str: Natural language explanation of the prediction.

        Raises:
            ValueError: If inputs are invalid.
        """
        # Set default target name
        if target_name is None:
            target_name = (
                "positive class" if self.language == "en" else "classe positive"
            )

        # Extract top features
        top_features = self._extract_top_features(contributions, values, top_n)

        # Format features list
        features_list = self._format_features_list(top_features)

        # Build explanation from template
        template = self.TEMPLATES[self.language]
        explanation = template.format(
            target_name=target_name, prediction=prediction, features_list=features_list
        )

        return explanation

    def _format_features_list(self, top_features: list) -> str:
        """Format the features list for the template.

        Args:
            top_features (list): List of feature dictionaries.

        Returns:
            str: Formatted features list with proper conjunction.
        """
        if not top_features:
            return ""

        feature_strings = []
        feature_template = self.FEATURE_TEMPLATES[self.language]

        for feature in top_features:
            name = feature["name"]
            value = feature["value"]
            contrib = feature["contribution"]
            impact = feature["impact"]

            # Apply feature name mapping if available
            display_name = self.feature_name_mapping.get(name, name)

            # Format feature string
            template = feature_template[impact]
            feature_str = template.format(
                feature_name=display_name, value=value, contrib=abs(contrib)
            )
            feature_strings.append(feature_str)

        # Join with proper conjunction
        if len(feature_strings) == 1:
            return feature_strings[0]
        elif len(feature_strings) == 2:
            conjunction = " et " if self.language == "fr" else " and "
            return f"{feature_strings[0]}{conjunction}{feature_strings[1]}"
        else:
            conjunction = " et " if self.language == "fr" else " and "
            return (
                ", ".join(feature_strings[:-1])
                + f"{conjunction}{feature_strings[-1]}"
            )
