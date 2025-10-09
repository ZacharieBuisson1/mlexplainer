"""Base class for text explanation generation."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseTextExplainer(ABC):
    """Abstract base class for generating natural language explanations.

    This class defines the interface for text explanation generators,
    supporting both LLM-based and template-based approaches.

    Attributes:
        language (str): Language for explanations ('fr' or 'en').
    """

    def __init__(self, language: str = "fr"):
        """Initialize the text explainer.

        Args:
            language (str): Language for explanations ('fr' or 'en'). Default: 'fr'.

        Raises:
            ValueError: If language is not 'fr' or 'en'.
        """
        if language not in ["fr", "en"]:
            raise ValueError(f"Language must be 'fr' or 'en', got: {language}")
        self.language = language

    @abstractmethod
    def generate_explanation(
        self,
        prediction: float,
        contributions: Dict[str, float],
        values: Dict[str, Any],
        top_n: int = 3,
        target_name: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """Generate a natural language explanation for a prediction.

        Args:
            prediction (float): The model's prediction (probability).
            contributions (Dict[str, float]): Feature SHAP contributions.
            values (Dict[str, Any]): Feature values (after preprocessing).
            top_n (int): Number of top contributing features to include. Default: 3.
            target_name (Optional[str]): Human-readable name for the target variable.
            **kwargs (Any): Additional parameters for specific implementations.

        Returns:
            str: Natural language explanation of the prediction.

        Raises:
            ValueError: If inputs are invalid.
        """

    def _rank_features_by_contribution(
        self, contributions: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Rank features by absolute contribution magnitude.

        Args:
            contributions (Dict[str, float]): Feature SHAP contributions.

        Returns:
            List[Dict[str, Any]]: Sorted list of feature info dictionaries with keys:
                - 'name' (str): Feature name
                - 'contribution' (float): SHAP contribution value
                - 'abs_contribution' (float): Absolute contribution value
                - 'impact' (str): 'positive' or 'negative'
        """
        feature_info = []
        for name, contrib in contributions.items():
            feature_info.append(
                {
                    "name": name,
                    "contribution": contrib,
                    "abs_contribution": abs(contrib),
                    "impact": "positive" if contrib > 0 else "negative",
                }
            )

        # Sort by absolute contribution (descending)
        return sorted(
            feature_info, key=lambda x: x["abs_contribution"], reverse=True
        )

    def _extract_top_features(
        self, contributions: Dict[str, float], values: Dict[str, Any], top_n: int = 3
    ) -> List[Dict[str, Any]]:
        """Extract top N features with their values and contributions.

        Args:
            contributions (Dict[str, float]): Feature SHAP contributions.
            values (Dict[str, Any]): Feature values.
            top_n (int): Number of top features to extract.

        Returns:
            List[Dict[str, Any]]: Top N features with keys:
                - 'name' (str): Feature name
                - 'value' (Any): Feature value
                - 'contribution' (float): SHAP contribution
                - 'impact' (str): 'positive' or 'negative'
        """
        ranked_features = self._rank_features_by_contribution(contributions)

        top_features = []
        for feature_info in ranked_features[:top_n]:
            name = feature_info["name"]
            top_features.append(
                {
                    "name": name,
                    "value": values.get(name, None),
                    "contribution": feature_info["contribution"],
                    "impact": feature_info["impact"],
                }
            )

        return top_features
