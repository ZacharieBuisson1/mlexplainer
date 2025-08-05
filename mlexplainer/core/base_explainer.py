from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional

from pandas import DataFrame, Series


class BaseMLExplainer(ABC):
    """Base class for Machine Learning Explainers."""

    def __init__(
        self,
        x_train: DataFrame,
        y_train: Series,
        features: List[str],
        model: Callable,
        global_explainer: bool = True,
        local_explainer: bool = True,
    ):
        """
        Initialize the BaseMLExplainer with training data, features, and model.
        Args:
            x_train (DataFrame): Training feature values.
            y_train (Series): Training target values.
            features (List[str]): List of feature names to interpret.
            model (Callable): The machine learning model to explain.
        Raises:
            ValueError: If x_train or y_train is None, or if features are not provided
                        or not present in x_train.
            ValueError: If any feature in features is not present in x_train.
            ValueError: If no features are provided.
        """

        self.x_train = x_train
        self.y_train = y_train
        self.features = features
        self.model = model
        self.global_explainer = global_explainer
        self.local_explainer = local_explainer

        # split features into categorical, numerical and string
        self.categorical_features: List[str] = [
            col for col in x_train.columns if x_train[col].dtype == "category"
        ]
        self.numerical_features: List[str] = [
            col
            for col in x_train.columns
            if x_train[col].dtype in [int, float]
        ]
        self.string_features: List[str] = [
            col for col in x_train.columns if x_train[col].dtype == "object"
        ]

        if self.x_train is None or self.y_train is None:
            raise ValueError("X_train and y_train must be provided.")

        if not self.features:
            raise ValueError("At least one feature must be provided.")

        if not all(
            [feature in self.x_train.columns for feature in self.features]
        ):
            raise ValueError(
                "All features must be present in x_train. Missing features: "
                f"{set(self.features) - set(self.x_train.columns)}"
            )

    @abstractmethod
    def explain(self, **kwargs: Any) -> None:
        """Interpret features for the machine learning model.
        This method should be implemented in subclasses to provide specific interpretations.
        """
        pass

    @abstractmethod
    def correctness_features(
        self,
        q: Optional[int] = None,
    ) -> dict:
        """Analyze the correctness of the analysis for every feature.

        This method validates interpretation consistency between actual target rates
        and SHAP values for all features in the explainer.

        Args:
            q (Optional[int]): Number of quantiles for continuous features. If None, uses adaptive quantiles.

        Returns:
            dict: Dictionary with feature names as keys and correctness results as values.
        """
        pass