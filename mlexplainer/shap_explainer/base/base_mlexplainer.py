from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional

from pandas import DataFrame, Series


class BaseMLExplainer(ABC):

    def __init__(
        self,
        x_train: DataFrame,
        y_train: Series,
        features: List[str],
        model: Callable,
    ):

        self.x_train = x_train
        self.y_train = y_train
        self.features = features
        self.model = model

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

    def __post_init__(self):
        """Post-initialization method to be overridden by subclasses."""
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
    def interpret_features(self, **kwargs: Any) -> None:
        pass
