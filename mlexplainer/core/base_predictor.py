"""Base class for Machine Learning Predictors with SHAP contributions.

This class is designed to be subclassed for specific prediction tasks,
providing SHAP-based feature contributions for individual observations.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

from pandas import DataFrame
from sklearn.pipeline import Pipeline


class BaseMLPredictor(ABC):
    """Base class for Machine Learning Predictors with SHAP contributions.

    This class provides a structure for making predictions on individual observations
    while calculating SHAP-based feature contributions. It automatically detects
    feature names and types from the model, and handles raw data preprocessing.

    Attributes:
        model (Callable): The machine learning model to use for predictions.
        x_train (DataFrame): Training feature values (required for TreeExplainer).
        pipeline (Optional[Pipeline]): Optional scikit-learn pipeline for preprocessing.
        features (List[str]): Auto-detected list of feature names from model.
        categorical_features (List[str]): Auto-detected categorical feature names.
        numerical_features (List[str]): Auto-detected numerical feature names.
        string_features (List[str]): Auto-detected string feature names.
    """

    def __init__(
        self,
        model: Callable,
        x_train: DataFrame,
        pipeline: Optional[Pipeline] = None,
    ):
        """Initialize the BaseMLPredictor with model, training data, and optional pipeline.

        This class automatically detects:
        - Feature names from the model (feature_names_in_, get_booster().feature_names, etc.)
        - Feature types (categorical vs numerical) from model metadata
        - Required data transformations based on pipeline

        Args:
            model (Callable): The machine learning model to use for predictions.
                Supports: XGBoost, LightGBM, CatBoost, scikit-learn models.
            x_train (DataFrame): Training feature values (required for SHAP TreeExplainer).
                This should be the PROCESSED data (after pipeline transformation).
            pipeline (Optional[Pipeline]): Optional scikit-learn Pipeline for preprocessing
                raw input data. If provided, raw observations will be transformed before
                prediction and SHAP calculation.

        Raises:
            ValueError: If x_train is None.
            ValueError: If features cannot be extracted from model.
        """
        self.model = model
        self.x_train = x_train
        self.pipeline = pipeline

        # Validation
        if self.x_train is None:
            raise ValueError("x_train must be provided.")

        # Auto-detect features from model
        self.features = self._extract_feature_names()

        # Auto-detect categorical features from model
        self.categorical_features = self._extract_categorical_features()

        # Split remaining features into numerical and string (based on x_train)
        self.numerical_features: List[str] = [
            col
            for col in self.features
            if col not in self.categorical_features
            and col in self.x_train.columns
            and self.x_train[col].dtype in [int, float]
        ]
        self.string_features: List[str] = [
            col
            for col in self.features
            if col not in self.categorical_features
            and col in self.x_train.columns
            and self.x_train[col].dtype == "object"
        ]

    def _extract_feature_names(self) -> List[str]:
        """Extract feature names from the model.

        Supports multiple model types:
        - XGBoost: get_booster().feature_names or feature_names_in_
        - LightGBM: feature_name() or feature_names_in_
        - CatBoost: feature_names_
        - Scikit-learn: feature_names_in_

        Returns:
            List[str]: List of feature names used by the model.

        Raises:
            ValueError: If feature names cannot be extracted from model.
        """
        # Try common attributes for feature names
        if hasattr(self.model, "feature_names_in_"):
            return list(self.model.feature_names_in_)

        # XGBoost specific
        if hasattr(self.model, "get_booster"):
            try:
                booster = self.model.get_booster()
                if hasattr(booster, "feature_names"):
                    return list(booster.feature_names)
            except Exception:
                pass

        # LightGBM specific
        if hasattr(self.model, "feature_name"):
            try:
                return list(self.model.feature_name())
            except Exception:
                pass

        # CatBoost specific
        if hasattr(self.model, "feature_names_"):
            return list(self.model.feature_names_)

        # Fallback: use x_train columns
        if self.x_train is not None:
            return list(self.x_train.columns)

        raise ValueError(
            "Could not extract feature names from model. "
            "Model must have one of: feature_names_in_, get_booster().feature_names, "
            "feature_name(), or feature_names_"
        )

    def _extract_categorical_features(self) -> List[str]:
        """Extract categorical feature names from the model metadata.

        Detects categorical features from:
        - XGBoost: feature_types (check for 'c' type)
        - LightGBM: categorical_feature attribute
        - CatBoost: cat_features_
        - x_train: dtype == 'category'

        Returns:
            List[str]: List of categorical feature names.
        """
        categorical_features = []

        # XGBoost: check feature_types
        if hasattr(self.model, "get_booster"):
            try:
                booster = self.model.get_booster()
                if hasattr(booster, "feature_types") and booster.feature_types:
                    categorical_features = [
                        self.features[i]
                        for i, ftype in enumerate(booster.feature_types)
                        if ftype == "c"
                    ]
                    return categorical_features
            except Exception:
                pass

        # LightGBM: categorical_feature
        if hasattr(self.model, "categorical_feature"):
            try:
                cat_features = self.model.categorical_feature
                if cat_features and cat_features != "auto":
                    # Can be indices or names
                    if isinstance(cat_features[0], int):
                        categorical_features = [
                            self.features[i] for i in cat_features
                        ]
                    else:
                        categorical_features = list(cat_features)
                    return categorical_features
            except Exception:
                pass

        # CatBoost: cat_features_
        if hasattr(self.model, "cat_features_"):
            try:
                categorical_features = list(self.model.cat_features_)
                return categorical_features
            except Exception:
                pass

        # Fallback: detect from x_train dtype
        if self.x_train is not None:
            categorical_features = [
                col
                for col in self.features
                if col in self.x_train.columns
                and self.x_train[col].dtype == "category"
            ]

        return categorical_features

    @abstractmethod
    def predict_with_contributions(
        self, observation: Union[DataFrame, Dict[str, Any]], **kwargs: Any
    ) -> Dict[str, Any]:
        """Make a prediction and calculate SHAP contributions for an observation.

        This method should be implemented in subclasses to provide specific prediction
        logic and contribution calculation for different task types (binary, multilabel, etc.).

        Args:
            observation (Union[DataFrame, Dict[str, Any]]): A single observation to predict on.
                Can be:
                - DataFrame with 1 row (raw or processed depending on pipeline)
                - Dict with feature names as keys (will be converted to DataFrame)
                - JSON-like dict from API request
            **kwargs (Any): Additional keyword arguments for customization.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'prediction': The model's prediction (float or array)
                - 'base_value': The expected value from SHAP (baseline)
                - 'contributions': Dict mapping feature names to their SHAP contributions
                - 'contributions_pct': Dict mapping feature names to their relative contributions (%)

        Raises:
            ValueError: If observation format is invalid.
        """

    def _prepare_observation(
        self, observation: Union[DataFrame, Dict[str, Any]]
    ) -> tuple[DataFrame, Dict[str, Any], Dict[str, Any]]:
        """Prepare and validate observation data for prediction.

        This method:
        1. Converts dict/JSON to DataFrame if needed
        2. Applies pipeline transformation if pipeline is provided
        3. Converts categorical features to proper dtype based on model metadata
        4. Validates the final processed observation

        Args:
            observation (Union[DataFrame, Dict[str, Any]]): Raw observation data.

        Returns:
            tuple: A tuple containing:
                - DataFrame: Processed observation ready for prediction (1 row)
                - Dict[str, Any]: Values before processing (all input columns)
                - Dict[str, Any]: Values after processing (only model features used for prediction)

        Raises:
            ValueError: If observation format is invalid or transformation fails.
        """
        # Convert dict to DataFrame
        if isinstance(observation, dict):
            observation = DataFrame([observation])
        elif not isinstance(observation, DataFrame):
            raise ValueError(
                "observation must be a pandas DataFrame or a dict. "
                f"Got {type(observation)}"
            )

        # Validate single row
        if len(observation) != 1:
            raise ValueError(
                f"observation must contain exactly 1 row, got {len(observation)} rows."
            )

        # Store values before processing - ALL input columns (convert to regular dict with Python native types)
        # Ensure keys are Python strings (not numpy.str_)
        values_before = {
            str(col): observation[col].iloc[0].item()
            if hasattr(observation[col].iloc[0], 'item')
            else observation[col].iloc[0]
            for col in observation.columns
        }

        # Apply pipeline transformation if provided
        if self.pipeline is not None:
            try:
                observation_processed = self.pipeline.transform(observation)

                # If pipeline returns numpy array, convert back to DataFrame
                if not isinstance(observation_processed, DataFrame):
                    observation_processed = DataFrame(
                        observation_processed, columns=self.features
                    )
            except Exception as e:
                raise ValueError(
                    f"Pipeline transformation failed: {str(e)}"
                ) from e
        else:
            observation_processed = observation.copy()

        # Convert categorical features to proper dtype
        observation_processed = self._convert_categorical_features(
            observation_processed
        )

        # Validate features in processed observation
        self._validate_observation(observation_processed)

        # Store values after processing - ONLY model features (convert to regular dict with Python native types)
        # This filters out any intermediate columns created during preprocessing
        # Ensure keys are Python strings (not numpy.str_)
        values_after = {
            str(col): observation_processed[col].iloc[0].item()
            if hasattr(observation_processed[col].iloc[0], 'item')
            else observation_processed[col].iloc[0]
            for col in self.features
            if col in observation_processed.columns
        }

        return observation_processed, values_before, values_after

    def _convert_categorical_features(self, observation: DataFrame) -> DataFrame:
        """Convert categorical features to proper dtype based on model metadata.

        Args:
            observation (DataFrame): The observation DataFrame.

        Returns:
            DataFrame: Observation with categorical features properly typed.
        """
        observation = observation.copy()

        for feature in self.categorical_features:
            if feature in observation.columns:
                # Convert to category dtype if not already
                if observation[feature].dtype != "category":
                    # Get categories from x_train if available
                    if (
                        feature in self.x_train.columns
                        and self.x_train[feature].dtype == "category"
                    ):
                        categories = self.x_train[feature].cat.categories
                        observation[feature] = observation[feature].astype(
                            "category"
                        )
                        observation[feature] = observation[feature].cat.set_categories(
                            categories
                        )
                    else:
                        observation[feature] = observation[feature].astype(
                            "category"
                        )

        return observation

    def _validate_observation(self, observation: DataFrame) -> None:
        """Validate that the observation has required features after processing.

        Args:
            observation (DataFrame): The processed observation to validate.

        Raises:
            ValueError: If observation is missing required features.
        """
        missing_features = set(self.features) - set(observation.columns)
        if missing_features:
            raise ValueError(
                f"Processed observation is missing required features: {missing_features}"
            )
