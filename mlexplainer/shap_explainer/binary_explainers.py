from typing import Callable, List

from pandas import DataFrame, Series

from mlexplainer.shap_explainer.base import BaseMLExplainer
from mlexplainer.utils_shap_explainer.shap_wrapper import 


class BinaryMLExplainer(BaseMLExplainer):

    def __init__(
        self,
        x_train: DataFrame,
        y_train: Series,
        features: List[str],
        model: Callable,
    ):

        super().__init__(x_train, y_train, features, model)

        shap_values_train = 

    def interpret_features(self, **kwargs):
        """Interpret features for binary classification.

        This method should be implemented to provide feature interpretation
        specific to binary classification tasks.
        """
        raise NotImplementedError(
            "This method should be implemented in subclasses."
        )
