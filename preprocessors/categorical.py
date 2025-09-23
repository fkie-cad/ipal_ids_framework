from typing import Any, Dict, List, Union

import numpy as np

import ipal_iids.settings as settings
from preprocessors.preprocessor import Preprocessor


class CategoricalPreprocessor(Preprocessor):
    _name = "Categorical"
    _description = "Encode as categorical"
    encoder: List[Union[Dict[str, Any], None]]

    def __init__(self, features, names=None):
        super().__init__(features, names=names)

        self.encoder = [None] * len(self.features)

    def fit(self, values):
        if len(values[0]) != len(self.features):
            settings.logger.critical("Feature length does not match data length!")

        for i in range(len(self.features)):
            if not self.features[i]:
                continue

            X = list(set([v[i] for v in values]))
            classes = np.eye(len(X))
            settings.logger.info(
                f"Feature {self._lookup_feature_name(i)} has {len(X)} categories."
            )
            self.encoder[i] = {str(x): list(classes[X.index(x)]) for x in X}

    def transform(self, value):
        if len(value) != len(self.features):
            settings.logger.critical("Feature length does not match data length!")

        for i in range(len(self.features)):
            if not self.features[i]:
                continue

            try:
                value[i] = self.encoder[i][str(value[i])]
            except ValueError:
                settings.logger.critical(
                    f"Value {value[i]} of feature {self._lookup_feature_name(i)} has not been seen during category training"
                )

        return value

    def reset(self):
        pass  # Nothing to reset

    def get_fitted_model(self):
        return {"features": self.features, "encoder": self.encoder}

    @classmethod
    def from_fitted_model(cls, model):
        categorical = CategoricalPreprocessor(model["features"])
        categorical.encoder = model["encoder"]
        return categorical
