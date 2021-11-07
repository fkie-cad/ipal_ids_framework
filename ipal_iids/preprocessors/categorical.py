import numpy as np

from typing import Any, Dict, Union, List

import ipal_iids.settings as settings

from .preprocessor import Preprocessor


class CategoricalPreprocessor(Preprocessor):

    _name = "categorical"
    _description = "Encode as categorical"
    encoder: List[Union[Dict[str, Any], None]]

    def __init__(self, features):
        super().__init__(features)

        self.encoder = [None] * len(self.features)

    def fit(self, values):
        if len(values[0]) != len(self.features):
            settings.logger.critical("Feature length does not match data length!")

        for i in range(len(self.features)):
            if not self.features[i]:
                continue

            X = list(set([v[i] for v in values]))
            classes = np.eye(len(X))
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
                    "Value {} not in trained categories".format(value[i])
                )

        return value

    def get_fitted_model(self):
        return {"features": self.features, "encoder": self.encoder}

    @classmethod
    def from_fitted_model(cls, model):
        categorical = CategoricalPreprocessor(model["features"])
        categorical.encoder = model["encoder"]
        return categorical
