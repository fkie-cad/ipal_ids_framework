from typing import List

import numpy as np

import ipal_iids.settings as settings
from preprocessors.preprocessor import Preprocessor


class MeanPreprocessor(Preprocessor):
    _name = "Mean"
    _description = "Scale by mean-standard deviation"
    means: List[float]
    stds: List[float]

    def __init__(self, features):
        super().__init__(features)
        self.means = [None] * len(self.features)
        self.stds = [None] * len(self.features)

    def fit(self, values):
        if len(values[0]) != len(self.features):
            settings.logger.critical("Feature length does not match data length!")

        for i in range(len(self.features)):
            if not self.features[i]:
                continue

            X = [v[i] for v in values if v[i] is not None]
            self.means[i] = float(np.mean(X))
            self.stds[i] = float(np.std(X))

            if self.stds[i] == 0:
                settings.logger.info(
                    f"Standard deviation is zero. Adjusting values of {i} to std 1.0"
                )
                self.stds[i] = 1

    def transform(self, value):
        if len(value) != len(self.features):
            settings.logger.critical("Feature length does not match data length!")

        for i in range(len(self.features)):
            if not self.features[i] or value[i] is None:
                continue

            value[i] = (value[i] - self.means[i]) / self.stds[i]

        return value

    def reset(self):
        pass  # Nothing to reset

    def get_fitted_model(self):
        return {"features": self.features, "means": self.means, "stds": self.stds}

    @classmethod
    def from_fitted_model(cls, model):
        mean = MeanPreprocessor(model["features"])
        mean.means = model["means"]
        mean.stds = model["stds"]
        return mean
