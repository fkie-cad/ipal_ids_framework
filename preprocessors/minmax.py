import numpy as np

from typing import List

import ipal_iids.settings as settings

from .preprocessor import Preprocessor


class MinMaxPreprocessor(Preprocessor):
    _name = "minmax"
    _description = "Scale by mininum and maximum"
    mins: List[float]
    maxs: List[float]

    def __init__(self, features):
        super().__init__(features)
        self.mins = [None] * len(self.features)
        self.maxs = [None] * len(self.features)

    def fit(self, values):
        if len(values[0]) != len(self.features):
            settings.logger.critical("Feature length does not match data length!")

        for i in range(len(self.features)):
            if not self.features[i]:
                continue

            X = [v[i] for v in values if v[i] is not None]
            self.mins[i] = float(np.min(X))
            self.maxs[i] = float(np.max(X))

            if self.mins[i] == self.maxs[i]:
                settings.logger.info(
                    "Min Max is the same. Adjusting values of feature {} to 0.5".format(
                        i
                    )
                )
                self.mins[i] -= 1
                self.maxs[i] += 1

            assert self.maxs[i] - self.mins[i] > 0

    def transform(self, value):
        if len(value) != len(self.features):
            settings.logger.critical("Feature length does not match data length!")

        for i in range(len(self.features)):
            if not self.features[i] or value[i] is None:
                continue

            if value[i] < self.mins[i] or self.maxs[i] < value[i]:
                settings.logger.warning(
                    "Value {} out of trained range ({} - {})".format(
                        value[i], self.mins[i], self.maxs[i]
                    )
                )

            value[i] = (value[i] - self.mins[i]) / (self.maxs[i] - self.mins[i])

        return value

    def reset(self):
        pass  # Nothing to reset

    def get_fitted_model(self):
        return {"features": self.features, "mins": self.mins, "maxs": self.maxs}

    @classmethod
    def from_fitted_model(cls, model):
        minmax = MinMaxPreprocessor(model["features"])
        minmax.mins = model["mins"]
        minmax.maxs = model["maxs"]
        return minmax
