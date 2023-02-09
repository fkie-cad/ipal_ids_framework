from typing import List, Optional
from sklearn.preprocessing import LabelEncoder

import ipal_iids.settings as settings

from .preprocessor import Preprocessor


class LabelEncoderPreprocessor(Preprocessor):
    _name = "label"
    _description = "Encode as labels"
    encoder: List[Optional[LabelEncoder]]
    fitdata: List[Optional[List[set]]]

    def __init__(self, features):
        super().__init__(features)
        self.encoder = [None] * len(self.features)
        self.fitdata = [None] * len(self.features)

    def fit(self, values):
        if len(values[0]) != len(self.features):
            settings.logger.critical("Feature length does not match data length!")

        for i in range(len(self.features)):
            if not self.features[i]:
                continue

            self.encoder[i] = LabelEncoder()
            self.fitdata[i] = list(set([v[i] for v in values]))
            self.encoder[i].fit(self.fitdata[i])

    def transform(self, value):
        if len(value) != len(self.features):
            settings.logger.critical("Feature length does not match data length!")

        for i in range(len(self.features)):
            if not self.features[i] or value[i] is None:
                continue

            try:
                value[i] = float(self.encoder[i].transform([value[i]])[0])
            except ValueError:
                settings.logger.critical(
                    "Value {} not in trained categories".format(value[i])
                )

        return value

    def reset(self):
        pass  # Nothing to reset

    def get_fitted_model(self):
        return {"features": self.features, "fitdata": self.fitdata}

    @classmethod
    def from_fitted_model(cls, model):
        labelencoder = LabelEncoderPreprocessor(model["features"])
        labelencoder.fitdata = model["fitdata"]

        for i in range(len(labelencoder.features)):
            if not labelencoder.features[i]:
                continue

            labelencoder.encoder[i] = LabelEncoder()
            labelencoder.encoder[i].fit(labelencoder.fitdata[i])

        return labelencoder
