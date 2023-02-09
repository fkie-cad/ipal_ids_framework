import joblib
import os
import random

from sklearn.decomposition import PCA

import ipal_iids.settings as settings

from .preprocessor import Preprocessor


class PCAPreprocessor(Preprocessor):
    _name = "pca"
    _description = "Performs a principal component analysis"

    def __init__(self, features):
        super().__init__(features)
        self.encoder = None

    def fit(self, values):
        if len(values[0]) != len(self.features):
            settings.logger.critical("Feature length does not match data length!")

        self.encoder = PCA()
        self.encoder.fit(values)

    def transform(self, value):
        if len(value) != len(self.features):
            settings.logger.critical("Feature length does not match data length!")

        return self.encoder.transform([value])[0]

    def reset(self):
        pass  # Nothing to reset

    def get_fitted_model(self):
        # Save model to temprary file
        tmp = ".tmp-{}".format(random.randint(1000, 9999))
        joblib.dump(self.encoder, tmp, compress=3)

        # Load file to string
        with open(tmp, "rb") as f:
            model = list(f.read())

        # Remove temporary file
        os.remove(tmp)

        return {"features": self.features, "model": model}

    @classmethod
    def from_fitted_model(cls, model):
        # Save model to temporary file
        tmp = ".tmp-{}".format(random.randint(1000, 9999))
        with open(tmp, "wb") as f:
            f.write(bytes(model["model"]))

        # Load model
        pca = PCAPreprocessor(model["features"])
        pca.encoder = joblib.load(tmp)

        # Remove temprary file
        os.remove(tmp)

        return pca
