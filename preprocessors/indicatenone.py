import ipal_iids.settings as settings
from .preprocessor import Preprocessor


class IndicateNonePreprocessor(Preprocessor):

    _name = "indicate-none"
    _description = "Set None to 0 and indicate with new feature"

    def __init__(self, features):
        super().__init__(features)

    def fit(self, values):
        pass

    def transform(self, value):
        if len(value) != len(self.features):
            settings.logger.critical("Feature length does not match data length!")

        for i in range(len(self.features)):
            if not self.features[i]:
                continue

            if value[i] is None:
                value[i] = [0, 1]
            else:
                value[i] = [value[i], 0]

        return value

    def reset(self):
        pass  # Nothing to reset

    def get_fitted_model(self):
        return {"features": self.features}

    @classmethod
    def from_fitted_model(cls, model):
        return IndicateNonePreprocessor(model["features"])
