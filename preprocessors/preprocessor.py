from typing import List


class Preprocessor:
    _name = None
    _description = ""
    _default_settings = {}

    features: List[bool]

    # Features is a list of booleans. True indicates preprocessing on this feature.
    def __init__(self, features: List[bool], names: None):
        self.features = features
        self.names = names

    def _lookup_feature_name(self, i):
        if self.names and i < len(self.names):
            return self.names[i]
        else:
            return "unknown"

    def fit(self, values):
        raise NotImplementedError

    def transform(self, values):
        raise NotImplementedError

    def get_fitted_model(self):
        raise NotImplementedError

    def reset(self):
        pass

    @classmethod
    def from_fitted_model(cls, model):
        raise NotImplementedError
