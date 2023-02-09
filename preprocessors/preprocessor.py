from typing import List


class Preprocessor:
    _name = None
    _description = ""
    _default_settings = {}

    features: List[bool]

    # Features is a list of booleans. True indicates preprocessing on this feature.
    def __init__(self, features: List[bool]):
        self.features = features

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
