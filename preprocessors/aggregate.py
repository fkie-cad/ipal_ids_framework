import ipal_iids.settings as settings

from .preprocessor import Preprocessor


class AggregatePreprocessor(Preprocessor):

    _name = "aggregate"
    _description = "Aggregates multiple vectors into one feature"

    def __init__(self, features):
        super().__init__(features)

        self.num = 90
        self.N = 0
        self.aggregate = []

    def fit(self, values):
        pass

    def transform(self, value):

        self.aggregate += value
        self.N += 1

        if self.N == self.num:
            buf = self.aggregate
            self.aggregate = []
            self.N = 0
            return buf

        else:
            return None

    def get_fitted_model(self):
        return {"features": self.features, "num": self.num}

    @classmethod
    def from_fitted_model(cls, model):
        aggregate = AggregatePreprocessor(model["features"])
        aggregate.num = model["num"]
        return aggregate
