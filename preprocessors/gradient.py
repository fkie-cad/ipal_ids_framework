from .preprocessor import Preprocessor


class GradientPreprocessor(Preprocessor):

    _name = "gradient"
    _description = "Calculate gradient"
    # NOTE Does not consider time!

    def __init__(self, features, window_size=10):  # NOTE remove window size?
        super().__init__(features)
        self.window_size = window_size

        self.last_value = [None] * len(self.features)
        self.sliding_window = [[] for _ in range(len(self.features))]

    def fit(self, values):
        pass

    def transform(self, value):

        for i in range(len(self.features)):
            if self.features[i]:
                out = None  # None if buffer not full yet

                if self.last_value[i] is not None:  # Skip if first value

                    # calculate new gradient
                    self.sliding_window[i].append(value[i] - self.last_value[i])

                    if len(self.sliding_window[i]) > self.window_size:
                        self.sliding_window[i].pop(0)
                    if len(self.sliding_window[i]) == self.window_size:
                        out = sum(self.sliding_window[i]) / self.window_size

                self.last_value[i] = value[i]  # Save last value
                value[i] = out  # set output

        return value

    def reset(self):
        self.sliding_window = [[] for _ in range(len(self.features))]
        self.last_value = [None] * len(self.features)

    def get_fitted_model(self):
        return {"features": self.features, "window_size": self.window_size}

    @classmethod
    def from_fitted_model(cls, model):
        gradient = GradientPreprocessor(model["features"], model["window_size"])
        return gradient
