import ipal_iids.settings as settings

from .combiner import Combiner


class MatrixCombiner(Combiner):
    _name = "Matrix"
    _description = "Similar to a weighted combiner but with a timeframe."
    _requires_training = False
    _matrix_default_settings = {
        "matrix": [],  # matrix of weights for each IDS
        "threshold": 0,  # Threshold after which to emit an alert
        "use_scores": False,  # Whether to use alerts or confidence scores
        "keys": [],  # name of the IIDSs determining the order of the weihts
        "lookahead": 0,  # Steps to look into the "future" (requires ipal-extend-alarms)
    }

    def __init__(self):
        super().__init__()
        self._add_default_settings(self._matrix_default_settings)

        assert len(self.settings["keys"]) == len(self.settings["matrix"])
        assert self.settings["lookahead"] >= 0

        self._latest_alerts = []
        for i, ids in enumerate(self.settings["keys"]):
            assert len(self.settings["matrix"][i]) > self.settings["lookahead"]
            self._latest_alerts.append([0] * len(self.settings["matrix"][i]))

    def _update_latest_alerts(self, alerts, scores):
        data = scores if self.settings["use_scores"] else alerts
        score = 0

        # Validate data
        if not set(data.keys()) == set(self.settings["keys"]):
            settings.logger.error("Keys of combiner do not match data")

        for i, ids in enumerate(self.settings["keys"]):
            # Update values
            self._latest_alerts[i].pop(0)
            self._latest_alerts[i].append(data[ids])

            # Pointwise multiplication and sum of weights and alert/score
            val = sum(
                [
                    a * b
                    for a, b in zip(self._latest_alerts[i], self.settings["matrix"][i])
                ]
            )
            score += max(-1, min(1, val))  # clamp between -1 and 1

        return score

    def train(self, file):
        pass

    def combine(self, alerts, scores):
        score = self._update_latest_alerts(alerts, scores)

        return (
            score >= self.settings["threshold"],
            score / self.settings["threshold"]
            if self.settings["threshold"] != 0
            else 0,
            -self.settings["lookahead"],
        )

    def save_trained_model(self):
        settings.logger.info("Model of combiner does not need to be saved.")
        return False

    def load_trained_model(self):
        settings.logger.info("Model of combiner does not need to be loaded.")
        return False
