import json
import joblib

from sklearn.linear_model import LogisticRegression as LogisticRegressionModel

import ipal_iids.settings as settings
from .combiner import Combiner


class LogisticRegressionCombiner(Combiner):

    _name = "LogisticRegression"
    _description = "Learns a logistic regression combiner."
    _requires_training = True
    _logistic_default_settings = {"use_scores": False}

    def __init__(self):
        super().__init__()
        self._add_default_settings(self._logistic_default_settings)

        self.model = None
        self.keys = None

    def _get_activations(self, alerts, scores):
        data = scores if self.settings["use_scores"] else alerts

        if not set(data.keys()) == set(self.keys):
            settings.logger.error("Keys of combiner do not match data")
            settings.logger.error("- data keys: {}".format(",".join(data.keys())))
            settings.logger.error("- combiner keys: {}".format(",".join(self.keys)))
            exit(1)

        return [float(data[ids]) for ids in self.keys]

    def train(self, file):
        self.model = LogisticRegressionModel()
        events = []
        annotations = []

        settings.logger.info("Loading combiner training file")
        with self._open_file(file, "r") as f:
            for line in f.readlines():
                js = json.loads(line)

                if self.keys is None:
                    self.keys = sorted(js["scores"].keys())

                events.append(self._get_activations(js["alerts"], js["scores"]))
                annotations.append(js["malicious"] is not False)

        settings.logger.info("Fitting LogisticRegression Combiner")
        self.model.fit(events, annotations)

    def combine(self, alerts, scores):
        activations = self._get_activations(alerts, scores)
        alert = bool(self.model.predict([activations])[0])
        return alert, 1 if alert else 0

    def save_trained_model(self):

        if self.settings["model-file"] is None:
            return False

        model = {
            "_name": self._name,
            "settings": self.settings,
            "keys": self.keys,
            "model": self.model,
        }

        joblib.dump(model, self._resolve_model_file_path(), compress=3)

        return True

    def load_trained_model(self):
        if self.settings["model-file"] is None:
            return False

        try:  # Open model file
            model = joblib.load(self._resolve_model_file_path())
        except FileNotFoundError:
            settings.logger.info(
                "Model file {} not found.".format(str(self._resolve_model_file_path()))
            )
            return False

        # Load model
        assert self._name == model["_name"]
        self.settings = model["settings"]
        self.keys = model["keys"]
        self.model = model["model"]

        return True
