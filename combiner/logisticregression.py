import joblib
from sklearn.linear_model import LogisticRegression

import ipal_iids.settings as settings
from combiner.combiner import Combiner


class LogisticRegressionCombiner(Combiner):
    _name = "LogisticRegression"
    _description = "Learns a logistic regression combiner."
    _requires_training = True
    _logistic_default_settings = {
        "keys": None,
        "use_scores": False,
    }

    def __init__(self):
        super().__init__()
        self._add_default_settings(self._logistic_default_settings)

        self.model = None

    def train(self, file):
        events, annotations = self._load_training(file)

        settings.logger.info("Fitting Logistic Regression Combiner")
        self.model = LogisticRegression()
        self.model.fit(events, annotations)

    def combine(self, alerts, scores):
        activations = self._get_activations(alerts, scores)
        alert = bool(self.model.predict([activations])[0])
        return alert, 1 if alert else 0, 0

    def save_trained_model(self):
        if self.settings["model-file"] is None:
            return False

        model = {
            "_name": self._name,
            "settings": self.settings,
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
                f"Model file {str(self._resolve_model_file_path())} not found."
            )
            return False

        # Load model
        assert self._name == model["_name"]
        self.settings = model["settings"]
        self.model = model["model"]

        return True
