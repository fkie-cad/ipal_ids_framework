import ipal_iids.settings as settings

from .combiner import Combiner


class AnyCombiner(Combiner):
    _name = "Any"
    _description = "Alerts if any IDS emits an alert."
    _requires_training = False
    _any_default_settings = {}

    def __init__(self):
        super().__init__()
        self._add_default_settings(self._any_default_settings)

    def train(self, file):
        pass

    def combine(self, alerts, scores):
        alerts = list(alerts.values())
        return any(alerts), alerts.count(True), 0

    def save_trained_model(self):
        settings.logger.info("Model of combiner does not need to be saved.")
        return False

    def load_trained_model(self):
        settings.logger.info("Model of combiner does not need to be loaded.")
        return False
