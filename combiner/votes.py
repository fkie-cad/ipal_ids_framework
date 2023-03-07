import ipal_iids.settings as settings

from .combiner import Combiner, RunningAverageCombiner


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
        return any(alerts), alerts.count(True)

    def save_trained_model(self):
        settings.logger.info("Model of combiner does not need to be saved.")
        return False

    def load_trained_model(self):
        settings.logger.info("Model of combiner does not need to be loaded.")
        return False


class AllCombiner(Combiner):
    _name = "All"
    _description = "Alerts if all IDSs emit an alert."
    _requires_training = False
    _all_default_settings = {}

    def __init__(self):
        super().__init__()
        self._add_default_settings(self._all_default_settings)

    def train(self, file):
        pass

    def combine(self, alerts, scores):
        alerts = list(alerts.values())
        return all(alerts), alerts.count(True) / len(alerts)

    def save_trained_model(self):
        settings.logger.info("Model of combiner does not need to be saved.")
        return False

    def load_trained_model(self):
        settings.logger.info("Model of combiner does not need to be loaded.")
        return False


class MajorityCombiner(RunningAverageCombiner):
    _name = "Majority"
    _description = "Alerts if the majority of IDSs emit an alert."
    _requires_training = False
    _majority_default_settings = {}

    def __init__(self):
        super().__init__()
        self._add_default_settings(self._majority_default_settings)

    def train(self, file):
        pass

    def combine(self, alerts, scores):
        votes = self._get_activations(alerts, scores)
        threshold = 0.5 * len(votes)
        return sum(votes) >= threshold, sum(votes) / threshold

    def save_trained_model(self):
        settings.logger.info("Model of combiner does not need to be saved.")
        return False

    def load_trained_model(self):
        settings.logger.info("Model of combiner does not need to be loaded.")
        return False


class WeightsCombiner(RunningAverageCombiner):
    _name = "Weights"
    _description = "Each IDS gets assigned a dedicated weight. The combiner alerts if a weighted sum of alerts/scores is greater than a threshold."
    _requires_training = False
    _weights_default_settings = {
        "weights": [],  # list of weights for IDSs (order of keys)
        "threshold": 1,  # threshold upon which to emit an alert
    }

    def __init__(self):
        super().__init__()
        self._add_default_settings(self._weights_default_settings)

    def train(self, file):
        pass

    def combine(self, alerts, scores):
        votes = self._get_activations(alerts, scores)
        sums = sum([v * w for v, w in zip(votes, self.settings["weights"])])
        return sums >= self.settings["threshold"], sums / self.settings["threshold"]

    def save_trained_model(self):
        settings.logger.info("Model of combiner does not need to be saved.")
        return False

    def load_trained_model(self):
        settings.logger.info("Model of combiner does not need to be loaded.")
        return False
