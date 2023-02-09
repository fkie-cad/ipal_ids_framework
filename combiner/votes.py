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


class MajorityCombiner(Combiner):
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
        alerts = list(alerts.values())
        return (
            alerts.count(True) >= len(alerts) / 2,
            alerts.count(True) / len(alerts) * 2,
        )

    def save_trained_model(self):
        settings.logger.info("Model of combiner does not need to be saved.")
        return False

    def load_trained_model(self):
        settings.logger.info("Model of combiner does not need to be loaded.")
        return False


class WeightsCombiner(Combiner):
    _name = "Weights"
    _description = "Each IDS gets assigned a dedicated weight. The combiner alerts if a weighted sum of alerts/scores is greater than a threshold."
    _requires_training = False
    _weights_default_settings = {
        "weights": {},  # dict of IDS name and weight
        "threshold": 1,  # threshold upon which to emit an alert
        "use_scores": False,
    }

    def __init__(self):
        super().__init__()
        self._add_default_settings(self._weights_default_settings)

    def train(self, file):
        pass

    def combine(self, alerts, scores):
        data = scores if self.settings["use_scores"] else alerts

        if not set(data.keys()) == set(self.settings["weights"].keys()):
            settings.logger.error("Keys of weights combiner do not match data")
            settings.logger.error("- data keys: {}".format(",".join(data.keys())))
            settings.logger.error(
                "- weights keys: {}".format(",".join(self.settings["weights"].keys()))
            )
            exit(1)

        # sum of weights for alarms (True * num = num, False * num = 0)
        sums = sum([data[name] * self.settings["weights"][name] for name in data])
        return sums >= self.settings["threshold"], sums / self.settings["threshold"]

    def save_trained_model(self):
        settings.logger.info("Model of combiner does not need to be saved.")
        return False

    def load_trained_model(self):
        settings.logger.info("Model of combiner does not need to be loaded.")
        return False
