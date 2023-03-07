import gzip
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import ipal_iids.settings as settings


class Combiner:
    _name = None
    _description = ""
    _requires_training = True
    _metacombiner_default_settings = {"model-file": None}

    def __init__(self):
        self.settings = settings.combiner

        self._default_settings = {}
        self._add_default_settings(self._metacombiner_default_settings)

    def _add_default_settings(self, settings):
        for key, value in settings.items():
            assert key not in self._default_settings
            self._default_settings[key] = value

            # Fill current configuration with default values
            if key not in self.settings:
                self.settings[key] = value

    def _open_file(self, filename, mode="r"):
        filename = str(filename)
        if filename is None:
            return None
        elif filename.endswith(".gz"):
            return gzip.open(filename, mode)
        elif filename == "-":
            return sys.stdin
        else:
            return open(filename, mode)

    def _relative_to_config(self, file: str) -> Path:
        """
        translate string of a file path to the resolved Path when
        interpreting it as relative to the config file's location
        """
        config_file = Path(settings.config).resolve()
        file_path = Path(file)
        return (config_file.parent / file_path).resolve()

    def _resolve_model_file_path(self) -> Path:
        """
        translate model-file path into absolute Path using
        config file location
        """
        if self.settings["model-file"] is None:
            raise Exception("Can't resolve model file since no model file was provided")
        return self._relative_to_config(self.settings["model-file"])

    def train(self, file):
        raise NotImplementedError

    def combine(self, alerts, scores):
        raise NotImplementedError

    def save_trained_model(self):
        raise NotImplementedError

    def load_trained_model(self):
        raise NotImplementedError


class RunningAverageCombiner(Combiner):
    _runningaveragecombiner_default_settings = {
        "new_weight": 1.0,  # new_weight * current + (1 - new_weight) * old
        # Note: new_weight can also be an array defining one value for each IIDS
        "use_scores": False,  # Whether to use alerts or confidence scores
        "keys": None,  # May remain none. Defines keys of IDSs to combine
    }

    def __init__(self):
        super().__init__()
        self._add_default_settings(self._runningaveragecombiner_default_settings)

        self._running_avg = None

    def _get_activations(self, alerts, scores):
        data = scores if self.settings["use_scores"] else alerts

        # Setup variables
        if self.settings["keys"] is None:
            self.settings["keys"] = sorted(data.keys())

        if self._running_avg is None:
            self._running_avg = [0] * len(self.settings["keys"])

            # Validate if we have one weight for each IDS
            if isinstance(self.settings["new_weight"], Sequence):
                assert len(self.settings["new_weight"]) == len(self.settings["keys"])
            else:
                # Otherwise dupplicate the new_weight value accordingly
                self.settings["new_weight"] = [self.settings["new_weight"]] * len(
                    self.settings["keys"]
                )

        # Validate data
        if not set(data.keys()) == set(self.settings["keys"]):
            settings.logger.error("Keys of combiner do not match data")
            settings.logger.error(f'- data {",".join(data.keys())}')
            settings.logger.error(f'- training {",".join(self.settings["keys"])}')
            exit(1)

        # Calculate running average
        for i, key in enumerate(self.settings["keys"]):
            self._running_avg[i] *= 1 - self.settings["new_weight"][i]
            self._running_avg[i] += float(data[key]) * self.settings["new_weight"][i]
        return self._running_avg.copy()

    def train(self, file):
        events = []
        annotations = []

        settings.logger.info("Loading combiner training file")
        with self._open_file(file, "r") as f:
            for line in f.readlines():
                js = json.loads(line)

                events.append(self._get_activations(js["alerts"], js["scores"]))
                annotations.append(js["malicious"] is not False)

        self._running_avg = None
        return events, annotations

    def combine(self, scores, alerts):
        return self._get_activations(scores, alerts)
