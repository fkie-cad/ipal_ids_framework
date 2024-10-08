from pathlib import Path

import orjson

import ipal_iids.iids as iids
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

    def _open_file(self, filename, mode="r", force_gzip=False):
        return iids.open_file(filename, mode, force_gzip=force_gzip)

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

    def _get_activations(self, alerts, scores):
        if "use_scores" not in self.settings or self.settings["use_scores"] is None:
            settings.logger.info(
                "Combiner: 'use_scores' not defined. Defaulting to 'False'"
            )
            self.settings["use_scores"] = False

        if "keys" not in self.settings or self.settings["keys"] is None:
            self.settings["keys"] = list(alerts.keys())
            settings.logger.info(
                f"Combiner: 'keys' not defined. Defaulting to '{self.settings['keys']}'"
            )

        data = scores if self.settings["use_scores"] else alerts
        return [data[ids] for ids in self.settings["keys"]]

    def _load_training(self, file):
        events = []
        annotations = []

        settings.logger.info("Loading combiner training file")
        with self._open_file(file, "r") as f:
            for line in f:
                js = orjson.loads(line)

                events.append(self._get_activations(js["alerts"], js["scores"]))
                annotations.append(js["malicious"] is not False)

        return events, annotations

    def train(self, file):
        raise NotImplementedError

    def combine(self, alerts, scores):
        raise NotImplementedError

    def save_trained_model(self):
        raise NotImplementedError

    def load_trained_model(self):
        raise NotImplementedError
