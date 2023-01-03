import gzip
import json
import sys

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
