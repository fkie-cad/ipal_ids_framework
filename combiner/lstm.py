import json
import logging

# Silence tensorflow
import os

import joblib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.FATAL)

from tensorflow.keras.layers import LSTM  # noqa: E402
from tensorflow.keras.layers import Dense  # noqa: E402
from tensorflow.keras.models import Sequential  # noqa: E402
from tensorflow.keras.optimizers import Adam  # noqa: E402

import ipal_iids.settings as settings  # noqa: E402

from .combiner import Combiner  # noqa: E402


class LSTMCombiner(Combiner):
    _name = "LSTM"
    _description = "Learns a time-aware LSTM combiner."
    _requires_training = True
    _lstm_default_settings = {
        "epochs": 20,
        # Overall, the combiner looks back lookback * stride data points
        "lookback": 30,
        "stride": 1,
        "use_scores": False,
        "verbose": 0,
    }

    def __init__(self):
        super().__init__()
        self._add_default_settings(self._lstm_default_settings)

        self.model = None
        self.keys = None
        self.buffer = []
        self.window_size = self.settings["lookback"] * self.settings["stride"]

    def _get_activations(self, alerts, scores):
        data = scores if self.settings["use_scores"] else alerts

        if not set(data.keys()) == set(self.keys):
            settings.logger.error("Keys of combiner do not match data")
            settings.logger.error("- data keys: {}".format(",".join(data.keys())))
            settings.logger.error("- combiner keys: {}".format(",".join(self.keys)))
            exit(1)

        return [float(data[ids]) for ids in self.keys]

    def _lstm_model(self, input_dim):
        model = Sequential()
        model.add(LSTM(input_dim, input_shape=(self.settings["lookback"], input_dim)))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(loss="binary_crossentropy", optimizer=Adam(), metrics=["acc"])
        model.summary(print_fn=settings.logger.info)

        return model

    def train(self, file):
        buffer = []
        seq = []
        annotations = []

        settings.logger.info("Loading combiner training file")
        with self._open_file(file, "r") as f:
            for line in f.readlines():
                js = json.loads(line)

                if self.keys is None:
                    self.keys = sorted(js["scores"].keys())

                # Manage buffer
                buffer.append(self._get_activations(js["alerts"], js["scores"]))
                if len(buffer) < self.window_size:
                    continue
                elif len(buffer) > self.window_size:
                    buffer.pop(0)

                # Add training sequence
                seq.append(buffer[:: -self.settings["stride"]])
                annotations.append(js["malicious"] is not False)

        self.model = self._lstm_model(len(self.keys))
        settings.logger.info(f"Training LSTM for {self.settings['epochs']} epochs...")
        self.model.fit(
            seq,
            annotations,
            epochs=self.settings["epochs"],
            verbose=self.settings["verbose"],
        )

    def combine(self, alerts, scores):
        # Manage buffer
        self.buffer.append(self._get_activations(alerts, scores))
        if len(self.buffer) < self.window_size:
            return False, 0
        elif len(self.buffer) > self.window_size:
            self.buffer.pop(0)

        sequence = self.buffer[:: -self.settings["stride"]]
        prediction = float(self.model.predict([sequence], verbose=False)[0][0])
        return prediction > 0.5, prediction

    def save_trained_model(self):
        if self.settings["model-file"] is None:
            return False

        model = {
            "_name": self._name,
            "settings": self.settings,
            "model": self.model,
            "keys": self.keys,
            "window_size": self.window_size,
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
        self.model = model["model"]
        self.keys = model["keys"]
        self.window_size = model["window_size"]
        self.buffer = []

        return True
