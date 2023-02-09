import json
import itertools
import numpy as np

# Silence tensorflow
import os
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.FATAL)

import tensorflow.keras.callbacks  # noqa: E402
from tensorflow.keras.models import Sequential  # noqa: E402
from tensorflow.keras.layers import TimeDistributed  # noqa: E402
from tensorflow.keras.layers import Dense  # noqa: E402
from tensorflow.keras.layers import Dropout  # noqa: E402
from tensorflow.keras.layers import Bidirectional  # noqa: E402
from tensorflow.keras.layers import LSTM  # noqa: E402
from tensorflow.keras.optimizers import Adam  # noqa: E402

import ipal_iids.settings as settings  # noqa: E402
from ids.featureids import FeatureIDS  # noqa: E402


class BLSTM(FeatureIDS):

    _name = "BLSTM"
    _description = "Bidirectional LSTM."
    _blstm_default_settings = {
        # BLSTM GridSearch Parameters
        # TODO Better sample random values?
        "learning_rate": [0.001, 0.01, 0.1],
        "batch_size": [64, 128, 256],
        "dropout": [0.0, 0.1, 0.2],
        "hidden_layer_size": [64, 128, 256],
        "epochs": 50,
        "sequence_length": 4,
        "step": 4,
        "verbose": 1,
        "adjust": True,  # to use the extend-alarms.py script afterward
    }

    def __init__(self, name=None):
        super().__init__(name=name)
        self._add_default_settings(self._blstm_default_settings)

        self.blstm = None
        self.parameters = None
        self.buffer = []

    def make_sequences(self, Xs, Ys, seqlen, step=1):
        Xseq, Yseq = [], []
        for i in range(0, Xs.shape[0] - seqlen + 1, step):
            Xseq.append(Xs[i : i + seqlen])
            Yseq.append(Ys[i : i + seqlen])
        return np.array(Xseq), np.array(Yseq)

    def lstm_model(
        self, input_dim, output_dim, seq_len, hidden=128, dropout=0.0, lr=0.1
    ):
        model = Sequential()
        layers = {"input": input_dim, "hidden": hidden, "output": output_dim}

        model.add(
            Bidirectional(
                LSTM(layers["input"], return_sequences=True),
                merge_mode="concat",
                input_shape=(seq_len, layers["input"]),
            )
        )
        model.add(Dropout(dropout))

        activation = "softmax" if output_dim > 1 else "sigmoid"
        loss = "categorical_crossentropy" if output_dim > 1 else "binary_crossentropy"

        model.add(TimeDistributed(Dense(layers["output"], activation=activation)))

        model.compile(loss=loss, optimizer=Adam(learning_rate=lr), metrics=["acc"])

        model.summary(print_fn=settings.logger.info)

        return model

    # the IDS is given the path to file(s) containing its requested training data
    def train(self, ipal=None, state=None):
        if ipal and state:
            settings.logger.error("Only state or message supported")
            exit(1)

        if state is None:
            state = ipal

        events, annotation, _ = super().train(state=state)

        events = np.array(events)
        annotation = np.array([a is not False for a in annotation])

        if len(set(annotation)) <= 1:
            settings.logger.warning(
                "Training with a single class ({}) only!".format(set(annotation))
            )

        label_dim = 1  # True and False
        input_dim = len(events[0])

        # Get all combinations of the possible parameters for a "poor-mans grid search"
        tuned_parameters = {
            # Static
            "epochs": [self.settings["epochs"]],
            "sequence_length": [self.settings["sequence_length"]],
            "step": [self.settings["step"]],
            # Variable
            "learning_rate": self.settings["learning_rate"],
            "batch_size": self.settings["batch_size"],
            "dropout": self.settings["dropout"],
            "hidden_layer_size": self.settings["hidden_layer_size"],
        }
        keys, values = zip(*tuned_parameters.items())
        tuned_parameters = [dict(zip(keys, v)) for v in itertools.product(*values)]
        settings.logger.info(
            "Training on {} combinations".format(len(tuned_parameters))
        )

        best_loss = None
        for parameters in tuned_parameters:
            settings.logger.info(str(parameters))

            model = self.lstm_model(
                input_dim,
                label_dim,
                parameters["sequence_length"],
                hidden=parameters["hidden_layer_size"],
                dropout=parameters["dropout"],
                lr=parameters["learning_rate"],
            )

            # Split in training and testing data
            X, Y = self.make_sequences(
                events, annotation, parameters["sequence_length"], parameters["step"]
            )

            reduce_lr = tensorflow.keras.callbacks.ReduceLROnPlateau(
                monitor="loss",  # monitor: loss, acc, or lr
                factor=0.2,
                patience=3,
                min_lr=0.001,
            )

            h = model.fit(
                X,
                Y,
                batch_size=parameters["batch_size"],
                epochs=parameters["epochs"],
                callbacks=[reduce_lr],
                verbose=self.settings["verbose"],
            )
            h.history["loss"] = [float(x) for x in h.history["loss"]]
            h.history["acc"] = [float(x) for x in h.history["acc"]]
            h.history["lr"] = [float(x) for x in h.history["lr"]]

            if best_loss is None or h.history["loss"][-1] < best_loss:
                best_loss = h.history["loss"][-1]
                self.blstm = model
                self.parameters = parameters
                self.history = h.history

            settings.logger.info(
                "Model loss {}, acc {}, learning rate {}, best loss {}".format(
                    h.history["loss"][-1],
                    h.history["acc"][-1],
                    h.history["lr"][-1],
                    best_loss,
                )
            )

    def new_state_msg(self, msg):
        state = super().new_state_msg(msg)
        if state is None:
            return False, None

        self.buffer.append(state)
        if len(self.buffer) != self.parameters["sequence_length"]:
            return False, 0

        predict = self.blstm.predict(
            [self.buffer],
            batch_size=self.parameters["batch_size"],
            verbose=self.settings["verbose"],
        ).astype("float32")

        self.buffer = []
        predict = [float(x[0]) for x in predict[0]]
        alerts = [bool(x > 0.5) for x in predict]

        if "adjust" in self.settings:  # Annotate offset for adjust script
            if "adjust" not in msg:
                msg["adjust"] = {}

            offsets = list(range(-self.parameters["sequence_length"] + 1, 1))
            msg["adjust"][self._name] = list(zip(offsets, alerts, predict))

        return any(alerts), max(predict)

    def new_ipal_msg(self, msg):
        # There is no difference for this IDS in state or message format! It only depends on the configuration which features are used.
        return self.new_state_msg(msg)

    def save_trained_model(self):
        if self.settings["model-file"] is None:
            return False

        model = {
            "_name": self._name,
            "settings": self.settings,
            "preprocessors": super().save_trained_model(),
            "parameters": self.parameters,
            "history": self.history,
        }

        with self._open_file(self._resolve_model_file_path(), "wt") as f:
            f.write(json.dumps(model, indent=4))
        self.blstm.save(str(self._resolve_model_file_path()) + ".kreas")

        return True

    def load_trained_model(self):
        if self.settings["model-file"] is None:
            return False

        model_file_path = self._resolve_model_file_path()

        try:  # Open model file
            with self._open_file(model_file_path, mode="rt") as f:
                model = json.load(f)

        except FileNotFoundError:
            settings.logger.info(
                "Model file {} not found.".format(self.settings["model-file"])
            )
            return False

        # Load basic model
        assert self._name == model["_name"]
        self.settings = model["settings"]
        super().load_trained_model(model["preprocessors"])
        self.parameters = model["parameters"]
        self.history = model["history"]
        self.blstm = tensorflow.keras.models.load_model(str(model_file_path) + ".kreas")

        return True

    def visualize_model(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1)
        ax.plot(range(len(self.history["loss"])), self.history["loss"], label="loss")
        ax.plot(range(len(self.history["loss"])), self.history["acc"], label="accuracy")
        ax.plot(
            range(len(self.history["loss"])), self.history["lr"], label="learning rate"
        )
        ax.set_xlim(0, self.settings["epochs"])

        ax.set_ylabel("Training Loss/Accuracy/Learning Rate")
        ax.set_title(
            "Epochs: {} Seq-Len: {} Step: {} LR: {} BS: {} DR: {} HL: {}".format(
                self.parameters["epochs"],
                self.parameters["sequence_length"],
                self.parameters["step"],
                self.parameters["learning_rate"],
                self.parameters["batch_size"],
                self.parameters["dropout"],
                self.parameters["hidden_layer_size"],
            )
        )

        ax.legend()

        return plt, fig
