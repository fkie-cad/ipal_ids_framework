import decimal
import sys

import orjson

import ipal_iids.settings as settings
from ids.ids import MetaIDS


class DecimalPlaces(MetaIDS):
    _name = "DecimalPlaces"
    _description = "Counts the number of decimal places."
    _requires = ["train.state", "live.state", "train.ipal", "live.ipal"]
    _decimal_default_settings = {}

    def __init__(self, name=None):
        super().__init__(name=name)
        self._add_default_settings(self._decimal_default_settings)

        self.key = None
        self.mins = {}
        self.maxs = {}

    def _get_decimal_places(self, value):
        if value is None or value is False or value is True:
            return None

        try:
            return decimal.Decimal(str(value)).as_tuple().exponent
        except:  # noqa: E722
            settings.logger.error(value)
            return None

    def train(self, ipal=None, state=None):
        # Figure out whether we train on ipal or state
        if ipal is not None and state is not None:
            settings.logger.warning("IPAL and State provided. Using state only!")
        elif ipal is not None:
            self.key = "data"
            fd = ipal
        elif state is not None:
            self.key = "state"
            fd = state

        # Load training file and parse data
        with self._open_file(fd) as f:
            for line in f:
                for k, v in orjson.loads(line)[self.key].items():
                    decimal = self._get_decimal_places(v)
                    if decimal is None:
                        continue

                    if k not in self.mins:
                        self.mins[k] = sys.maxsize
                        self.maxs[k] = -1
                    self.mins[k] = min(self.mins[k], decimal)
                    self.maxs[k] = max(self.maxs[k], decimal)

    def new_state_msg(self, msg):
        likelihood = 0
        alert = False

        for k, v in msg[self.key].items():
            if k not in self.mins:
                continue

            decimal_places = self._get_decimal_places(v)
            if decimal_places is None:
                continue

            if decimal_places < self.mins[k] or self.maxs[k] < decimal_places:
                alert |= True
                likelihood = max(likelihood, 1)

        return alert, likelihood

    def new_ipal_msg(self, msg):
        # There is no difference for this IDS in state or message format! It only depends on the configuration which features are used.
        return self.new_state_msg(msg)

    def save_trained_model(self):
        if self.settings["model-file"] is None:
            return False

        model = {
            "_name": self._name,
            "settings": self.settings,
            "key": self.key,
            "mins": self.mins,
            "maxs": self.maxs,
        }

        with self._open_file(self._resolve_model_file_path(), "wb") as f:
            f.write(orjson.dumps(model, option=orjson.OPT_INDENT_2))

        return True

    def load_trained_model(self):
        if self.settings["model-file"] is None:
            return False

        try:  # Open model file
            with self._open_file(self._resolve_model_file_path(), "rt") as f:
                model = orjson.loads(f.read())
        except FileNotFoundError:
            settings.logger.info(
                f"Model file {str(self._resolve_model_file_path())} not found."
            )
            return False

        # Load model
        assert self._name == model["_name"]
        self.settings = model["settings"]
        assert len(model["mins"]) == len(model["maxs"])
        self.key = model["key"]
        self.mins = model["mins"]
        self.maxs = model["maxs"]

        return True

    def visualize_model(self):
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(1)

        xs = np.arange(len(self.mins))
        labels = self.mins.keys()
        mins = np.array([self.mins[i] for i in labels])
        maxs = np.array([self.maxs[i] for i in labels])
        means = np.array([(min + max) / 2 for min, max in zip(mins, maxs)])

        ax.errorbar(
            xs,
            means,
            yerr=[means - mins, maxs - means],
            fmt=".k",
            lw=1,
            label="w/o delta",
            zorder=2,
        )

        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=90, ha="center")

        ax.legend()
        ax.set_ylabel("Minimum and Maximum")
        ax.set_title("DecimalPlacesIDS")

        return plt, fig
