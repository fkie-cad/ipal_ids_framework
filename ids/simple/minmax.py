import json

import ipal_iids.settings as settings
from ids.featureids import FeatureIDS


class MinMax(FeatureIDS):

    _name = "MinMax"
    _description = "The MinMax detects whether a sensor's/actuator's current value exceeds the range observed in the training data and raises an alarm if any observation falls outside that range. This approach is motivated by the intuition that process values of industrial systems relate to physical measurements or setpoints and thus usually obey certain limits. E.g., temperatures below the freezing point of a liquid are not desirable for pumping it through pipes. Even if the physical setup does not limit the value range, operational requirements may impose restrictions on the allowed data range, e.g., the pH value of a liquid may not exceed a specific range to be non-hazardous. Thus, we assume that an industrial system exhibits a class of values inside defined minimum and maximum limits."
    _requires = ["train.ipal", "live.ipal", "train.state", "live.state"]
    _minmax_default_settings = {"threshold": 1.0, "discrete_threshold": 10}

    def __init__(self, name=None):
        super().__init__(name=name)
        self._add_default_settings(self._minmax_default_settings)

        self.maxs = {}
        self.mins = {}
        self.deltas = {}

    def train(self, ipal=None, state=None):
        if ipal is not None and state is not None:
            settings.logger.warning("Only state OR ipal supported. Using state now.")
        elif state is None:
            state = ipal

        # Call preprocessor
        events, annotations, _ = super().train(state=state)

        # Check input
        if len(set(annotations) - set([False])) > 0:
            settings.logger.warning("IDS expects benign data only!")

        # Train on input
        for i in range(len(events[0])):
            data = [e[i] for e in events if e[i] is not None]
            self.mins[i] = min(data)
            self.maxs[i] = max(data)
            self.deltas[i] = (self.maxs[i] - self.mins[i]) / 2

            if len(set(data)) <= self.settings["discrete_threshold"]:
                self.deltas[i] = 0  # No threshold for discrete process values

            settings.logger.info(
                "Sensor {} Min: {} Max: {} Delta: {}".format(
                    i, self.mins[i], self.maxs[i], self.deltas[i]
                )
            )

    def new_state_msg(self, msg):
        likelihood = 0
        alert = False

        state = super().new_state_msg(msg)
        if state is None:
            return alert, likelihood

        for value, minimum, maximum, delta in zip(
            state, self.mins.values(), self.maxs.values(), self.deltas.values()
        ):
            err = delta * self.settings["threshold"]

            if value is None:  # None is not malicious
                likelihood = max(likelihood, 0)

            elif minimum <= value and value <= maximum:  # 0: if within min/max region
                likelihood = max(likelihood, 0)

            else:  # scale from 0 - 1 in error region
                overshoot = (
                    abs(((maximum + minimum) * 0.5 - value)) - (maximum - minimum) * 0.5
                )
                likelihood = max(likelihood, overshoot / (1 if err == 0 else err))

            # Trigger alert
            if value is not None and (value < minimum - err or maximum + err < value):
                alert |= True

        return alert, likelihood

    def new_ipal_msg(self, msg):
        # There is no difference for this IDS in state or message format! It only depends on the configuration which features are used.
        return self.new_state_msg(msg)

    def save_trained_model(self):
        if self.settings["model-file"] is None:
            return False

        model = {
            "_name": self._name,
            "preprocessors": super().save_trained_model(),
            "settings": self.settings,
            "mins": self.mins,
            "maxs": self.maxs,
            "deltas": self.deltas,
        }

        with self._open_file(self._resolve_model_file_path(), "wt") as f:
            f.write(json.dumps(model, indent=4))

        return True

    def load_trained_model(self):
        if self.settings["model-file"] is None:
            return False

        try:  # Open model file
            with self._open_file(self._resolve_model_file_path(), "rt") as f:
                model = json.loads(f.read())
        except FileNotFoundError:
            settings.logger.info(
                "Model file {} not found.".format(str(self._resolve_model_file_path()))
            )
            return False

        # Load model
        assert self._name == model["_name"]
        super().load_trained_model(model["preprocessors"])
        self.settings = model["settings"]
        assert len(model["mins"]) == len(model["maxs"]) == len(model["deltas"])
        self.mins = model["mins"]
        self.maxs = model["maxs"]
        self.deltas = model["deltas"]

        return True

    def visualize_model(self):
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(1)

        xs = np.arange(len(self.mins))
        labels = self.mins.keys()
        mins = np.array([self.mins[i] for i in labels])
        maxs = np.array([self.maxs[i] for i in labels])
        deltas = np.array([self.deltas[i] * self.settings["threshold"] for i in labels])
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

        ax.errorbar(
            xs,
            means,
            yerr=[means - mins + deltas, maxs - means + deltas],
            fmt=".r",
            lw=1,
            label="w/ delta",
            zorder=1,
        )

        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=90, ha="center")

        ax.legend()
        ax.set_ylabel("Minimum and Maximum")
        ax.set_title("MinMaxIDS")

        return plt, fig
