import json

import ipal_iids.settings as settings
from ids.featureids import FeatureIDS


class Histogram(FeatureIDS):

    _name = "Histogram"
    _description = "The Histogram approach tracks the process values' distribution within a fixed-sized window and tests whether it is in line with a histogram seen during training. The underlying intuition expects a similar distribution of reoccurring values between process cycles. This approach can detect the existence and absence of frequent value changes. The histograms are created by counting the number of times each distinct value appears in a sliding window. We merge them into a single histogram that covers each value's minimum and maximum occurrences across all distinct fixed-sized windows. The window size should match the duration of a process cycle, which could be automatically determined in an additional run over the dataset prior to training the histograms. Histogram only applies for process values with a few distinct values, as comparing two histograms value-by-value is unfeasible for noisy sensor data."
    _requires = ["train.ipal", "live.ipal", "train.state", "live.state"]
    _histogram_default_settings = {
        "window_size": 100,
        "threshold": 1.0,
        "discrete_threshold": 10,
    }

    def __init__(self, name=None):
        super().__init__(name=name)
        self._add_default_settings(self._histogram_default_settings)

        self.hist = {}
        self.deltas = {}

        self._reset()

    def _reset(self):
        self._cur = {}
        self._buffer = {}

    def _update(self, sensor, value):

        if sensor not in self._cur:
            self._cur[sensor] = {i: 0 for i in self.hist[sensor].keys()}
            self._buffer[sensor] = []

        self._cur[sensor][value] += 1
        self._buffer[sensor].append(value)

        if len(self._buffer[sensor]) > self.settings["window_size"]:
            self._cur[sensor][self._buffer[sensor].pop(0)] -= 1

        return len(self._buffer[sensor]) == self.settings["window_size"]

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

        # Prepare data structures and find non-discrete values
        hist = {}
        for i in range(len(events[0])):

            vals = set([e[i] for e in events])
            if len(vals) > self.settings["discrete_threshold"]:  # Skip non-discrete
                hist[i] = None
                self.hist[i] = None
                self.deltas[i] = None

            else:  # Prepare histogram
                hist[i] = {val: [] for val in vals}
                self.hist[i] = {val: [self.settings["window_size"], 0] for val in vals}
                self.deltas[i] = {val: None for val in vals}

        # Train on input
        for e in events:
            for i in range(len(events[0])):  # For each sensor
                if self.hist[i] is None:  # Skip non-discrete sensors
                    continue

                complete = self._update(i, e[i])
                if complete:  # Store histogram
                    for val in self.hist[i]:
                        hist[i][val].append(self._cur[i][val])

        # Calculate deltas and real values
        for i in range(len(events[0])):

            if self.hist[i] is None:  # Skip non-discrete sensors
                settings.logger.info("Sensor {} ignored".format(i))
                continue

            for val in self.hist[i]:
                self.hist[i][val] = [min(hist[i][val]), max(hist[i][val])]
                self.deltas[i][val] = (self.hist[i][val][1] - self.hist[i][val][0]) / 2

                settings.logger.info(
                    "Sensor {} Val {}: [{}, {}] +-{}".format(
                        i,
                        val,
                        self.hist[i][val][0],
                        self.hist[i][val][1],
                        self.deltas[i][val],
                    )
                )

        # Reset values
        self._reset()

    def _is_valid(self, sensor):
        likelihood = 0

        for val in self.hist[sensor]:  # One failed bucket suffices
            tmin, tmax = self.hist[sensor][val]
            err = self.deltas[sensor][val] * self.settings["threshold"]
            cur = self._cur[sensor][val]

            if cur < tmin or tmax < cur:  # outside normal bounary
                overshoot = abs(((tmax + tmin) * 0.5 - cur)) - (tmax - tmin) * 0.5
                likelihood = max(likelihood, overshoot / (1 if err == 0 else err))

            if cur < tmin - err or tmax + err < cur:  # Outside the histogram
                return False, likelihood

        return True, likelihood

    def new_state_msg(self, msg):
        likelihood = 0
        alert = False

        state = super().new_state_msg(msg)
        if state is None:
            return alert, likelihood

        for i in range(len(state)):
            if self.hist[i] is None:  # Ignore sensors with too many values
                continue

            if state[i] not in self.hist[i]:  # Alert unknown value
                alert |= True
                likelihood = max(likelihood, 1)
                continue

            complete = self._update(i, state[i])
            if not complete:  # histogram not full yet
                continue

            is_valid, local_likelihood = self._is_valid(i)
            likelihood = max(likelihood, local_likelihood)
            alert |= not is_valid

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
            "hist": self.hist,
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
        assert len(model["deltas"]) == len(model["hist"])

        super().load_trained_model(model["preprocessors"])
        self.settings = model["settings"]
        self.hist = {int(k): v for k, v in model["hist"].items()}
        self.deltas = {int(k): v for k, v in model["deltas"].items()}

        for k, v in self.hist.items():  # str -> int for keys
            if v is not None:
                self.hist[k] = {
                    float(kk) if "." in kk else int(kk): vv for kk, vv in v.items()
                }
        for k, v in self.deltas.items():  # str -> int for keys
            if v is not None:
                self.deltas[k] = {
                    float(kk) if "." in kk else int(kk): vv for kk, vv in v.items()
                }

        self._reset()

        return True

    def visualize_model(self):
        import matplotlib.pyplot as plt
        import numpy as np

        histkeys = [k for k, v in self.hist.items() if v is not None]

        if len(histkeys) > 1:
            fig, axs = plt.subplots(nrows=(len(histkeys) + 1) // 2, ncols=2)
            axs = [ax for row in axs for ax in row]
        else:
            fig, axs = plt.subplots(nrows=1, ncols=1)
            axs = [axs]

        for i, ax in zip(histkeys, axs):

            xs = np.arange(len(self.hist[i]))
            mins = np.array([v[0] for _, v in self.hist[i].items()])
            maxs = np.array([v[1] for _, v in self.hist[i].items()])
            means = np.array([(min + max) / 2 for min, max in zip(mins, maxs)])
            deltas = np.array(
                [v * self.settings["threshold"] for _, v in self.deltas[i].items()]
            )

            ax.errorbar(
                xs,
                means,
                [means - mins, maxs - means],
                fmt=".k",
                lw=1,
                zorder=2,
            )

            ax.errorbar(
                xs,
                means,
                [means - mins + deltas, maxs - means + deltas],
                fmt=".r",
                lw=1,
                zorder=1,
            )

            ax.set_xticks(xs)
            ax.set_xticklabels(self.hist[i].keys())

            ax.set_ylim([0, self.settings["window_size"]])
            ax.set_ylabel(i)

        for i in range(len(histkeys), len(axs)):  # Remove remaining plot
            axs[i].axis("off")

        return plt, fig
