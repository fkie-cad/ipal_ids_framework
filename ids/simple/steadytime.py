import json

import ipal_iids.settings as settings
from ids.featureids import FeatureIDS


class SteadyTime(FeatureIDS):
    _name = "Steadytime"
    _description = "The Steadytime approach detects whether a sensor/actuator remains static, i.e., does not change its value, for a shorter or longer time than seen during training. This approach is motivated by the observation that an attack, e.g., freezing a sensor/actuator such as a pressure relief valve, cannot be detected by checking whether a value or the velocity of a value change remains within certain boundaries. Since a steady state is difficult to define for noisy sensor data, Steadytime takes only process values into account if the number of distinct values during training is sufficiently small."
    _requires = ["train.ipal", "live.ipal", "train.state", "live.state"]
    _steadytime_default_settings = {
        "threshold": 1.0,
        "discrete_threshold": 10,
        "adjust": True,  # to use the extend-alarms.py script afterward
    }

    def __init__(self, name=None):
        super().__init__(name=name)
        self._add_default_settings(self._steadytime_default_settings)

        self.time = {}
        self.deltas = {}

        self._reset()

    def _reset(self):
        self._cur = {}
        self._cur_time = {}

    def _update(self, sensor, value):
        if sensor not in self._cur:
            self._cur[sensor] = value
            self._cur_time[sensor] = 0

        if self._cur[sensor] == value:  # Still the same value
            self._cur[sensor] = value
            self._cur_time[sensor] += 1

            return False, value, self._cur_time[sensor]

        else:  # Block ended (safe and reset to new value)
            oldtime = self._cur_time[sensor]
            oldvalue = self._cur[sensor]

            self._cur[sensor] = value
            self._cur_time[sensor] = 1

            return True, oldvalue, oldtime

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
        time = {}
        for i in range(len(events[0])):
            vals = set([e[i] for e in events])
            if len(vals) > self.settings["discrete_threshold"]:  # Skip non-discrete
                time[i] = None
                self.time[i] = None
                self.deltas[i] = None

            else:  # Prepare time
                time[i] = {val: [] for val in vals}
                self.time[i] = {val: None for val in vals}
                self.deltas[i] = {val: None for val in vals}

        # Train on input
        for e in events:
            for i in range(len(events[0])):  # For each sensor
                if self.time[i] is None:  # Skip non-discrete sensors
                    continue

                complete, val, t = self._update(i, e[i])
                if complete:  # Store time
                    time[i][val].append(t)

        # Calculate deltas and real values
        for i in range(len(events[0])):
            if self.time[i] is None:  # Skip non-discrete sensors
                settings.logger.info("Sensor {} ignored".format(i))
                continue

            for val in self.time[i]:
                if len(time[i][val]) > 0:
                    self.time[i][val] = [min(time[i][val]), max(time[i][val])]
                    self.deltas[i][val] = (
                        self.time[i][val][1] - self.time[i][val][0]
                    ) / 2

                settings.logger.info(
                    "Sensor {} Val {}: {} +-{}".format(
                        i,
                        val,
                        self.time[i][val],
                        self.deltas[i][val],
                    )
                )

        # Reset values
        self._reset()

    def _is_valid(self, sensor, val, time):
        likelihood = 0

        if self.time[sensor][val] is None:  # No data is valid
            return True, likelihood

        tmin, tmax = self.time[sensor][val]
        err = self.deltas[sensor][val] * self.settings["threshold"]

        if time < tmin or tmax < time:  # outside normal bounary
            overshoot = abs(((tmax + tmin) * 0.5 - time)) - (tmax - tmin) * 0.5
            likelihood = max(likelihood, overshoot / (1 if err == 0 else err))

        return tmin - err <= time and time <= tmax + err, likelihood

    def new_state_msg(self, msg):
        likelihood = 0
        alert = False

        state = super().new_state_msg(msg)
        if state is None:
            return alert, likelihood

        for i in range(len(state)):
            if self.time[i] is None:  # Ignore sensors with too many values
                continue

            if state[i] not in self.time[i]:  # Alert unknown value
                alert |= True
                likelihood = max(likelihood, 1)
                continue

            complete, val, time = self._update(i, state[i])
            if not complete:  # state not changed yet
                continue

            is_valid, local_likelihood = self._is_valid(i, val, time)
            likelihood = max(likelihood, local_likelihood)

            if not is_valid:  # found a violation
                alert |= True

                # Add alarm adjustment information if required
                if self.settings["adjust"]:
                    if "adjust" not in msg:
                        msg["adjust"] = {}

                    if time > self.time[i][val][1]:  # only alert excess
                        time -= self.time[i][val][1]

                    msg["adjust"][self._name] = [
                        [t, True, likelihood] for t in range(-int(time), 0)
                    ]
                    msg["adjust"][self._name] += [[0, False, 0]]  # Reset current alarm

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
            "time": self.time,
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
        assert len(model["deltas"]) == len(model["time"])

        super().load_trained_model(model["preprocessors"])
        self.settings = model["settings"]
        self.time = {int(k): v for k, v in model["time"].items()}
        self.deltas = {int(k): v for k, v in model["deltas"].items()}

        for k, v in self.time.items():  # str -> int for keys
            if v is not None:
                self.time[k] = {
                    float(kk) if "." in kk else int(kk): vv for kk, vv in v.items()
                }
        for k, v in self.deltas.items():  # str -> int for keys
            if v is not None:
                self.deltas[k] = {
                    float(kk) if "." in kk else int(kk): vv for kk, vv in v.items()
                }

        self._reset()

        return True

    def visualize_model(self):  # TODO prettify plot
        import matplotlib.pyplot as plt
        import numpy as np

        timekeys = [k for k, v in self.time.items() if v is not None]

        if len(timekeys) > 1:
            fig, axs = plt.subplots(nrows=(len(timekeys) + 1) // 2, ncols=2)
            axs = [ax for row in axs for ax in row]
        else:
            fig, axs = plt.subplots(nrows=1, ncols=1)
            axs = [axs]

        for i, ax in zip(timekeys, axs):
            xs = np.arange(len(self.time[i]))
            mins = np.array(
                [v[0] if v is not None else np.nan for _, v in self.time[i].items()]
            )
            maxs = np.array(
                [v[1] if v is not None else np.nan for _, v in self.time[i].items()]
            )
            means = np.array([(min + max) / 2 for min, max in zip(mins, maxs)])
            deltas = np.array(
                [
                    v * self.settings["threshold"] if v is not None else np.nan
                    for _, v in self.deltas[i].items()
                ]
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
            ax.set_xticklabels(self.time[i].keys())

            ax.set_ylabel(i)

        for i in range(len(timekeys), len(axs)):  # Remove remaining plot
            axs[i].axis("off")

        return plt, fig
