import json

import numpy as np

import ipal_iids.settings as settings
from ids.ids import MetaIDS

# This is an implementation of the two IDSs proposed in:
#
#    Lin, Chih-Yuan, Simin Nadjm-Tehrani, and Mikael Asplund. "Timing-based
#    anomaly detection in SCADA networks." International Conference on
#    Critical Information Infrastructures Security. Springer, Cham, 2017.
#
# Look into that paper to understand what is going on here


class InterArrivalTimeMean(MetaIDS):
    _name = "inter-arrival-mean"
    _description = "Mean inter-arrival time"
    _requires = ["train.ipal", "live.ipal"]
    _interarrivaltimemean_default_settings = {"N": 4, "W": 5, "alert_unknown": True}
    _supports_preprocessor = False

    def __init__(self, name=None):
        super().__init__(name=name)
        self._add_default_settings(self._interarrivaltimemean_default_settings)

        self.mean_model = {}
        self.sliding_windows = {}

    def _get_identifier(self, msg):
        # compute event identifier by concatenating source, destination, activity, message type and accessed data
        identifier = [
            msg["src"].split(":")[0],
            msg["dest"].split(":")[0],
            msg["activity"],
            str(msg["type"]),
        ]
        identifier += msg["data"].keys()
        return "-".join([str(i) for i in identifier])

    def train(self, ipal=None, state=None):
        events = {}

        # Load timestamps for each identifier
        with self._open_file(ipal) as f:
            for line in f.readlines():
                ipal_msg = json.loads(line)

                timestamp = ipal_msg["timestamp"]
                identifier = self._get_identifier(ipal_msg)

                if identifier not in events:
                    events[identifier] = []
                events[identifier].append(timestamp)

        # Calculate inter-arrival time and mean model
        settings.logger.info("Inter-arrival-time mean models:")

        for k in events.keys():
            interevent_times = []

            for i in range(len(events[k]) - 1):
                interevent_times.append(events[k][i + 1] - events[k][i])

            if len(interevent_times) <= self.settings["W"]:
                settings.logger.warning("Only single window of type {}".format(k))
                continue

            mu = np.mean(interevent_times)
            sigma = np.std(interevent_times)

            ul = mu + self.settings["N"] * sigma
            ll = mu - self.settings["N"] * sigma
            ll = max(0, ll)  # Only positive inter-arrival times

            self.mean_model[k] = {
                "ll": float(ll),
                "ul": float(ul),
                "mu": float(mu),
                "sigma": float(sigma),
            }
            self.sliding_windows[k] = {
                "timestamp": [],
                "malicious": [],
                "interevents": [],
            }

            settings.logger.info(
                "- {} [{}, {}] mean: {} sigma: {}".format(k, ll, ul, mu, sigma)
            )

    def new_ipal_msg(self, msg):
        identifier = self._get_identifier(msg)

        if identifier not in self.sliding_windows:  # Unknown message
            return self.settings["alert_unknown"], None

        else:  # Known message
            # Slide window
            if len(self.sliding_windows[identifier]["timestamp"]) == self.settings["W"]:
                del self.sliding_windows[identifier]["timestamp"][0]
                del self.sliding_windows[identifier]["malicious"][0]
                del self.sliding_windows[identifier]["interevents"][0]

            # Calculate inter-event time for last message
            if len(self.sliding_windows[identifier]["timestamp"]) > 0:
                intereventtime = (
                    msg["timestamp"] - self.sliding_windows[identifier]["timestamp"][-1]
                )
                self.sliding_windows[identifier]["interevents"].append(intereventtime)

            # Fill sliding window
            self.sliding_windows[identifier]["timestamp"].append(msg["timestamp"])
            self.sliding_windows[identifier]["malicious"].append(msg["malicious"])

            # Reached desired window size?
            if len(self.sliding_windows[identifier]["timestamp"]) < self.settings["W"]:
                return False, 0
            assert (
                len(self.sliding_windows[identifier]["interevents"])
                == self.settings["W"] - 1
            )

            # Check mean model
            iet_mean = np.mean(self.sliding_windows[identifier]["interevents"])
            alert = not (
                self.mean_model[identifier]["ll"] <= iet_mean
                and iet_mean <= self.mean_model[identifier]["ul"]
            )

            return alert, iet_mean - self.mean_model[identifier]["mu"]

    def save_trained_model(self):
        if self.settings["model-file"] is None:
            return False

        model = {
            "_name": self._name,
            "settings": self.settings,
            "mean_model": self.mean_model,
        }

        with self._open_file(self._resolve_model_file_path(), mode="wt") as f:
            f.write(json.dumps(model, indent=4) + "\n")

        return True

    def load_trained_model(self):
        if self.settings["model-file"] is None:
            return False

        try:  # Open model file
            with self._open_file(self._resolve_model_file_path(), mode="rt") as f:
                model = json.load(f)
        except FileNotFoundError:
            settings.logger.info(
                "Model file {} not found.".format(str(self._resolve_model_file_path()))
            )
            return False

        # Load model
        assert self._name == model["_name"]
        self.settings = model["settings"]
        self.mean_model = model["mean_model"]

        for k in model["mean_model"].keys():
            self.sliding_windows[k] = {
                "timestamp": [],
                "malicious": [],
                "interevents": [],
            }

        return True

    def visualize_model(self):
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(1)

        xs = np.arange(len(self.mean_model))
        labels = [x for x in self.mean_model]
        mins = np.array([self.mean_model[label]["ll"] for label in labels])
        maxs = np.array([self.mean_model[label]["ul"] for label in labels])
        means = np.array([self.mean_model[label]["mu"] for label in labels])
        stds = np.array([self.mean_model[label]["sigma"] for label in labels])

        ax.errorbar(
            xs,
            means,
            [means - mins, maxs - means],
            fmt=".k",
            ecolor="gray",
            lw=1,
            zorder=2,
        )
        ax.errorbar(xs, means, stds, fmt="ok", lw=3, zorder=1)

        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=90, ha="center")

        ax.set_ylabel("Inter arrival time [in s]")
        ax.set_title("N: {} W: {}".format(self.settings["N"], self.settings["W"]))

        return plt, fig
