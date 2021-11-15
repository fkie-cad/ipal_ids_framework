import json
import math

from collections.abc import Iterable

import ipal_iids.settings as settings

from ipal_iids.preprocessors.utils import get_all_preprocessors
from .ids import MetaIDS


class FeatureIDS(MetaIDS):

    _requires = ["train.ipal", "train.state", "live.ipal", "live.state"]
    _featureids_default_settings = {
        "features": [],  # Feature list forwarded to the IDS after preprocessing
        "preprocessors": [],  # List of preprocessors applied to the data
        "trainon": 1.0,  # Train preprocessor on 100% and IDS on first x% of the data
        "save-training": None,
        "allow-none": False,
    }
    _supports_preprocessor = True

    preprocessors = []

    def __flatten(self, array):
        # https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists?page=1&tab=votes#tab-top
        for el in array:
            if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
                yield from self.__flatten(el)
            else:
                yield el

    def __init__(self, args, name=None):
        super().__init__(args, name=name)
        self._add_default_settings(self._featureids_default_settings)

        self.preprocessors = []
        self.features = []

    def _get_val(self, msg, val):
        # Lookup value
        for index in val:
            if index in msg:
                msg = msg[index]
            else:
                if not self.settings["allow-none"]:
                    settings.logger.warning(
                        "Index '{}' not found in msg {}. Returning None".format(
                            index, msg
                        )
                    )
                return None

        if msg is None and self.settings["allow-none"]:
            return None

        # Try to parse as float
        try:
            val = float(msg)
            if not math.isnan(val):
                return val
            else:
                settings.logger.warning("Found Nan in data. Replacing with '0'")
                return 0
        except ValueError:  # Non-float data
            return msg

    def _extract_features(self, msg):
        if ["hash"] in self.features:
            self._add_msg_hash(msg, nbytes=2)

        return [self._get_val(msg, feature) for feature in self.features]

    # the IDS is given the path to file(s) containing its requested training data
    def train(self, state=None):

        # Build preprocessors from settings
        for pre in self.settings["preprocessors"]:
            apply = [f in pre["features"] for f in self.settings["features"]]
            self.preprocessors.append(get_all_preprocessors()[pre["method"]](apply))

        self.features = [f.split(";") for f in self.settings["features"]]

        events = []
        annotations = []
        timestamps = []

        # Load features from training file
        with self._open_file(state) as state_file:
            for msg in state_file.readlines():
                msg = json.loads(msg)
                state = self._extract_features(msg)

                if None not in state or self.settings["allow-none"]:
                    events.append(state)
                    annotations.append(msg["malicious"])
                    timestamps.append(msg["timestamp"])
                else:
                    settings.logger.info("None in state. Skipping message!")

        # Train and apply preprocessors
        settings.logger.info("Raw features: {}".format(events[0]))
        for pre in self.preprocessors:
            pre.fit(events)
            events = [pre.transform(e) for e in events]
            settings.logger.info("{} features: {}".format(pre._name, events[0]))
        events = [list(self.__flatten(e)) for e in events]
        settings.logger.info("Final features: {}".format(events[0]))

        # Train IDS only on first x% of the data
        N = int(len(events) * self.settings["trainon"])
        settings.logger.info(
            "Preprocesser trained on {} IDS is training on first {}".format(
                len(events), len(events[:N])
            )
        )

        if "save-training" in self.settings and self.settings["save-training"]:
            with self._open_file(
                self._relative_to_config(self.settings["save-training"]), mode="wt"
            ) as f:
                for e, a, t in zip(events[:N], annotations[:N], timestamps[:N]):
                    f.write(
                        json.dumps(
                            {
                                "timestamp": t,
                                "state": {i: e[i] for i in range(len(e))},
                                "malicious": a,
                            }
                        )
                        + "\n"
                    )

        for pre in self.preprocessors:
            pre.reset()  # reset preprocessors before going live

        return events[:N], annotations[:N], timestamps[:N]

    def new_state_msg(self, msg):

        state = self._extract_features(msg)
        if None in state and not self.settings["allow-none"]:
            settings.logger.info("None in state. Skipping message")
            return None

        for pre in self.preprocessors:
            state = pre.transform(state)
        state = list(self.__flatten(state))

        return state

    def save_trained_model(self):
        model = {
            "features": self.features,
            "settings": self.settings,
            "preprocessors": [],
        }

        for pre in self.preprocessors:
            model["preprocessors"].append((pre._name, pre.get_fitted_model()))

        return model

    def load_trained_model(self, model):
        self.settings = model["settings"]
        self.features = model["features"]

        for name, pre_model in model["preprocessors"]:
            self.preprocessors.append(
                get_all_preprocessors()[name].from_fitted_model(pre_model)
            )
