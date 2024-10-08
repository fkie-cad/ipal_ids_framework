import orjson

import ipal_iids.settings as settings
from ids.ids import MetaIDS


class ExistsIDS(MetaIDS):
    _name = "ExistsIDS"
    _description = "Tests if a value has been seen before or not."
    _requires = ["train.state", "live.state", "train.ipal", "live.ipal"]
    _exists_default_settings = {"exclude": [], "threshold": 10}

    def __init__(self, name=None):
        super().__init__(name=name)
        self._add_default_settings(self._exists_default_settings)

        self.key = None
        self.exists = {}
        self.count = 0

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
                    if k in self.settings["exclude"]:
                        continue

                    if k not in self.exists:
                        self.exists[k] = set()
                    self.exists[k].add(v)

    def new_state_msg(self, msg):
        anomaly = False

        for k, v in msg[self.key].items():
            if k in self.settings["exclude"]:
                continue
            elif k not in self.exists:
                anomaly |= True
            elif v not in self.exists[k]:
                settings.logger.debug(f"{k} {v}")
                anomaly |= True

        if anomaly:
            self.count += 1
        else:
            self.count = 0

        alert = self.count > self.settings["threshold"]
        return alert, self.count / self.settings["threshold"]

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
            "exists": {k: list(v) for k, v in self.exists.items()},
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
        self.key = model["key"]
        self.exists = {k: set(v) for k, v in model["exists"].items()}

        return True
