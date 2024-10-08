import random

import orjson

import ipal_iids.settings as settings
from ids.ids import MetaIDS


class DummyIDS(MetaIDS):
    _name = "DummyIDS"
    _description = "Dummy IDS returns either True, False, or random."
    _requires = ["train.ipal", "live.ipal", "train.state", "live.state"]
    _optimalids_default_settings = {
        "ids-value": False
    }  # True, False, 'probability of alert'
    _supports_preprocessor = False

    def __init__(self, name=None):
        super().__init__(name=name)
        self._add_default_settings(self._optimalids_default_settings)

    def train(self, ipal=None, state=None):
        pass

    def new_ipal_msg(self, msg):
        if isinstance(self.settings["ids-value"], bool):
            alert = self.settings["ids-value"]

        else:
            alert = random.choices(
                [True, False],
                weights=(self.settings["ids-value"], 1 - self.settings["ids-value"]),
                k=1,
            )[0]

        return alert, 1 if alert else 0

    def new_state_msg(self, msg):
        return self.new_ipal_msg(msg)

    def save_trained_model(self):
        if self.settings["model-file"] is None:
            return False

        model = {"_name": self._name, "settings": self.settings}
        with self._open_file(self._resolve_model_file_path(), mode="wb") as f:
            f.write(
                orjson.dumps(
                    model, option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE
                )
            )

        return True

    def load_trained_model(self):
        if self.settings["model-file"] is None:
            return False

        try:  # Open model file
            with self._open_file(self._resolve_model_file_path(), mode="rb") as f:
                model = orjson.loads(f.read())
        except FileNotFoundError:
            settings.logger.info(
                f"Model file {str(self._resolve_model_file_path())} not found."
            )
            return False

        # Load model
        assert self._name == model["_name"]
        self.settings = model["settings"]

        return True

    def visualize_model(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1)
        plt.text(0.5, 0.5, "Nothing to plot for DummyIDS", ha="center", va="center")

        return plt, fig
