import json

import ipal_iids.settings as settings
from ids.ids import MetaIDS


class OptimalIDS(MetaIDS):

    _name = "Optimal"
    _description = "Optimal IDS returns the malicious field as classification."
    _requires = ["train.ipal", "live.ipal", "train.state", "live.state"]
    _optimalids_default_settings = {"invert": False}
    _supports_preprocessor = False

    def __init__(self, name=None):
        super().__init__(name=name)
        self._add_default_settings(self._optimalids_default_settings)

    def train(self, ipal=None, state=None):
        pass

    def new_ipal_msg(self, msg):
        alert = (msg["malicious"] not in [False, None]) ^ self.settings["invert"]
        score = 1 if alert else 0
        return alert, score

    def new_state_msg(self, msg):
        alert = (msg["malicious"] not in [False, None]) ^ self.settings["invert"]
        score = 1 if alert else 0
        return alert, score

    def save_trained_model(self):
        if self.settings["model-file"] is None:
            return False

        model = {"_name": self._name, "settings": self.settings}
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

        return True

    def visualize_model(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1)
        plt.text(0.5, 0.5, "Nothing to plot for OptimalIDS", ha="center", va="center")

        return plt, fig
