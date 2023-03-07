import joblib
from sklearn.ensemble import IsolationForest as IsolationForestClassifier

import ipal_iids.settings as settings
from ids.featureids import FeatureIDS


class IsolationForest(FeatureIDS):
    _name = "IsolationForest"
    _description = "Isolation forest classifier."
    _isolationforest_default_settings = {
        # IsolationForest Parameters
        "n_estimators": 100,
        "max_samples": "auto",
        "contamination": "auto",
        "max_features": 1,
        "bootstrap": False,
        "random_state": None,
        "warm_start": False,
    }

    def __init__(self, name=None):
        super().__init__(name=name)
        self._add_default_settings(self._isolationforest_default_settings)

        self.ifc = None

    # the IDS is given the path to file(s) containing its requested training data
    def train(self, ipal=None, state=None):
        if ipal and state:
            settings.logger.error("Only state or message supported")
            exit(1)

        if state is None:
            state = ipal

        events, _, _ = super().train(state=state)  # Does not train on annotations

        # Learn Isolation Forest
        settings.logger.info("Learning Isolation Forest")

        self.ifc = IsolationForestClassifier(
            n_estimators=self.settings["n_estimators"],
            max_samples=self.settings["max_samples"],
            contamination=self.settings["contamination"],
            max_features=self.settings["max_features"],
            bootstrap=self.settings["bootstrap"],
            random_state=self.settings["random_state"],
            warm_start=self.settings["warm_start"],
        )
        self.ifc.fit(events)

    def new_state_msg(self, msg):
        state = super().new_state_msg(msg)
        if state is None:
            return False, None

        # Returns -1 for outliers and 1 for inliers.
        alert = bool(self.ifc.predict([state])[0] == -1)
        return alert, 1 if alert else 0

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
            "classifier": self.ifc,
        }

        joblib.dump(model, self._resolve_model_file_path(), compress=3)

        return True

    def load_trained_model(self):
        if self.settings["model-file"] is None:
            return False

        try:  # Open model file
            model = joblib.load(self._resolve_model_file_path())
        except FileNotFoundError:
            settings.logger.info(
                "Model file {} not found.".format(str(self._resolve_model_file_path()))
            )
            return False

        # Load model
        assert self._name == model["_name"]
        super().load_trained_model(model["preprocessors"])
        self.settings = model["settings"]
        self.ifc = model["classifier"]

        return True

    def visualize_model(self, max_depth=3):
        import matplotlib.pyplot as plt
        from sklearn.tree import plot_tree

        fig, axs = plt.subplots(nrows=(len(self.ifc.estimators_) + 1) // 2, ncols=2)
        axs = [ax for row in axs for ax in row]

        for ax in axs:
            ax.axis("off")

        for i in range(len(self.ifc.estimators_)):
            settings.logger.info(
                "Plotting tree {}/{}".format(i + 1, len(self.ifc.estimators_))
            )

            plot_tree(
                self.ifc.estimators_[i],
                max_depth=max_depth,
                filled=True,
                impurity=True,
                rounded=True,
                ax=axs[i],
            )

        return plt, fig
