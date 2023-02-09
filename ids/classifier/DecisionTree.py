import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

import ipal_iids.settings as settings
from ids.featureids import FeatureIDS


class DecisionTree(FeatureIDS):
    _name = "DecisionTree"
    _description = "Decision tree classifier."
    _decisiontree_default_settings = {
        # DecisionTree GridSearch Parameters
        # TODO Better sample random values?
        "criterion": ["gini", "entropy"],
        "splitter": ["best", "random"],
        "max_depth": [None],
        "min_samples_split": [2],
        "min_samples_leaf": [1],
        "min_weight_fraction_leaf": [0],
        "max_features": ["sqrt", "log2", None],
        "max_leaf_nodes": [None],
        "min_impurity_decrease": [0.0],
        "random_state": [None],
        "class_weight": ["balanced", "balanced_subsample", None],
        "ccp_alpha": [0.0],
        # GridSearch Options
        "scoring": None,  # accuracy ..
        "jobs": 4,
        "verbose": 10,
        "no-probability": False,
    }

    def __init__(self, name=None):
        super().__init__(name=name)
        self._add_default_settings(self._decisiontree_default_settings)

        self.dtc = None
        self.classes = None

    # the IDS is given the path to file(s) containing its requested training data
    def train(self, ipal=None, state=None):
        if ipal and state:
            settings.logger.error("Only state or message supported")
            exit(1)

        if state is None:
            state = ipal

        events, annotation, _ = super().train(state=state)
        annotation = [a is not False for a in annotation]

        if len(set(annotation)) <= 1:
            settings.logger.warning(
                "Training with a single class ({}) only!".format(set(annotation))
            )

        # Learn DecisionTree
        settings.logger.info("Learning DecisionTree")
        tuned_parameters = {
            "criterion": self.settings["criterion"],
            "splitter": self.settings["splitter"],
            "max_depth": self.settings["max_depth"],
            "min_samples_split": self.settings["min_samples_split"],
            "min_samples_leaf": self.settings["min_samples_leaf"],
            "min_weight_fraction_leaf": self.settings["min_weight_fraction_leaf"],
            "max_features": self.settings["max_features"],
            "max_leaf_nodes": self.settings["max_leaf_nodes"],
            "min_impurity_decrease": self.settings["min_impurity_decrease"],
            "random_state": self.settings["random_state"],
            "class_weight": self.settings["class_weight"],
            "ccp_alpha": self.settings["ccp_alpha"],
        }

        settings.logger.info(tuned_parameters)
        dtc = GridSearchCV(
            DecisionTreeClassifier(),
            [tuned_parameters],
            scoring=self.settings["scoring"],
            n_jobs=self.settings["jobs"],
            verbose=self.settings["verbose"],
        )

        dtc.fit(events, annotation)

        settings.logger.info("Best parameters set found on development set:")
        settings.logger.info(dtc.best_params_)
        settings.logger.info("Grid scores on development set:")
        means = dtc.cv_results_["mean_test_score"]
        stds = dtc.cv_results_["std_test_score"]
        for mean, std, params in zip(means, stds, dtc.cv_results_["params"]):
            settings.logger.info("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

        # Save best estimator
        self.dtc = dtc.best_estimator_
        self.classes = list(self.dtc.classes_)

    def new_state_msg(self, msg):
        state = super().new_state_msg(msg)
        if state is None:
            return False, None

        alert = bool(self.dtc.predict([state])[0])

        if self.settings["no-probability"]:  # takes less time
            return alert, 1 if alert else 0

        else:
            probability = self.dtc.predict_proba([state])[0][self.classes.index(True)]
            return alert, probability

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
            "classifier": self.dtc,
            "classes": self.classes,
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
        self.dtc = model["classifier"]
        self.classes = model["classes"]

        return True

    def visualize_model(self, max_depth=3):
        import matplotlib.pyplot as plt
        from sklearn.tree import plot_tree

        fig, ax = plt.subplots(nrows=1, ncols=1)

        ax.axis("off")

        settings.logger.info("Plotting tree.")

        plot_tree(
            self.dtc,
            max_depth=max_depth,
            filled=True,
            impurity=True,
            rounded=True,
            ax=ax,
        )

        return plt, fig
