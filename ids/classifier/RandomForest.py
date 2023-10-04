import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import ipal_iids.settings as settings
from ids.featureids import FeatureIDS


class RandomForest(FeatureIDS):
    _name = "RandomForest"
    _description = "Random forest classifier."
    _randomforest_default_settings = {
        # RandomForest GridSearch Parameters
        # TODO Better sample random values?
        "n_estimators": [1, 10, 100, 1000],
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 10, 100, 1000],
        "min_samples_split": [2],
        "min_samples_leaf": [1],
        "min_weight_fraction_leaf": [0.0],
        "max_features": ["sqrt", "log2"],
        "max_leaf_nodes": [None],
        "min_impurity_decrease": [0.0],
        "bootstrap": [True, False],
        "oob_score": [True, False],
        "random_state": [None],
        "warm_start": [True, False],
        "class_weight": ["balanced", "balanced_subsample", None],
        "ccp_alpha": [0.0],
        "max_samples": [None],
        # GridSearch Options
        "scoring": None,  # accuracy ..
        "jobs": 4,
        "verbose": 10,
        "no-probability": False,
    }

    def __init__(self, name=None):
        super().__init__(name=name)
        self._add_default_settings(self._randomforest_default_settings)

        self.rfc = None
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

        # Learn Random Forest
        settings.logger.info("Learning Random Forest")
        tuned_parameters = {
            "n_estimators": self.settings["n_estimators"],
            "criterion": self.settings["criterion"],
            "max_depth": self.settings["max_depth"],
            "min_samples_split": self.settings["min_samples_split"],
            "min_samples_leaf": self.settings["min_samples_leaf"],
            "min_weight_fraction_leaf": self.settings["min_weight_fraction_leaf"],
            "max_features": self.settings["max_features"],
            "max_leaf_nodes": self.settings["max_leaf_nodes"],
            "min_impurity_decrease": self.settings["min_impurity_decrease"],
            "bootstrap": self.settings["bootstrap"],
            "oob_score": self.settings["oob_score"],
            "random_state": self.settings["random_state"],
            "warm_start": self.settings["warm_start"],
            "class_weight": self.settings["class_weight"],
            "ccp_alpha": self.settings["ccp_alpha"],
            "max_samples": self.settings["max_samples"],
        }
        settings.logger.info(tuned_parameters)

        # Test if gridsearch is neccesary
        if max([len(v) for v in tuned_parameters.values()]) > 1:
            settings.logger.info("Finding best parameteres with GirdSearchCV")
            rfc = GridSearchCV(
                RandomForestClassifier(),
                [tuned_parameters],
                scoring=self.settings["scoring"],
                n_jobs=self.settings["jobs"],
                verbose=self.settings["verbose"],
            )

            rfc.fit(events, annotation)

            settings.logger.info("Best parameters set found on development set:")
            settings.logger.info(rfc.best_params_)
            settings.logger.info("Grid scores on development set:")
            means = rfc.cv_results_["mean_test_score"]
            stds = rfc.cv_results_["std_test_score"]
            for mean, std, params in zip(means, stds, rfc.cv_results_["params"]):
                settings.logger.info(
                    "%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params)
                )

            # Save best estimator
            self.rfc = rfc.best_estimator_

        else:
            tuned_parameters = {k: v[0] for k, v in tuned_parameters.items()}
            self.rfc = RandomForestClassifier(**tuned_parameters)
            self.rfc.fit(events, annotation)

        self.classes = list(self.rfc.classes_)

    def new_state_msg(self, msg):
        state = super().new_state_msg(msg)
        if state is None:
            return False, None

        alert = bool(self.rfc.predict([state])[0])

        if self.settings["no-probability"]:  # takes less time
            return alert, 1 if alert else 0

        else:
            probability = self.rfc.predict_proba([state])[0][self.classes.index(True)]
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
            "classifier": self.rfc,
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
        self.rfc = model["classifier"]
        self.classes = model["classes"]

        return True

    def visualize_model(self, max_depth=3):
        import matplotlib.pyplot as plt
        from sklearn.tree import plot_tree

        fig, axs = plt.subplots(nrows=(len(self.rfc.estimators_) + 1) // 2, ncols=2)
        axs = [ax for row in axs for ax in row]

        for ax in axs:
            ax.axis("off")

        for i in range(len(self.rfc.estimators_)):
            settings.logger.info(
                "Plotting tree {}/{}".format(i + 1, len(self.rfc.estimators_))
            )

            plot_tree(
                self.rfc.estimators_[i],
                max_depth=max_depth,
                filled=True,
                impurity=True,
                rounded=True,
                ax=axs[i],
            )

        return plt, fig
