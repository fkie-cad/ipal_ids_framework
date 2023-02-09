import joblib

from sklearn.naive_bayes import (
    GaussianNB,
    MultinomialNB,
    ComplementNB,
    BernoulliNB,
    CategoricalNB,
)

import ipal_iids.settings as settings
from ids.featureids import FeatureIDS


class NaiveBayes(FeatureIDS):
    _name = "NaiveBayes"
    _description = "Naive bayes classifier."
    _naivebayes_default_settings = {
        # Naive Bayes type
        "nb-classifier": "Gaussian",  # Gaussian, Multinomal, Complement, Bernoulli, Categorical
        "no-probability": False,
    }

    def __init__(self, name=None):
        super().__init__(name=name)
        self._add_default_settings(self._naivebayes_default_settings)

        self.nbc = None
        self.classes = None

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

        # Learning Naive Bayes
        settings.logger.info("Learning Naive Bayes")

        if self.settings["nb-classifier"] == "Gaussian":
            self.nbc = GaussianNB()
        elif self.settings["nb-classes"] == "Multinomial":
            self.nbc = MultinomialNB()
        elif self.setting["nb-classes"] == "Complement":
            self.nbc = ComplementNB()
        elif self.settings["nb-classes"] == "Bernoulli":
            self.nbc = BernoulliNB()
        elif self.settings["nb-classes"] == "Categorical":
            self.nbc = CategoricalNB()
        else:
            settings.logger.error(
                "Unknown Naive Bayes classifier {}. Falling back to Gaussian".format(
                    self.settings["nb-classifier"]
                )
            )
            self.nbc = GaussianNB()

        self.nbc.fit(events, annotation)
        self.classes = list(self.nbc.classes_)

    def new_state_msg(self, msg):
        state = super().new_state_msg(msg)
        if state is None:
            return False, False

        alert = bool(self.nbc.predict([state])[0])

        if self.settings["no-probability"]:  # takes less time
            return alert, 1 if alert else 0

        else:
            probability = self.nbc.predict_proba([state])[0][self.classes.index(True)]
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
            "classifier": self.nbc,
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
        self.nbc = model["classifier"]
        self.classes = model["classes"]

        return True
