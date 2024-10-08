import orjson
from ar import arsel

import ipal_iids.settings as settings
from ids.ids import MetaIDS

"""
Deprecation Note:
This IDS bases on the 'ar' library from https://github.com/RhysU/ar.git
Since numpy version 2 the installation of this library fails. Therefore, this
IDS is currently marked as deprecated.
"""


class Autoregression(MetaIDS):
    _name = "Autoregression"
    _description = "Autoregression and CUSUM"
    _requires = ["train.state", "live.state"]
    _ar_default_settings = {
        "sensor": None,
        "firstN": None,
        "subtractMean": False,
        "absrho": True,
        "criterion": "BIC",
        "minorder": 1,
        "maxorder": 500,
        "eval": False,
    }
    _supports_preprocessor = False

    def __init__(self, name=None):
        super().__init__(name=name)
        self._add_default_settings(self._ar_default_settings)

        self.model = None
        self.delta = None
        self.previous = []
        self.cusum = 0

    def _calc_residual(self, values, coefficients):
        return sum([val * coeff for val, coeff in zip(values, coefficients[::-1])])

    def train(self, ipal=None, state=None):
        training_data = []

        # Load training data for each sensor
        with self._open_file(state) as f:
            for line in f:
                cur_state = orjson.loads(line)

                if self.settings["sensor"] in cur_state["state"]:
                    training_data.append(cur_state["state"][self.settings["sensor"]])
                else:
                    settings.logger.info(
                        f"Sensor {self.settings['sensor']} not in current state."
                    )

        if self.settings["firstN"] is None:
            settings.logger.info("Setting firstN for default 80%")
            self.settings["firstN"] = int(len(training_data) * 0.8)

        # Train autoregression model
        # For example, given a sequence or *row-vector* of samples 'd', one can fit a process, obtain its poles, and simulate a realization of length M using
        # Usage: M = arsel (data, submean, absrho, criterion, minorder, maxorder)
        self.model = arsel(
            training_data[: self.settings["firstN"]],
            self.settings["subtractMean"],
            self.settings["absrho"],
            self.settings["criterion"],
            self.settings["minorder"],
            self.settings["maxorder"],
        )

        settings.logger.info(
            f"Autoregression model for sensor {self.settings['sensor']} is {self.model}"
        )

        # Find threshold delta

        self.delta = 0

        for v in training_data[self.settings["firstN"] :]:
            self.previous.append(v)
            if self.model.submean:
                self.previous[-1] -= self.model.mu[0]

            if len(self.previous) < len(self.model.AR[0]):
                continue
            elif len(self.previous) > len(self.model.AR[0]):
                self.previous.pop(0)

            assert len(self.previous) == len(self.model.AR[0])

            rk = self._calc_residual(self.previous, self.model.AR[0])

            # self.delta = max(self.delta, abs(rk))
            self.delta += abs(rk)

        self.delta /= len(training_data) - self.settings["firstN"]
        settings.logger.info(
            f"Delta for sensor '{self.settings['sensor']}' is {self.delta}"
        )

        # Reset values
        self.previous = []
        self.cusum = 0

    def new_state_msg(self, msg):
        if self.settings["sensor"] not in msg["state"]:  # Sensor not available
            return False, 0
        value = msg["state"][self.settings["sensor"]]

        if self.model.submean:
            value -= self.model.mu[0]
        self.previous.append(value)

        if len(self.previous) < len(self.model.AR[0]):
            return False, 0
        elif len(self.previous) > len(self.model.AR[0]):
            self.previous.pop(0)  # Remove oldest item

        assert len(self.previous) == len(self.model.AR[0])

        rk = self._calc_residual(self.previous, self.model.AR[0])
        self.cusum = max(0, self.cusum + abs(rk) - self.delta)

        if self.settings["eval"]:
            settings.logger.info(f"{abs(rk)},{self.cusum}")

        # TODO decide on anomaly
        return None, self.cusum
