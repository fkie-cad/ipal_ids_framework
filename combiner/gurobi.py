import json
import gurobipy
from math import inf

import ipal_iids.settings as settings
from .combiner import Combiner


class GurobiCombiner(Combiner):

    _name = "Gurobi"
    _description = (
        "Solves an optimization problem with Gurobi to find optimal weights for IDSs."
    )
    _requires_training = True
    _gurobi_default_settings = {
        "use_scores": False,
        "time-limit": None,
        "threads-limit": None,
    }

    def __init__(self):
        super().__init__()
        self._add_default_settings(self._gurobi_default_settings)

        self.weights = None
        self.bias = None
        self.keys = None
        self.threshold = 1.5
        self.objective = None

    def _get_activations(self, alerts, scores):
        data = scores if self.settings["use_scores"] else alerts

        if not set(data.keys()) == set(self.keys):
            settings.logger.error("Keys of combiner do not match data")
            settings.logger.error("- data keys: {}".format(",".join(data.keys())))
            settings.logger.error("- combiner keys: {}".format(",".join(self.keys)))
            exit(1)

        return {ids: float(data[ids]) for ids in self.keys}

    def _gurobi(self, data, annotations):
        # supress stdout logging
        env = gurobipy.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()

        # Create the optimization model
        m = gurobipy.Model(self._name, env=env)

        if self.settings["time-limit"] is not None:
            m.setParam("TimeLimit", self.settings["time-limit"])
        if self.settings["threads-limit"] is not None:
            m.setParam("Threads", self.settings["threads-limit"])

        # Add a weight variable for each ids
        weight_vars = [m.addVar(name=f"w_{ids_name}") for ids_name in self.keys]

        # Add a general bias var
        bias_var = m.addVar(name="bias", lb=-inf)

        # Add a slack variable for each message
        slack_vars = [
            m.addVar(name=f"slack_{i}", vtype=gurobipy.GRB.BINARY)
            for i in range(len(data))
        ]

        # The objective is to minimize the slack
        m.setObjective(
            gurobipy.quicksum([var for var in slack_vars]),
            gurobipy.GRB.MINIMIZE,
        )

        # Add soft constraints for each message
        idx = -1
        for activation, malicious in zip(data, annotations):
            idx += 1

            s = (
                gurobipy.quicksum(
                    weight * act
                    for weight, act in zip(activation.values(), weight_vars)
                )
                + bias_var
            )

            if malicious:  # 1.5 (settings["threshold"] is in bettween of 1 and 2)
                m.addConstr(s + 10 * slack_vars[idx] >= 2)
            else:
                # We cannot use strict inequality as gurobi does not support it
                m.addConstr(s - 10 * slack_vars[idx] <= 1)

        settings.logger.info("Starting model optimization...")

        def callback(model, where):
            if where == gurobipy.GRB.Callback.MIPSOL:

                # Save and log intermediate model
                self.weights = {
                    k: v for k, v in zip(self.keys, model.cbGetSolution([*weight_vars]))
                }
                self.bias = model.cbGetSolution([bias_var])
                self.objective = model.cbGet(gurobipy.GRB.Callback.MIPSOL_OBJ)

                self.save_trained_model()

                settings.logger.info("Found intermediate model:")
                settings.logger.info(f"- Weights: {self.weights}")
                settings.logger.info(f"- Bias: {self.bias}")
                settings.logger.info(
                    f"- Objective: {model.cbGet(gurobipy.GRB.Callback.MIPSOL_OBJ)}"
                )

        m.optimize(callback)
        settings.logger.info(f"Optimization done, objective value: {m.objVal}")

        self.weights = {
            ids_name: weight_var.x
            for ids_name, weight_var in zip(self.keys, weight_vars)
        }
        self.bias = bias_var.x
        self.objective = m.objVal

        settings.logger.info("Found final model:")
        settings.logger.info(f"- Weights: {self.weights}")
        settings.logger.info(f"- Bias: {self.bias}")
        settings.logger.info(f"- Objective: {m.objVal}")

    def train(self, file):
        data = []
        annotations = []

        settings.logger.info("Loading combiner training file")
        with self._open_file(file, "r") as f:
            for line in f.readlines():
                js = json.loads(line)

                if self.keys is None:
                    self.keys = sorted(js["scores"].keys())

                data.append(self._get_activations(js["alerts"], js["scores"]))
                annotations.append(js["malicious"] is not False)

        settings.logger.info("Fitting Gurobi Combiner")
        self._gurobi(data, annotations)

    def combine(self, alerts, scores):
        data = self._get_activations(alerts, scores)
        sums = sum([data[name] * self.weights[name] for name in self.weights])

        return sums > self.threshold, sums / self.threshold

    def save_trained_model(self):

        if self.settings["model-file"] is None:
            return False

        model = {
            "_name": self._name,
            "settings": self.settings,
            "keys": self.keys,
            "weights": self.weights,
            "bias": self.bias,
            "threshold": self.threshold,
            "objective": self.objective,
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
        self.keys = model["keys"]
        self.weights = model["weights"]
        self.bias = model["bias"]
        self.threshold = model["threshold"]
        self.objective = model["objective"]

        return True
