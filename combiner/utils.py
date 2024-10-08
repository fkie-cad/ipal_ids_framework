import importlib.util
import os
import sys

# Dynamic loading

module_directory = os.path.dirname(os.path.abspath(__file__))

paths = {
    "Any": os.path.join(module_directory, "default.py"),
    "Matrix": os.path.join(module_directory, "matrix.py"),
    "Gurobi": os.path.join(module_directory, "gurobi.py"),
    "Heuristic": os.path.join(module_directory, "heuristic.py"),
    "LogisticRegression": os.path.join(module_directory, "logisticregression.py"),
    "MLP": os.path.join(module_directory, "mlp.py"),
    "SVM": os.path.join(module_directory, "svm.py"),
    "LSTM": os.path.join(module_directory, "lstm.py"),
}

combiner_to_use_name = None


def load_combiner(name):
    spec = importlib.util.spec_from_file_location(name, paths[name])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return {name: getattr(mod, f"{name}Combiner")}


def get_all_combiner():
    global combiner_to_use_name
    return load_combiner(combiner_to_use_name)
