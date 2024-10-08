import importlib.util
import os
import sys

module_directory = os.path.dirname(os.path.abspath(__file__))


preprocessor_paths = {
    "Aggregate": os.path.join(module_directory, "aggregate.py"),
    "Categorical": os.path.join(module_directory, "categorical.py"),
    "Gradient": os.path.join(module_directory, "gradient.py"),
    "IndicateNone": os.path.join(module_directory, "indicatenone.py"),
    "LabelEncoder": os.path.join(module_directory, "labelencoder.py"),
    "Mean": os.path.join(module_directory, "mean.py"),
    "MinMax": os.path.join(module_directory, "minmax.py"),
    "PCA": os.path.join(module_directory, "pca.py"),
}


def load_preprocessor(name):
    # Load preprocessor
    spec = importlib.util.spec_from_file_location(
        f"{name}Preprocessor", preprocessor_paths[name]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"{name}Preprocessor"] = mod
    spec.loader.exec_module(mod)
    # Return preprocessor
    return getattr(mod, f"{name}Preprocessor")
