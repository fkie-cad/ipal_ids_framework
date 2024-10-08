# from .autoregression.Autoregression import Autoregression # NOTE Deprecated Library

import importlib.util
import os
import sys
from collections import ChainMap

# Dynamic module loading

module_directory = os.path.dirname(os.path.abspath(__file__))

paths = {
    "BLSTM": os.path.join(module_directory, "classifier", "BLSTM.py"),
    "GeCo": os.path.join(module_directory, "geco", "geco.py"),
    "DecimalPlaces": os.path.join(module_directory, "simple", "decimal.py"),
    "DecisionTree": os.path.join(module_directory, "classifier", "DecisionTree.py"),
    "Dtmc": os.path.join(module_directory, "dtmc", "DTMC.py"),
    "DummyIDS": os.path.join(module_directory, "oracles", "DummyIDS.py"),
    "ExistsIDS": os.path.join(module_directory, "simple", "exists.py"),
    "ExtraTrees": os.path.join(module_directory, "classifier", "ExtraTrees.py"),
    "Histogram": os.path.join(module_directory, "simple", "histogram.py"),
    "InterArrivalTimeMean": os.path.join(
        module_directory, "interarrivaltime", "Mean.py"
    ),
    "InterArrivalTimeRange": os.path.join(
        module_directory, "interarrivaltime", "Range.py"
    ),
    "InvariantRules": os.path.join(
        module_directory, "invariant_rules", "invariant_rules_IDS.py"
    ),
    "IsolationForest": os.path.join(
        module_directory, "classifier", "IsolationForest.py"
    ),
    "Kitsune": os.path.join(module_directory, "kitsune", "kitsune.py"),
    "MinMax": os.path.join(module_directory, "simple", "minmax.py"),
    "NaiveBayes": os.path.join(module_directory, "classifier", "NaiveBayes.py"),
    "OptimalIDS": os.path.join(module_directory, "oracles", "OptimalIDS.py"),
    "Pasad": os.path.join(module_directory, "pasad", "Pasad.py"),
    "RandomForest": os.path.join(module_directory, "classifier", "RandomForest.py"),
    "SVM": os.path.join(module_directory, "classifier", "SVM.py"),
    "Seq2SeqNN": os.path.join(module_directory, "seq2seqnn", "Seq2SeqNN.py"),
    "SteadyTime": os.path.join(module_directory, "simple", "steadytime.py"),
    "TABOR": os.path.join(module_directory, "tabor", "TABOR.py"),
    "DummyBatchIDS": os.path.join(module_directory, "batch", "DummyBatchIDS.py"),
}

# Specifies which idss should be loaded
idss_to_load = []

# Contains already loaded modules
ids_cache = {}


def set_idss_to_load(arr):
    global idss_to_load
    idss_to_load = arr


def load_ids(name):
    global ids_cache
    # Check if module was already loaded
    if name in list(ids_cache.keys()):
        # If so return it
        return {name: ids_cache[name]}
    else:
        # Else load it
        spec = importlib.util.spec_from_file_location(name, paths[name])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        # And insert into cache
        ids_cache[name] = getattr(mod, name)
        # Then return
        return {name: getattr(mod, name)}


def get_all_iidss():
    return dict(ChainMap(*[load_ids(itl) for itl in idss_to_load]))
