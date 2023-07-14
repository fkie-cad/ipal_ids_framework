from .default import AnyCombiner
from .gurobi import GurobiCombiner
from .heuristic import HeuristicCombiner
from .logisticregression import LogisticRegressionCombiner
from .lstm import LSTMCombiner
from .matrix import MatrixCombiner
from .mlp import MLPCombiner
from .svm import SVMCombiner

combiners = [
    # Default
    AnyCombiner,  # Remains for simplicity as default combiner
    # Unsupervised & Timeaware
    MatrixCombiner,  # AllCombiner TemporalCombiner MajorityCombiner WeightsCombiner
    # Supervised & Point-based
    GurobiCombiner,
    HeuristicCombiner,
    LogisticRegressionCombiner,
    MLPCombiner,
    SVMCombiner,
    # Supervised & Timeaware
    LSTMCombiner,
]


def get_all_combiner():
    return {combiner._name: combiner for combiner in combiners}
