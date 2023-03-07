from .gurobi import GurobiCombiner
from .heuristic import HeuristicCombiner
from .logisticregression import LogisticRegressionCombiner
from .lstm import LSTMCombiner
from .svm import SVMCombiner
from .votes import AllCombiner, AnyCombiner, MajorityCombiner, WeightsCombiner

combiners = [
    AllCombiner,
    AnyCombiner,
    GurobiCombiner,  # RunningAverage
    HeuristicCombiner,
    LSTMCombiner,  # Time-aware
    LogisticRegressionCombiner,  # RunningAverage
    MajorityCombiner,  # RunningAverage
    SVMCombiner,  # RunningAverage
    WeightsCombiner,  # RunningAverage
]


def get_all_combiner():
    return {combiner._name: combiner for combiner in combiners}
