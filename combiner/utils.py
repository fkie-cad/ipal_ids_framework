from .gurobi import GurobiCombiner
from .heuristic import HeuristicCombiner
from .logisticregression import LogisticRegressionCombiner
from .svm import SVMCombiner
from .votes import AllCombiner
from .votes import AnyCombiner
from .votes import MajorityCombiner
from .votes import WeightsCombiner

combiners = [
    AllCombiner,
    AnyCombiner,
    GurobiCombiner,
    HeuristicCombiner,
    LogisticRegressionCombiner,
    MajorityCombiner,
    SVMCombiner,
    WeightsCombiner,
]


def get_all_combiner():
    return {combiner._name: combiner for combiner in combiners}
