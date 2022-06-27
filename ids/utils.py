from .autoregression.Autoregression import Autoregression
from .classifier.BLSTM import BLSTM
from .classifier.DecisionTree import DecisionTree
from .classifier.ExtraTrees import ExtraTrees
from .classifier.IsolationForest import IsolationForest
from .classifier.NaiveBayes import NaiveBayes
from .classifier.RandomForest import RandomForest
from .classifier.SVM import SVM
from .interarrivaltime.Mean import InterArrivalTimeMean
from .interarrivaltime.Range import InterArrivalTimeRange
from .oracles.OptimalIDS import OptimalIDS
from .oracles.DummyIDS import DummyIDS

idss = [
    Autoregression,
    BLSTM,
    DecisionTree,
    DummyIDS,
    ExtraTrees,
    InterArrivalTimeMean,
    InterArrivalTimeRange,
    IsolationForest,
    NaiveBayes,
    OptimalIDS,
    RandomForest,
    SVM,
]


def get_all_iidss():
    return {ids._name: ids for ids in idss}
