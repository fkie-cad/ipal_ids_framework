from .interarrivaltime.Mean import InterArrivalTimeMean
from .interarrivaltime.Range import InterArrivalTimeRange
from .classifier.BLSTM import BLSTM
from .classifier.RandomForest import RandomForest
from .classifier.SVM import SVM

idss = [
    BLSTM,
    InterArrivalTimeMean,
    InterArrivalTimeRange,
    RandomForest,
    SVM,
]


def get_all_iidss():
    return {ids._name: ids for ids in idss}
