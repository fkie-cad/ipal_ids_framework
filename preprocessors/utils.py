from .aggregate import AggregatePreprocessor
from .categorical import CategoricalPreprocessor
from .gradient import GradientPreprocessor
from .indicatenone import IndicateNonePreprocessor
from .labelencoder import LabelEncoderPreprocessor
from .mean import MeanPreprocessor
from .minmax import MinMaxPreprocessor
from .pca import PCAPreprocessor

preprocessors = [
    AggregatePreprocessor,
    CategoricalPreprocessor,
    GradientPreprocessor,
    IndicateNonePreprocessor,
    LabelEncoderPreprocessor,
    MeanPreprocessor,
    MinMaxPreprocessor,
    PCAPreprocessor,
]


def get_all_preprocessors():
    return {preprocessor._name: preprocessor for preprocessor in preprocessors}
