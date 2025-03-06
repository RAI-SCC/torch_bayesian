"""Offers a collection of predictive distributions."""

from .base import PredictiveDistribution
from .categorical import CategoricalPredictiveDistribution
from .normal import MeanFieldNormalPredictiveDistribution
from .skewnormal import SkewNormalPredictiveDistribution
from .student_t_fixed_dof import StudentTwithDOFPredictiveDistribution
from .mixturegaussian_equalstd import MixtureGaussianEqualStdPredictiveDistribution
from .student_t import StudentTPredictiveDistribution
from .mixturegaussian import MixtureGaussianPredictiveDistribution

__all__ = [
    "PredictiveDistribution",
    "MeanFieldNormalPredictiveDistribution",
    "CategoricalPredictiveDistribution",
    "SkewNormalPredictiveDistribution",
    "StudentTwithDOFPredictiveDistribution",
    'MixtureGaussianEqualStdPredictiveDistribution',
    'StudentTPredictiveDistribution',
    'MixtureGaussianPredictiveDistribution'
]
