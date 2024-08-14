"""Offers a collection of predictive distributions."""

from .base import PredictiveDistribution
from .categorical import CategoricalPredictiveDistribution
from .normal import MeanFieldNormalPredictiveDistribution

__all__ = [
    "PredictiveDistribution",
    "MeanFieldNormalPredictiveDistribution",
    "CategoricalPredictiveDistribution",
]
