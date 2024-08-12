"""Offers a collection of predictive distributions."""

from .base import PredictiveDistribution
from .normal import MeanFieldNormalPredictiveDistribution

__all__ = ["PredictiveDistribution", "MeanFieldNormalPredictiveDistribution"]
