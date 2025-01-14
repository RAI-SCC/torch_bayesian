"""Offers a collection of variational distributions."""

from .base import VariationalDistribution
from .non_bayesian import NonBayesian
from .normal import MeanFieldNormalVarDist

VarDist = VariationalDistribution

__all__ = [
    "MeanFieldNormalVarDist",
    "NonBayesian",
    "VarDist",
    "VariationalDistribution",
]
