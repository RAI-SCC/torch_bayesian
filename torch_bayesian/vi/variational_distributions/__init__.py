"""Offers a collection of variational distributions."""

from .base import VariationalDistribution
from .normal import MeanFieldNormalVarDist

VarDist = VariationalDistribution

__all__ = ["VariationalDistribution", "MeanFieldNormalVarDist", "VarDist"]
