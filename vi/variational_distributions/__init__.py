"""Offers a collection of variational distributions."""

from .base import VariationalDistribution
from .normal import MeanFieldNormalVarDist

__all__ = ["VariationalDistribution", "MeanFieldNormalVarDist"]
