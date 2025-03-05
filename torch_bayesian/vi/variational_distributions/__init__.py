"""Offers a collection of variational distributions."""

from .base import VariationalDistribution
from .non_bayesian import NonBayesian
from .normal import MeanFieldNormalVarDist
from .student_t import StudentTVarDist

VarDist = VariationalDistribution

__all__ = [
    "MeanFieldNormalVarDist",
    "StudentTVarDist",
    "NonBayesian",
    "VarDist",
    "VariationalDistribution",
]
