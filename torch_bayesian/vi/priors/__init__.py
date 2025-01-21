"""Offers a collection of prior distributions."""

from .base import Prior
from .normal import MeanFieldNormalPrior
from .quiet import BasicQuietPrior
from .uniform import UniformPrior

NonBayesian = UniformPrior

__all__ = [
    "BasicQuietPrior",
    "MeanFieldNormalPrior",
    "NonBayesian",
    "Prior",
    "UniformPrior",
]
