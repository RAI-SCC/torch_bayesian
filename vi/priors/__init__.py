"""Offers a collection of prior distributions."""

from .base import Prior
from .normal import MeanFieldNormalPrior
from .marginalisedquiet import MarginalisedQuietPrior

__all__ = ["Prior", "MeanFieldNormalPrior", "MarginalisedQuietPrior"]
