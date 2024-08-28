"""Offers a collection of prior distributions."""

from .base import Prior
from .normal import MeanFieldNormalPrior
from .quiet import BasicQuietPrior

__all__ = ["Prior", "MeanFieldNormalPrior", "BasicQuietPrior"]
