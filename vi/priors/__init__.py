"""Offers a collection of prior distributions."""

from .base import Prior
from .normal import MeanFieldNormalPrior

__all__ = ["Prior", "MeanFieldNormalPrior"]
