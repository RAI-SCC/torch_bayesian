"""This module provides basic layers and loss functions for BNN-training with Variational Inference."""

from .base import VIBaseModule, VIModule
from .kl_loss import KullbackLeiblerLoss
from .linear import VILinear
from .sequential import VISequential

__all__ = [
    "VIModule",
    "VIBaseModule",
    "VILinear",
    "VISequential",
    "KullbackLeiblerLoss",
]
