"""This module provides basic layers and loss functions for BNN-training with Variational Inference."""

from . import predictive_distributions, priors, utils, variational_distributions
from .base import VIBaseModule, VIModule
from .conv import VIConv1d, VIConv2d, VIConv3d
from .kl_loss import KullbackLeiblerLoss
from .linear import VILinear
from .sequential import VIResidualConnection, VISequential

__all__ = [
    "predictive_distributions",
    "priors",
    "utils",
    "variational_distributions",
    "VIModule",
    "VIBaseModule",
    "VILinear",
    "VISequential",
    "KullbackLeiblerLoss",
    "VIConv1d",
    "VIConv2d",
    "VIConv3d",
    "VIResidualConnection",
]
