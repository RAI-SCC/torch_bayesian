"""This module provides basic layers and loss functions for BNN-training with Variational Inference."""

from .base import VIBaseModule, VIModule
from .conv import VIConv1d, VIConv2d, VIConv3d
from .kl_loss import KullbackLeiblerLoss
from .linear import VILinear
from .sequential import VIResidualConnection, VISequential
from .mse_loss import MeanSquaredErrorLoss

__all__ = [
    "VIModule",
    "VIBaseModule",
    "VILinear",
    "VISequential",
    "KullbackLeiblerLoss",
    "VIConv1d",
    "VIConv2d",
    "VIConv3d",
    "VIResidualConnection",
    "MeanSquaredErrorLoss",
]
