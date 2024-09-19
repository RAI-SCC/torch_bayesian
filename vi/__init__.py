"""This module provides basic layers and loss functions for BNN-training with Variational Inference."""

from torch import nn

from .base import VIBaseModule, VIModule
from .conversion import convert_to_vi
from .kl_loss import KullbackLeiblerLoss
from .sequential import VIResidualConnection, VISequential

VILinear = convert_to_vi(nn.Linear)

VIConv1d = convert_to_vi(nn.Conv1d)
VIConv2d = convert_to_vi(nn.Conv2d)
VIConv3d = convert_to_vi(nn.Conv3d)

__all__ = [
    "VIModule",
    "VIBaseModule",
    "VISequential",
    "KullbackLeiblerLoss",
    "VIResidualConnection",
    "VILinear",
    "VIConv1d",
    "VIConv2d",
    "VIConv3d",
    "convert_to_vi",
]
