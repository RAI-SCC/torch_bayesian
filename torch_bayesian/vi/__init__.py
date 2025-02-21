"""This module provides basic layers and loss functions for BNN-training with Variational Inference."""

from .base import VIBaseModule, VIModule
from .conv import VIConv1d, VIConv2d, VIConv3d
from .kl_loss import KullbackLeiblerLoss
from .linear import VILinear
from .sequential import VIResidualConnection, VISequential
from .mse_loss import MeanSquaredErrorLoss
from .transformer import (
    VIMultiheadAttention,
    VITransformer,
    VITransformerDecoder,
    VITransformerDecoderLayer,
    VITransformerEncoder,
    VITransformerEncoderLayer,
)

__all__ = [
    "KullbackLeiblerLoss",
    "VIBaseModule",
    "VIConv1d",
    "VIConv2d",
    "VIConv3d",
    "VILinear",
    "VIModule",
    "VIMultiheadAttention",
    "VIResidualConnection",
    "VISequential",
    "MeanSquaredErrorLoss",
    "VITransformer",
    "VITransformerDecoder",
    "VITransformerDecoderLayer",
    "VITransformerEncoder",
    "VITransformerEncoderLayer",
]
