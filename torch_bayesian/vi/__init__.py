"""This module provides basic layers and loss functions for BNN-training with Variational Inference."""

from .analytical_kl_loss import (
    AnalyticalKullbackLeiblerLoss,
    KullbackLeiblerModule,
    NonBayesian,
    NormalNormalDivergence,
    UniformNormalDivergence,
)
from .base import VIBaseModule, VIModule
from .conv import VIConv1d, VIConv2d, VIConv3d
from .kl_loss import KullbackLeiblerLoss
from .linear import VILinear
from .sequential import VIResidualConnection, VISequential
from .transformer import (
    VIMultiheadAttention,
    VITransformer,
    VITransformerDecoder,
    VITransformerDecoderLayer,
    VITransformerEncoder,
    VITransformerEncoderLayer,
)
from .utils.common_types import VIkwargs

__all__ = [
    "AnalyticalKullbackLeiblerLoss",
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
    "VITransformer",
    "VITransformerDecoder",
    "VITransformerDecoderLayer",
    "VITransformerEncoder",
    "VITransformerEncoderLayer",
    "KullbackLeiblerModule",
    "NonBayesian",
    "NormalNormalDivergence",
    "UniformNormalDivergence",
    "VIkwargs",
]
