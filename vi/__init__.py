"""This module provides basic layers and loss functions for BNN-training with Variational Inference."""

from .base import VIBaseModule, VIModule
from .linear import VILinear

__all__ = ["VIModule", "VIBaseModule", "VILinear"]
