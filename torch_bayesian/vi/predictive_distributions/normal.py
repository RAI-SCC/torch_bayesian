from math import log
from typing import Tuple

import torch
from torch import Tensor

from torch_bayesian.vi import _globals

from .base import PredictiveDistribution


class MeanFieldNormalPredictiveDistribution(PredictiveDistribution):
    """Predictive distribution assuming uncorrelated, normal distributed forecasts."""

    predictive_parameters = ("mean", "std")

    @staticmethod
    def predictive_parameters_from_samples(samples: Tensor) -> Tuple[Tensor, Tensor]:
        """Calculate mean and standard deviation of samples."""
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        return mean, std

    @staticmethod
    def log_prob_from_parameters(
        reference: Tensor, parameters: Tuple[Tensor, Tensor]
    ) -> Tensor:
        """Calculate log probability of reference given mean and standard deviation."""
        mean, std = parameters
        variance = std**2
        data_fitting = (reference - mean) ** 2 / variance
        normalization = torch.log(variance)
        if _globals._USE_NORM_CONSTANTS:
            normalization = normalization + log(2 * torch.pi)
        return -0.5 * (data_fitting + normalization)
