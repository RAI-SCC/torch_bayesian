from math import log, sqrt
from typing import Tuple

import torch
from torch import Tensor
from matplotlib import pyplot as plt
from .base import PredictiveDistribution


class GammaPredictiveDistribution(PredictiveDistribution):
    """Predictive distribution assuming uncorrelated, normal distributed forecasts."""

    predictive_parameters = ("mean", "std", "skewness")

    @staticmethod
    def predictive_parameters_from_samples(samples: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Calculate mean and standard deviation of samples."""
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        skewness = torch.mean((samples - mean) ** 3) / (std ** (3 / 2))
        return mean, std, skewness

    @staticmethod
    def log_prob_from_parameters(
        reference: Tensor, parameters: Tuple[Tensor, Tensor]
    ) -> Tensor:
        """Calculate log probability of reference given mean and standard deviation."""
        mean, std, skewness = parameters
        variance = std**2

        shape = 4/(skewness**2)
        scale = torch.sqrt(variance/shape)
        location = mean - (shape*scale)

        data_fitting = (shape-1)*torch.log(reference-location) - ((reference-location)/scale)
        normalization = -torch.lgamma(shape) - (shape*torch.log(scale))
        print(reference-location)
        return normalization + data_fitting
