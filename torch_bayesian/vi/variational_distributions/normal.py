from math import log

import torch
from torch import Tensor

from torch_bayesian.vi import _globals

from .base import VariationalDistribution


class MeanFieldNormalVarDist(VariationalDistribution):
    """Variational distribution with uncorrelated, normal distributions."""

    def __init__(self, initial_std: float = 1.0) -> None:
        super().__init__()
        self.variational_parameters = ("mean", "log_std")
        self._default_variational_parameters = (0.0, log(initial_std))

    def sample(self, mean: Tensor, log_std: Tensor) -> Tensor:
        """Sample from a Gaussian distribution."""
        std = torch.exp(log_std)
        return self._normal_sample(mean, std)

    def log_prob(self, sample: Tensor, mean: Tensor, log_std: Tensor) -> Tensor:
        """Compute the log probability of sample based on a normal distribution."""
        variance = torch.exp(log_std) ** 2
        data_fitting = (sample - mean) ** 2 / variance
        normalization = 2 * log_std
        if _globals._USE_NORM_CONSTANTS:
            normalization = normalization + log(2 * torch.pi)
        return -0.5 * (data_fitting + normalization)

    @staticmethod
    def _normal_sample(mean: Tensor, std: Tensor) -> Tensor:
        base_sample = torch.normal(torch.zeros_like(mean), torch.ones_like(mean))
        sample = std * base_sample + mean
        return sample
