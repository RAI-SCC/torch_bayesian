from math import log

import torch
from torch import Tensor
from torch.distributions import StudentT

from torch_bayesian.vi import _globals

from .base import VariationalDistribution


class StudentTVarDist(VariationalDistribution):
    """Variational distribution with uncorrelated Student's t-distributions."""

    def __init__(
        self, initial_scale: float = 1.0, degrees_of_freedom: float = 4.0
    ) -> None:
        super().__init__()
        self.degrees_of_freedom = degrees_of_freedom
        self.variational_parameters = ("mean", "log_scale")
        self._default_variational_parameters = (0.0, log(initial_scale))

    def sample(self, mean: Tensor, log_scale: Tensor) -> Tensor:
        """Sample from a Student's t-distribution."""
        scale = torch.exp(log_scale)
        return self._student_t_sample(mean, scale, self.degrees_of_freedom)

    def log_prob(self, sample: Tensor, mean: Tensor, log_scale: Tensor) -> Tensor:
        """Compute the log probability of sample based on a Student's t-distribution."""
        scale = torch.exp(log_scale)
        data_fitting = (
            (self.degrees_of_freedom + 1.0)
            * 0.5
            * torch.log(1.0 + ((sample - mean) / scale) ** 2 / self.degrees_of_freedom)
        )
        normalization = log_scale
        if _globals._USE_NORM_CONSTANTS:
            normalization = (
                normalization
                + 0.5 * log(torch.pi * self.degrees_of_freedom)
                + torch.lgamma(torch.tensor(self.degrees_of_freedom / 2.0))
                - torch.lgamma(torch.tensor((self.degrees_of_freedom + 1.0) / 2.0))
            )
        return -(data_fitting + normalization)

    @staticmethod
    def _student_t_sample(
        mean: Tensor, scale: Tensor, degrees_of_freedom: float
    ) -> Tensor:
        base_distribution = StudentT(degrees_of_freedom * torch.ones_like(mean))
        base_sample = base_distribution.sample()
        sample = scale * base_sample + mean
        return sample
