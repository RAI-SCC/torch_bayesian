import torch
from torch import Tensor

from .base import VariationalDistribution


class NonBayesian(VariationalDistribution):
    """Makes the model deterministic, i.e. a classical neural network."""

    def __init__(self) -> None:
        super().__init__()
        self.variational_parameters = ("mean",)
        self._default_variational_parameters = (0.0,)

    def sample(self, mean: Tensor) -> Tensor:
        """
        Sample the distribution.

        Dummy sample that returns mean.
        """
        return mean

    def log_prob(self, sample: Tensor, mean: Tensor) -> Tensor:
        """
        Calculate log_prob.

        Dummy log_prob that returns 0.
        """
        return torch.zeros_like(sample)
