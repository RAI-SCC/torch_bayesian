import torch
from torch import Tensor

from .base import Prior


class UniformPrior(Prior):
    """Prior assuming uniformly distributed parameters."""

    distribution_parameters = ()

    @staticmethod
    def log_prob(sample: Tensor) -> Tensor:
        """Compute the log probability of sample based on the prior."""
        return torch.tensor([0.0], device=sample.device)
