from math import exp, log
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import init

from .base import Prior

if TYPE_CHECKING:
    from ..base import VIBaseModule  # pragma: no cover


class MeanFieldNormalPrior(Prior):
    """Prior assuming uncorrelated, normal distributed parameters."""

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        super().__init__()
        self.distribution_parameters = ("mean", "log_std")
        self.mean = mean
        self.log_std = log(std)

    @property
    def std(self) -> float:
        """Prior standard deviation."""
        return exp(self.log_std)

    def log_prob(self, sample: Tensor) -> Tensor:
        """Compute the log probability of sample based on the prior."""
        variance = self.std**2
        data_fitting = (sample - self.mean) ** 2 / variance
        normalization = 2 * self.log_std + log(2 * torch.pi)
        return -0.5 * (data_fitting + normalization)

    def reset_parameters(self, module: "VIBaseModule", variable: str) -> None:
        """Reset the parameters of the module to prior mean and standard deviation."""
        mean_name = module.variational_parameter_name(variable, "mean")
        init.constant_(getattr(module, mean_name), self.mean)
        log_std_name = module.variational_parameter_name(variable, "log_std")
        init.constant_(getattr(module, log_std_name), self.log_std)
