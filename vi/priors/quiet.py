from math import log
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import init

from ..utils import init as vi_init
from .base import Prior

if TYPE_CHECKING:
    from ..base import VIBaseModule  # pragma: no cover


class BasicQuietPrior(Prior):
    """Prior assuming normal distributed mean and std proportional to it."""

    def __init__(
        self,
        std_ratio: float = 1.0,
        mean_mean: float = 0.0,
        mean_std: float = 1.0,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.distribution_parameters = ("mean", "log_std")
        self._required_parameters = ("mean",)
        self._scaling_parameters = ("mean_mean", "mean_std", "eps")
        self._std_ratio = std_ratio
        self.mean_mean = mean_mean
        self.mean_std = mean_std
        self.eps = eps

    def log_prob(self, sample: Tensor, mean: Tensor) -> Tensor:
        """Compute the log probability of sample based on the prior."""
        variance = (self._std_ratio * mean) ** 2 + self.eps
        data_fitting = (sample - mean) ** 2 / variance
        mean_decay = (mean - self.mean_mean) ** 2 / (self.mean_std**2)
        normalization = variance.log() + 2 * log(self.mean_std) + 2 * log(2 * torch.pi)
        return -0.5 * (data_fitting + mean_decay + normalization)

    def reset_parameters(self, module: "VIBaseModule", variable: str) -> None:
        """Reset the parameters of the module to prior mean and standard deviation."""
        mean_name = module.variational_parameter_name(variable, "mean")
        init._no_grad_normal_(getattr(module, mean_name), self.mean_mean, self.mean_std)
        log_std_name = module.variational_parameter_name(variable, "log_std")
        log_std = torch.log(self._std_ratio * getattr(module, mean_name).abs())
        vi_init.fixed_(getattr(module, log_std_name), log_std)
