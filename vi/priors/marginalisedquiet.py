from math import exp, log
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import init

from .base import Prior

if TYPE_CHECKING:
    from ..base import VIBaseModule  # pragma: no cover


class MarginalisedQuietPrior(Prior):
    """Prior assuming uncorrelated parameters distributed according to the marginalised quiet prior from the paper."""

    def __init__(self, a: float = 1.0, rho: float = 1.0, samples: int = 10) -> None:
        super().__init__()
        self.distribution_parameters = ("log_a", "log_rho")
        self.log_a = log(a)
        self.log_rho = log(rho)
        self.samples = samples

    @property
    def a(self) -> float:
        """Prior std scaling parameter."""
        return exp(self.log_a)

    @property
    def rho(self) -> float:
        """Prior mean std parameter."""
        return exp(self.log_rho)

    def log_prob(self, sample: Tensor) -> Tensor:
        """Compute the log probability of sample based on the prior."""
        device = sample.device
        prob_sum = torch.zeros_like(sample).to(device)
        # mu_dist = torch.distributions.Normal(Tensor([0.0]), Tensor([self.rho]))
        for i in range(self.samples):
            # this_mu = mu_dist.sample(sample.shape)
            this_mu = self.rho * torch.normal(
                torch.zeros_like(sample), torch.ones_like(sample)
            )
            this_mu = this_mu.to(device)
            # print(sample.shape, this_mu.shape)
            variance = (self.a * this_mu) ** 2
            this_prob = torch.exp(
                -((sample - this_mu) ** 2) / (2 * variance)
            ) / torch.sqrt(2 * torch.pi * variance)

            prob_sum += this_prob

        prob = prob_sum / self.samples

        return torch.log(prob)

    def reset_parameters(self, module: "VIBaseModule", variable: str) -> None:
        """Reset the parameters of the module to prior mean and standard deviation."""
        log_a_name = module.variational_parameter_name(variable, "log_a")
        init.constant_(getattr(module, log_a_name), self.log_a)
        log_rho_name = module.variational_parameter_name(variable, "log_rho")
        init.constant_(getattr(module, log_rho_name), self.log_rho)
