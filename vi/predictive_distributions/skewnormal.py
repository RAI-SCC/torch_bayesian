from typing import Tuple

import torch
from math import sqrt
from torch import Tensor
from torch.distributions import Normal
from .base import PredictiveDistribution


class SkewNormalPredictiveDistribution(PredictiveDistribution):
    """Predictive distribution assuming uncorrelated, normal distributed forecasts."""

    predictive_parameters = ("mean", "std", "skewness")

    @staticmethod
    def predictive_parameters_from_samples(samples: Tensor) -> Tuple[Tensor, Tensor]:
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
        #wskew = (skewness/(4-torch.pi)) **2
        #k_square = -1/(6+24*wskew) * (-12*wskew*torch.pi +6*wskew*torch.pi**2 + (-6*wskew*torch.pi**2)/(torch.pi*(2*wskew**2)**(1/3)))
        #delta = torch.sqrt(k_square)
        #delta_abs = torch.sqrt(torch.pi/2 * (torch.abs(skewness)**(2/3))/(torch.abs(skewness)**(2/3)+((4-torch.pi)/2)**(2/3)))
        skewness = skewness/torch.abs(skewness) * torch.min(torch.abs(skewness), 0.99*torch.ones_like(skewness))
        constant = torch.pi/2
        numerator = torch.abs(skewness)**(2/3)
        denominator = torch.abs(skewness)**(2/3) + ((4-torch.pi)/2)**(2/3)
        delta_abs = torch.sqrt(constant*numerator/denominator)
        delta = skewness/torch.abs(skewness) *delta_abs
        alpha = delta/torch.sqrt(1-delta**2)
        omega = torch.sqrt(variance*(torch.pi/(torch.pi-2*delta**2)))
        xi = mean - omega*delta*sqrt(2/torch.pi)

        constant = 2/omega
        dist = Normal(0, 1)
        logpdf = dist.log_prob((reference-xi)/(omega))
        cdf = dist.cdf(alpha*((reference-xi)/(omega))) + 1e-8

        return torch.log(constant)+logpdf+torch.log(cdf)
