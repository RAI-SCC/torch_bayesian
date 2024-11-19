from typing import Tuple

import torch
from torch import Tensor

from .base import PredictiveDistribution


class StudentTPredictiveDistribution(PredictiveDistribution):
    """Predictive distribution assuming uncorrelated, normal distributed forecasts."""

    predictive_parameters = ("mean", "sigma", "dof")

    @staticmethod
    def predictive_parameters_from_samples(samples: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Calculate mean and standard deviation of samples."""
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        variance = std**2
        #kurtosis from Moments of Student’s T-Distribution: A Unified Approach
        kurtosis = torch.mean((samples - mean)**4)
        dof = (4*kurtosis - 6*variance**2) / (kurtosis - 3*variance**2)
        sigma = dof/(dof-2) * (1/variance)

        return mean, sigma, dof

    @staticmethod
    def log_prob_from_parameters(
        reference: Tensor, parameters: Tuple[Tensor, Tensor, Tensor]
    ) -> Tensor:
        """Calculate log probability of reference given mean and standard deviation."""
        mean, sigma, dof = parameters
        data_fitting = - 0.5*(dof+1) * torch.log(1 + (sigma/dof) * ((reference-mean)**2))
        normalization = torch.lgamma(0.5*(dof+1)) - torch.lgamma(0.5*dof) + 0.5 * torch.log(sigma) - 0.5*torch.log(torch.pi*dof)
        print(dof, torch.lgamma(0.5*(dof+1)), torch.lgamma(0.5*dof) ,torch.log(sigma))
        return data_fitting + normalization
