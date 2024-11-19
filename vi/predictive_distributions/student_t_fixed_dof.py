from typing import Tuple
from math import log, sqrt

import torch
from torch import Tensor

from .base import PredictiveDistribution


class StudentTwithDOFPredictiveDistribution(PredictiveDistribution):
    """Predictive distribution assuming uncorrelated, normal distributed forecasts."""

    predictive_parameters = ("mean", "std")

    @staticmethod
    def predictive_parameters_from_samples(samples: Tensor) -> Tuple[Tensor, Tensor]:
        """Calculate mean and standard deviation of samples."""
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        return mean, std

    @staticmethod
    def log_prob_from_parameters(
        reference: Tensor, parameters: Tuple[Tensor, Tensor]
    ) -> Tensor:
        """Calculate log probability of reference given mean and standard deviation."""
        mean, std = parameters
        dof = 5.
        tau = std * sqrt((dof-2)/dof) +1e-5

        data_fitting = (dof+1) * torch.log(1 + (1/(dof*(tau**2))) * ((reference-mean)**2))
        #print(data_fitting)
        #normalization = torch.lgamma(0.5*(dof+1)) - torch.lgamma(0.5*dof) - 0.5*torch.log(torch.pi*dof*(tau**2))
        normalization = torch.log(torch.pi * dof * (tau ** 2))
        return - 0.5 *(data_fitting + normalization)
