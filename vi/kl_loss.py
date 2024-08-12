from typing import Optional
from warnings import warn

from torch import Tensor
from torch.nn import Module

from .predictive_distributions import PredictiveDistribution


class KullbackLeiblerLoss(Module):
    """
    Kullback-Leibler divergence loss.

    Requires external calculation of prior and variational log probability, i.e. modules must have return_log_prob = True
    """

    def __init__(
        self,
        predictive_distribution: PredictiveDistribution,
        dataset_size: Optional[int] = None,
        heat: float = 1.0,
    ) -> None:
        super().__init__()
        self.predictive_distribution = predictive_distribution
        self.dataset_size = dataset_size
        self.heat = heat

    def forward(
        self,
        samples: Tensor,
        prior_log_prob: Tensor,
        variational_log_prob: Tensor,
        target: Tensor,
        dataset_size: Optional[int] = None,
    ) -> Tensor:
        """
        Forward pass.

        Accepts a Tensor of N samples, the associate log probabilities and a target to calculate the loss.

        Parameters
        ----------
        samples: Tensor
            Sample Tensor. Shape (N, *)
        prior_log_prob: Tensor
            Prior log probability. Shape (N,)
        variational_log_prob: Tensor,
            Variational log probability. Shape (N,)
        target: Tensor,
            Target prediction. Shape (*)
        dataset_size: Optional[int] = None
            Total number of samples in the dataset
        """
        prior_matching = (variational_log_prob - prior_log_prob).mean()
        data_fitting = self.predictive_distribution.log_prob_from_samples(
            target, samples
        )

        if (dataset_size is None) and (self.dataset_size is None):
            warn(
                f"No dataset_size is provided. Number of samples ({samples.shape[0]}) is used instead."
            )
            n_data = samples.shape[0]
        else:
            n_data = dataset_size or self.dataset_size

        return -n_data * data_fitting + self.heat * prior_matching
