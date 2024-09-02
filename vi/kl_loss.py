from typing import Optional
from warnings import warn

from torch import Tensor
from torch.nn import Module

from .predictive_distributions import PredictiveDistribution


class KullbackLeiblerLoss(Module):
    """
    Kullback-Leibler (KL) divergence loss.

    Calculates the Evidence Lower Bound (ELBO) loss which minimizes the KL-divergence
    between the variational distribution and the true posterior. Requires external
    calculation of prior and variational log probability, i.e. modules must have
    return_log_prob = True.

    Parameters
    ----------
    predictive_distribution: PredictiveDistribution
        Assumed distribution of the outputs. Typically, `CategoricalPredictiveDistribution`
        for classification and `MeanFieldNormalPredictiveDistribution` for regression.
    dataset_size: Optional[int]
        Size of the training dataset. Required for loss calculation. If not provided,
        it must be provided to the forward method.
    heat: float
        Temperature in the sense of the Cold Posterior effect. Default: 1.
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
        Calculate the negative ELBO loss from sampled evaluations, a target and the weight log probs.

        Accepts a Tensor of N samples, the associate log probabilities and a target to
        calculate the loss.

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
            Total number of samples in the dataset. Used in place of self.dataset_size
            if provided.

        Returns
        -------
        Tensor
            Negative ELBO loss. Shape: (1,)
        """
        prior_matching = (variational_log_prob - prior_log_prob).mean()
        # Sample average for predictive log prob is already done
        data_fitting = self.predictive_distribution.log_prob_from_samples(
            target, samples
        ).sum()

        if (dataset_size is None) and (self.dataset_size is None):
            warn(
                f"No dataset_size is provided. Number of samples ({samples.shape[0]}) is used instead."
            )
            n_data = samples.shape[0]
        else:
            n_data = dataset_size or self.dataset_size

        return -data_fitting + self.heat * prior_matching / n_data
