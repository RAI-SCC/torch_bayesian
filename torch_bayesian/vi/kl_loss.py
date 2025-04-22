from typing import Dict, List, Optional
from warnings import warn

from torch import Tensor
from torch.nn import Module

from .predictive_distributions import PredictiveDistribution
from .utils.common_types import _log_prob_return_format


class KullbackLeiblerLoss(Module):
    r"""
    Kullback-Leibler (KL) divergence loss.

    Calculates the Evidence Lower Bound (ELBO) loss which minimizes the KL-divergence
    between the variational distribution and the true posterior. Requires external
    calculation of prior and variational log probability, i.e. modules must have
    return_log_probs = True.

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
    track: bool, default: False
        Set True to track the loss components. The log is stored as a dictionary in
        `self.log`. Loss history is stored for three components as lists accessible via
        the respective keys:

        - data_fitting: data log likelihood
        - prior_matching: the Kullback-Leibler divergence of prior anc variational
          distribution
        - log_probs: the raw prior and variational distribution log probabilities of
          the sampled weights.

    """

    def __init__(
        self,
        predictive_distribution: PredictiveDistribution,
        dataset_size: Optional[int] = None,
        heat: float = 1.0,
        track: bool = False,
    ) -> None:
        super().__init__()
        self.predictive_distribution = predictive_distribution
        self.dataset_size = dataset_size
        self.heat = heat
        self._track = track

        self.log: Optional[Dict[str, List[Tensor]]] = None
        if self._track:
            self._init_log()

    def track(self, mode: bool = True) -> None:
        """
        Enable or disable loss tracking.

        Any existing loss history is kept and continued if tracking is reenabled.

        Parameters
        ----------
        mode: bool
            If `True`, enable loss tracking if `False` disable it. Default: True.
        """
        if mode and self.log is None:
            self._init_log()
        self._track = mode

    def _init_log(self) -> None:
        self.log = dict(data_fitting=[], prior_matching=[], log_probs=[])

    def forward(
        self,
        model_output: _log_prob_return_format[Tensor],
        target: Tensor,
        dataset_size: Optional[int] = None,
    ) -> Tensor:
        r"""
        Calculate the negative ELBO loss from sampled evaluations, a target and the weight log probs.

        Accepts a Tensor of N samples, the associate log probabilities and a target to
        calculate the loss.

        Parameters
        ----------
        model_output: Tuple[Tensor, Tensor]
            The model output in with return_log_probs = True. The first Tensor is the
            sampled model prediction (Shape: (N, \*). The second Tensor contains
            prior_log_prob and variational_log_prob - the log probability of the sampled
            weights under the prior and variational distribution respectively - and has
            shape (N, 2).
        target: Tensor,
            Target prediction. Shape (\*)
        dataset_size: Optional[int] = None
            Total number of samples in the dataset. Used in place of self.dataset_size
            if provided.

        Returns
        -------
        Tensor
            Negative ELBO loss. Shape: (1,)
        """
        samples, log_probs = model_output
        # Average log probs separately and calculate prior matching term
        mean_log_probs = log_probs.mean(dim=0)

        if (dataset_size is None) and (self.dataset_size is None):
            warn(
                f"No dataset_size is provided. Batch size ({samples.shape[1]}) is used instead."
            )
            n_data = samples.shape[1]
        else:
            n_data = dataset_size or self.dataset_size

        prior_matching = self.heat * (mean_log_probs[1] - mean_log_probs[0])
        # Sample average for predictive log prob is already done
        data_fitting = (
            -n_data
            * self.predictive_distribution.log_prob_from_samples(target, samples)
            .mean(0)
            .sum()
        )

        if self._track and self.log is not None:
            self.log["data_fitting"].append(data_fitting.item())
            self.log["prior_matching"].append(prior_matching.item())
            self.log["log_probs"].append(mean_log_probs.clone().detach())

        return data_fitting + prior_matching
