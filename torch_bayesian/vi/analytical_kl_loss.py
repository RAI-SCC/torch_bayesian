from abc import ABC
from math import log, pi
from typing import Callable, Dict, Iterable, Optional, Tuple, Type, Union
from warnings import warn

import torch
from torch import Tensor
from torch.nn import Module

from . import _globals
from .base import VIBaseModule, VIModule
from .predictive_distributions import PredictiveDistribution
from .priors import MeanFieldNormalPrior, Prior, UniformPrior
from .variational_distributions import (
    MeanFieldNormalVarDist,
    NonBayesian,
    VariationalDistribution,
)


def _forward_unimplemented(
    self: "KullbackLeiblerModule", *input_: Optional[Tensor]
) -> Tensor:
    """
    Define the computation performed at every call.

    Should be overridden by all subclasses.

    .. note::
        Although the recipe for forward pass needs to be defined within this function,
        one should call the :class:`KullbackLeiblerModule` instance afterward instead of
        this.
    """
    raise NotImplementedError(
        f'{type(self).__name__} is missing the required "forward" function'
    )


class KullbackLeiblerModule(ABC):
    """Base class for modules calculating the Kullback-Leibler divergence from distribution parameters."""

    forward: Callable[..., Tensor] = _forward_unimplemented

    def __call__(
        self,
        prior_parameters: Iterable[Tensor],
        variational_parameters: Iterable[Tensor],
    ) -> Tensor:
        """Distribute parameters to forward function."""
        return self.forward(*prior_parameters, *variational_parameters)


class NormalNormalDivergence(KullbackLeiblerModule):
    """Kullback-Leibler divergence between two normal distributions."""

    @staticmethod
    def forward(
        prior_mean: Tensor,
        prior_log_std: Tensor,
        variational_mean: Tensor,
        variational_log_std: Tensor,
    ) -> Tensor:
        """
        Calculate the Kullback-Leibler divergence.

        Calculates the KL-Divergence between a normal prior and a normal variational distribution.
        Both distributions are expected to be uncorrelated, i.e. have diagonal variance.
        All input tensors must have the same shape.

        Parameters
        ----------
        prior_mean: Tensor
            Means of the prior distribution.
        prior_log_std: Tensor
            Log standard deviations of the prior distribution.
        variational_mean: Tensor
            Means of the variational distribution.
        variational_log_std: Tensor
            Standard deviations of the variational distribution.

        Returns
        -------
        Tensor
            The KL-Divergence of the two distributions.
        """
        variational_variance = torch.exp(2 * variational_log_std)
        prior_variance = torch.exp(2 * prior_log_std)
        variance_ratio = prior_variance / variational_variance

        raw_kl = (
            variance_ratio.log()
            + (prior_mean - variational_mean).pow(2) / variance_ratio
        ) / 2
        if _globals._USE_NORM_CONSTANTS:
            raw_kl = raw_kl - 0.5

        return raw_kl.sum()


class NonBayesianDivergence(KullbackLeiblerModule):
    """Placeholder Kullback-Leibler divergence for non-Bayesian models."""

    @staticmethod
    def forward(*args: Tensor) -> Tensor:
        """Return placeholder zero."""
        return torch.tensor([0.0], device=args[0].device)


class UniformNormalDivergence(KullbackLeiblerModule):
    """Kullback-Leibler divergence between a uniform and normal distribution."""

    @staticmethod
    def forward(variational_mean: Tensor, variational_log_std: Tensor) -> Tensor:
        """
        Calculate the Kullback-Leibler divergence.

        Calculates the KL-Divergence between a uniform prior and a normal, uncorrelated variational distribution.
        All input tensors must have the same shape.

        Parameters
        ----------
        variational_mean: Tensor
            Means of the variational distribution.
        variational_log_std: Tensor
            Standard deviations of the variational distribution.

        Returns
        -------
        Tensor
            The KL-Divergence of the two distributions.
        """
        raw_kl = variational_log_std
        if _globals._USE_NORM_CONSTANTS:
            raw_kl = raw_kl - 0.5 * (1 + log(2 * pi))
        return raw_kl.sum()


_kl_div_dict: Dict[str, Type[KullbackLeiblerModule]] = dict(
    NormalNormalDivergence=NormalNormalDivergence,
    UniformNormalDivergence=UniformNormalDivergence,
)


class AnalyticalKullbackLeiblerLoss(Module):
    """Analytical Kullback Leibler Loss function."""

    def __init__(
        self,
        model: VIModule,
        predictive_distribution: PredictiveDistribution,
        dataset_size: Optional[int] = None,
        divergence_type: Union[
            "KullbackLeiblerModule", Tuple[Prior, VariationalDistribution], None
        ] = None,
        heat: float = 1.0,
        track: bool = False,
    ) -> None:
        super().__init__()
        self.predictive_distribution = predictive_distribution
        self.dataset_size = dataset_size
        self.heat = heat
        self._track = track

        if divergence_type is None:
            for module in model.modules():
                if not isinstance(module, VIBaseModule):
                    continue
                for prior, var_dist in zip(
                    module.priors, module.variational_distributions
                ):
                    kl_type = self._detect_divergence(prior, var_dist)
                    if divergence_type is None:
                        divergence_type = kl_type
                    else:
                        try:
                            assert isinstance(kl_type, type(divergence_type))
                        except AssertionError:
                            raise NotImplementedError(
                                "Handling of inconsistent distributions types is not implemented yet."
                            )
        if divergence_type is None:
            raise ValueError("Provided model is not bayesian.")

        if isinstance(divergence_type, tuple):
            divergence_type = self._detect_divergence(*divergence_type)
        self.kl_module: KullbackLeiblerModule = divergence_type
        model.return_log_probs(False)
        self.model = model

    @staticmethod
    def _detect_divergence(
        prior: Prior, var_dist: VariationalDistribution
    ) -> KullbackLeiblerModule:
        if isinstance(prior, MeanFieldNormalPrior):
            prior_name = "Normal"
        elif isinstance(prior, UniformPrior):
            prior_name = "Uniform"
        else:
            prior_name = None
        if isinstance(var_dist, NonBayesian):
            return NonBayesianDivergence()
        elif isinstance(var_dist, MeanFieldNormalVarDist):
            vardist_name = "Normal"
        else:
            vardist_name = None

        assert (
            prior_name is not None and vardist_name is not None
        ), f"Analytical loss is not implemented for {prior.__class__.__name__} and {var_dist.__class__.__name__}."

        return _kl_div_dict[prior_name + vardist_name + "Divergence"]()

    def prior_matching(self) -> Tensor:
        """Calculate the prior matching KL-Divergence of ``self.model``."""
        total_kl = torch.tensor([])
        for module in self.model.modules():
            if not isinstance(module, VIBaseModule):
                continue

            for var, prior in zip(module.random_variables, module.prior):
                prior_params = []
                for param in prior.distribution_parameters:
                    prior_params.append(getattr(prior, param))
                variational_params = module.get_variational_parameters(var)

                variable_kl = self.kl_module(prior_params, variational_params)
                total_kl = total_kl + variable_kl

        return total_kl

    def forward(
        self, model_output: Tensor, target: Tensor, dataset_size: Optional[int] = None
    ) -> Tensor:
        """
        Calculate the negative ELBO loss from sampled evaluations and a target.

        Accepts a Tensor of N samples and a target to calculate the loss.

        Parameters
        ----------
        model_output: Tensor
            The model output in with return_log_probs = False, i.e. the sampled model
            prediction. Shape: (N, *)
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
        samples = model_output

        if (dataset_size is None) and (self.dataset_size is None):
            warn(
                f"No dataset_size is provided. Number of samples ({samples.shape[0]}) is used instead."
            )
            n_data = samples.shape[0]
        else:
            n_data = dataset_size or self.dataset_size

        prior_matching = self.heat * self.prior_matching()
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

        return data_fitting + prior_matching
