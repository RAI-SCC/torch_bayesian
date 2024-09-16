from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812

from .base import VIBaseModule
from .priors import MeanFieldNormalPrior
from .utils.common_types import VIReturn, _prior_any_t, _vardist_any_t
from .variational_distributions import MeanFieldNormalVarDist


class VILinear(VIBaseModule):
    """
    Equivalent of nn.Linear with variational inference.

    Called with the same arguments as nn.Linear, but accepts additional arguments.
    This module's random variables are
        ("weight", "bias") if bias == True
        ("weight", )       if bias == False

    Additional Parameters
    ---------------------
    variational_distribution: Union[VarDist, List[VarDist]]
        Variational distribution which specifies the assumed weight distribution. A list of
        distributions may be provided to specify different choices for each random variable.
        Default: MeanFieldNormalVarDist()
    prior: Union[Prior, List[Prior]]
        Prior distribution which specifies the previous knowledge about the weight distribution.
        A list of distributions may be provided to specify different choices for each random
        variable. Default: MeanFieldNormalPrior()
    rescale_prior: bool
        If True prior._scaling_parameters are scaled with the sqrt of the layer width.
        This may be necessary to maintain normalization for wide layers. Default: False
    prior_initialization: bool
        If True parameters are initialized according to the prior. If False parameters are
        initialized similar to non-Bayesian networks. Default: False
    return_log_probs: bool
        If True the model forward pass returns the log probability of the sampled weight.
        This is required for the standard loss calculation. Default: True
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        variational_distribution: _vardist_any_t = MeanFieldNormalVarDist(),
        prior: _prior_any_t = MeanFieldNormalPrior(),
        bias: bool = True,
        rescale_prior: bool = False,
        prior_initialization: bool = False,
        return_log_probs: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self._fastpath = isinstance(variational_distribution, MeanFieldNormalVarDist)

        if bias:
            self.random_variables = ("weight", "bias")
        else:
            self.random_variables = ("weight",)

        variable_shapes = dict(
            weight=(out_features, in_features),
            bias=(out_features,),
        )

        super().__init__(
            variable_shapes=variable_shapes,
            variational_distribution=variational_distribution,
            prior=prior,
            rescale_prior=rescale_prior,
            prior_initialization=prior_initialization,
            return_log_probs=return_log_probs,
            **factory_kwargs,
        )

    def _gaussian_stable_fast_forward(self, input_: Tensor) -> Tensor:
        """Alternate faster forward that can be used with MeanFieldNormalVarDist."""
        weight_mean = self._weight_mean
        weight_variance = (2 * self._weight_log_std).exp()

        if "bias" in self.random_variables:
            bias_mean = self._bias_mean
            bias_variance = (2 * self._bias_log_std).exp()
        else:
            bias_mean = None
            bias_variance = None

        output_mean = F.linear(input_, weight_mean, bias_mean)
        output_std = F.linear(input_.pow(2), weight_variance, bias_variance).sqrt()

        output = MeanFieldNormalVarDist._normal_sample(output_mean, output_std)

        return output

    def forward(self, input_: Tensor) -> VIReturn[Tensor]:
        """
        Forward computation.

        Parameters
        ----------
        input_: Tensor
            Input tensor of shape (*, in_features).

        Returns
        -------
        output, log_probs if return_log_probs else output

        output: Tensor
            Output tensor of shape (*, out_features).
            Auto-sampling will add a sample dimension at the start for the overall output.
        log_probs: Tensor
            Tensor of shape (2,) containing the total prior and variational log
            probability (in that order) of the sampled weights and biases.
            Only returned if return_log_probs.
        """
        if self._fastpath and not self._return_log_probs:
            output = self._gaussian_stable_fast_forward(input_)
            return output
        else:
            params = self.sample_variables()
            output = F.linear(input_, *params)

            if self._return_log_probs:
                log_probs = self.get_log_probs(params)
                return output, log_probs
            else:
                return output


class ApproximateFastVILinear(VILinear):
    """Alpha test of universal fastpath for VILinear."""

    def __post_init__(self) -> None:
        """Assert MeanFieldNormalVarDist is used."""
        super().__post_init__()
        for vardist in self.variational_distribution:
            assert isinstance(vardist, MeanFieldNormalVarDist)

        for prior in self.prior:
            assert prior._required_parameters == ()

    def forward(self, input_: Tensor) -> VIReturn[Tensor]:
        """
        Forward computation.

        Parameters
        ----------
        input_: Tensor
            Input tensor of shape (*, in_features).

        Returns
        -------
        output, log_probs if return_log_probs else output

        output: Tensor
            Output tensor of shape (*, out_features).
            Auto-sampling will add a sample dimension at the start for the overall output.
        log_probs: Tensor
            Tensor of shape (2,) containing the total prior and variational log
            probability (in that order) of the sampled weights and biases.
            Only returned if return_log_probs.
        """
        weight_mean = self._weight_mean
        weight_variance = (2 * self._weight_log_std).exp()

        if "bias" in self.random_variables:
            bias_mean = self._bias_mean
            bias_variance = (2 * self._bias_log_std).exp()
        else:
            bias_mean = None
            bias_variance = None

        output_mean = F.linear(input_, weight_mean, bias_mean)
        output_std = F.linear(input_.pow(2), weight_variance, bias_variance).sqrt()

        output = MeanFieldNormalVarDist._normal_sample(output_mean, output_std)

        if self._return_log_probs:
            variational_log_prob = (
                self.variational_distribution[0]
                .log_prob(output, output_mean, output_std.log())
                .sum()
                .unsqueeze(0)
            )
            prior_log_prob = self.prior[0].log_prob(output).sum().unsqueeze(0)
            return output, torch.cat([prior_log_prob, variational_log_prob])
        else:
            return output
