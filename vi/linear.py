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
        params = self.sample_variables()

        output = F.linear(input_, *params)

        if self._return_log_probs:
            log_probs = self.get_log_probs(params)
            return output, log_probs
        else:
            return output
