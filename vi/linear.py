from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812

from .base import VIBaseModule
from .priors import MeanFieldNormalPrior, Prior
from .variational_distributions import MeanFieldNormalVarDist, VariationalDistribution


class VILinear(VIBaseModule):
    """Equivalent of nn.Linear."""

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        variational_distribution: Union[
            VariationalDistribution, List[VariationalDistribution]
        ] = MeanFieldNormalVarDist(),
        prior: Union[Prior, List[Prior]] = MeanFieldNormalPrior(),
        prior_initialization: bool = False,
        bias: bool = True,
        return_log_prob: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        if not bias:
            self.random_variables = ("weight",)
        else:
            self.random_variables = ("weight", "bias")

        variable_shapes = dict(
            weight=(in_features, out_features),
            bias=(out_features,),
        )

        super().__init__(
            variable_shapes=variable_shapes,
            variational_distribution=variational_distribution,
            prior=prior,
            prior_initialization=prior_initialization,
            return_log_prob=return_log_prob,
            **factory_kwargs,
        )

    def forward(self, input_: Tensor) -> Union[Tuple[Tensor, Tensor, Tensor], Tensor]:
        """Forward computation."""
        params = []
        for variable, vardist in zip(
            self.random_variables, self.variational_distribution
        ):
            variational_parameters = self.get_variational_parameters(variable)
            params.append(vardist.sample(*variational_parameters))

        output = F.linear(input_, *params)

        if self.return_log_prob:
            variational_log_prob = 0.0
            prior_log_prob = 0.0
            for sample, variable, vardist, prior in zip(
                params, self.random_variables, self.variational_distribution, self.prior
            ):
                variational_parameters = self.get_variational_parameters(variable)
                variational_log_prob = (
                    variational_log_prob
                    + vardist.log_prob(sample, *variational_parameters).sum()
                )

                prior_params = [
                    getattr(self, self.variational_parameter_name(variable, param))
                    for param in prior._required_parameters
                ]
                prior_log_prob = (
                    prior_log_prob + prior.log_prob(sample, *prior_params).sum()
                )
            return output, variational_log_prob, prior_log_prob
        else:
            return output
