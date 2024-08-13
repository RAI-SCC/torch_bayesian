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
        bias: bool = True,
        prior_initialization: bool = False,
        return_log_prob: bool = False,
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
            prior_initialization=prior_initialization,
            return_log_prob=return_log_prob,
            **factory_kwargs,
        )

    def forward(self, input_: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        """Forward computation."""
        params = self.sample_variables()

        output = F.linear(input_, *params)

        if self._return_log_prob:
            prior_log_prob, variational_log_prob = self.get_log_probs(params)
            return output, prior_log_prob, variational_log_prob
        else:
            return output
