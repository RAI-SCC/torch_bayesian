from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812

from .base import VIBaseModule
from .variational_distributions import VariationalDistribution


class VILinear(VIBaseModule):
    """Equivalent of nn.Linear."""

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int

    def __init__(
        self,
        variational_distribution: VariationalDistribution,
        in_features: int,
        out_features: int,
        bias: bool = True,
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
            **factory_kwargs,
        )

    def forward(self, input_: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward computation."""
        weight_variational_parameters = self.get_variational_parameters("weight")
        weight = self.variational_distribution.sample(*weight_variational_parameters)
        if self.bias:
            bias_variational_parameters = self.get_variational_parameters("bias")
            bias = self.variational_distribution.sample(*bias_variational_parameters)
        else:
            bias = torch.tensor(False)

        return F.linear(input_, weight, bias), weight.clone(), bias.clone()
