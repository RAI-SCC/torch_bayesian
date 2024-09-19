import warnings
from typing import Any, Optional, Type

import torch
from torch import Tensor
from torch.nn import Module

from vi import VIBaseModule
from vi.priors import MeanFieldNormalPrior
from vi.utils.common_types import VIReturn, _prior_any_t, _vardist_any_t
from vi.variational_distributions import MeanFieldNormalVarDist


def convert_to_vi(module: Type[Module]) -> Type[VIBaseModule]:
    """Convert torch nn.Module to VIModule."""
    assert issubclass(module, Module), f"{module.__name__} is not a torch Module"

    class VIClass(VIBaseModule, module):
        __name__ = "VI" + module.__name__

        def __init__(
            self,
            *args: Any,
            variational_distribution: _vardist_any_t = MeanFieldNormalVarDist(),
            prior: _prior_any_t = MeanFieldNormalPrior(),
            rescale_prior: bool = False,
            prior_initialization: bool = False,
            return_log_probs: bool = True,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            **kwargs: Any,
        ) -> None:
            factory_kwargs = {"device": device, "dtype": dtype}
            module.__init__(self, *args, **kwargs, **factory_kwargs)
            assert (
                self._modules == {}
            ), f"{module.__name__} cannot be converted since it has submodules"

            variable_shapes = {}
            for name in self._parameters:
                if self._parameters[name] is not None:
                    variable_shapes[name] = self._parameters[name].shape
            for name in variable_shapes:
                del self._parameters[name]
                setattr(self, name, torch.zeros(variable_shapes[name]))

            self.random_variables = tuple(variable_shapes.keys())

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message=f"{self.__class__.__name__} skipped super*"
                )
                VIBaseModule.__init__(
                    self,
                    variable_shapes=variable_shapes,
                    variational_distribution=variational_distribution,
                    prior=prior,
                    rescale_prior=rescale_prior,
                    prior_initialization=prior_initialization,
                    return_log_probs=return_log_probs,
                    **factory_kwargs,
                )

        def forward(self, *input_: Tensor) -> VIReturn[Tensor]:
            params = self.sample_variables()
            for name, value in zip(self.random_variables, params):
                setattr(self, name, value)

            output = module.forward(self, *input_)

            if self._return_log_probs:
                log_probs = self.get_log_probs(params)
                return output, log_probs
            else:
                return output

    return VIClass
