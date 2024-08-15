from typing import OrderedDict, overload

import torch
from torch.nn import Module, Sequential

from vi import VIModule


class VISequential(VIModule, Sequential):
    """Sequential container equivalent to torch.nn.Sequential, that manages VIModules too."""

    @overload
    def __init__(self, *args: Module) -> None: ...

    @overload
    def __init__(self, arg: OrderedDict[str, Module]) -> None: ...

    def __init__(self, *args) -> None:  # type: ignore
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, input_):  # type: ignore
        """Forward pass that manages log probs, if required."""
        if self._return_log_prob:
            input_ = [input_]
            total_prior_log_prob = total_var_log_prob = torch.tensor(0.0)
            for module in self:
                if isinstance(module, VIModule):
                    out = module(*input_)
                    input_ = out[:-2]
                    prior_log_prob = out[-2]
                    var_log_prob = out[-1]

                    total_prior_log_prob = total_prior_log_prob + prior_log_prob
                    total_var_log_prob = total_var_log_prob + var_log_prob
                else:
                    input_ = [module(*input_)]
            return *input_, total_prior_log_prob, total_var_log_prob
        else:
            for module in self:
                input_ = module(input_)
            return input_
