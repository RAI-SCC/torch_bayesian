from typing import OrderedDict, overload

import torch
from torch import Tensor
from torch.nn import Module, Sequential

from vi import VIModule


class VISequential(VIModule, Sequential):
    """
    Sequential container equivalent to torch.nn.Sequential, that manages VIModules too.

    Detects and aggregates prior_log_prob and variational_log_prob from submodules, if
    needed. Then passes on only the output to the next module making mixed sequences of
    VIModules and nn.Modules work with and without return_log_prob.
    """

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
        """
        Forward pass that manages log probs, if required.

        Parameters
        ----------
        input_ : Varies
            Input for the first module in the stack. Passed on to it unchanged.

        Returns
        -------
        output, prior_log_prob, variational_log_prob if return_log_prob else output

        output: Varies
            Output of the module stack.
        prior_log_prob: Tensor
            Total prior log probability all internal VIModules.
            Only returned if return_log_prob.
        variational_log_prob: Tensor
            Total variational log probability all internal VIModules.
            Only returned if return_log_prob.
        """
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


class VIResidualConnection(VISequential):
    """
    A version of VISequential that supports residual connections.

    This class is identical to VISequential, but adds the input to the output.
    Importantly it manages log prob tracking, if required. Note that a single
    module can also be wrapped to add a residual connection around it.
    """

    def forward(self, input_):  # type: ignore
        """
        Forward pass that manages log probs, if required and adds the input to the output.

        Parameters
        ----------
        input_ : Varies
            Input for the first module in the stack. Passed on to it unchanged.

        Returns
        -------
        output, prior_log_prob, variational_log_prob if return_log_prob else output

        output: Varies
            Output of the module stack plus the input to the residual connection.
        prior_log_prob: Tensor
            Total prior log probability all internal VIModules.
            Only returned if return_log_prob.
        variational_log_prob: Tensor
            Total variational log probability all internal VIModules.
            Only returned if return_log_prob.
        """
        if self._return_log_prob:
            output, prior_log_prob, variational_log_prob = super().forward(input_)
            return (
                self._catch_shape_mismatch(input_, output),
                prior_log_prob,
                variational_log_prob,
            )
        else:
            output = super().forward(input_)
            return self._catch_shape_mismatch(input_, output)

    @staticmethod
    def _catch_shape_mismatch(input_: Tensor, output_: Tensor) -> Tensor:
        try:
            return output_ + input_
        except RuntimeError as e:
            if str(e).startswith("The size of tensor a"):
                raise RuntimeError(
                    f"Output shape ({output_.shape}) of residual connection must match input shape ({input_.shape})"
                )
            else:
                raise e  # pragma: no cover
