import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module, Parameter, init

from .priors import Prior
from .variational_distributions import VariationalDistribution


def _forward_unimplemented(self: Module, *input_: Optional[Tensor]) -> Tuple[Tensor]:
    r"""Define the computation performed at every call.

    Should be overridden by all subclasses.

    .. note::
        Although the recipe for forward pass needs to be defined within
        this function, one should call the :class:`Module` instance afterwards
        instead of this since the former takes care of running the
        registered hooks while the latter silently ignores them.
        For VIModules all inputs must be Tensors.
    """
    raise NotImplementedError(
        f'Module [{type(self).__name__}] is missing the required "forward" function'
    )


class VIModule(Module):
    """Base class for Modules using Variational Inference."""

    forward: Callable[..., Tuple[Tensor, ...]] = _forward_unimplemented

    @staticmethod
    def _expand_to_samples(input_: Optional[Tensor], samples: int) -> Tensor:
        if input_ is None:
            input_ = torch.tensor(False)
        return input_.expand(samples, *input_.shape)

    def sampled_forward(
        self, *input_: Optional[Tensor], samples: int = 10
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the module evaluating multiple weight samples.

        Parameters
        ----------
         input_: Tensor
            Any number of input Tensors
        samples : int
            Number of weight samples to evaluate

        Returns
        -------
        Tensor
            All model outputs as Tensor with the sample dimension first
        Tensor
            Tensor containing the sampled weights and associated variational parameters (need for some losses)
        """
        expanded = [self._expand_to_samples(x, samples=samples) for x in input_]
        return torch.vmap(self.forward, randomness="different")(*expanded)


class VIBaseModule(VIModule):
    """Base class for VIModules that draw weights from a variational distribution."""

    random_variables: Tuple[str, ...] = ("weight", "bias")

    def __init__(
        self,
        variable_shapes: Dict[str, Tuple[int, ...]],
        variational_distribution: Union[
            VariationalDistribution, List[VariationalDistribution]
        ],
        prior: Union[Prior, List[Prior]],
        prior_initialization: bool = False,
        return_log_prob: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if isinstance(variational_distribution, List):
            assert (
                len(variational_distribution) == len(self.random_variables)
            ), "Provide either exactly one variational distribution or exactly one for each random variable"
            self.variational_distribution = variational_distribution
        else:
            self.variational_distribution = [variational_distribution] * len(
                self.random_variables
            )

        if isinstance(prior, List):
            assert (
                len(prior) == len(self.random_variables)
            ), "Provide either exactly one prior distribution or exactly one for each random variable"
            self.prior = prior
        else:
            self.prior = [prior] * len(self.random_variables)

        self._prior_init = prior_initialization
        self.return_log_prob = return_log_prob

        for variable, vardist in zip(
            self.random_variables, self.variational_distribution
        ):
            assert variable in variable_shapes, f"shape of {variable} is missing"
            shape = variable_shapes[variable]
            for variational_parameter in vardist.variational_parameters:
                parameter_name = self.variational_parameter_name(
                    variable, variational_parameter
                )
                setattr(
                    self,
                    parameter_name,
                    Parameter(torch.empty(shape, **factory_kwargs)),
                )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset or initialize the parameters of the Module."""
        self.reset_mean()
        for variable, vardist, prior in zip(
            self.random_variables, self.variational_distribution, self.prior
        ):
            vardist.reset_parameters(self, variable)
            if self._prior_init:
                prior.reset_parameters(self, variable)

    def reset_mean(self) -> None:
        """Reset the means of random variables similar to non-Bayesian Networks."""
        for variable in self.random_variables:
            parameter_name = self.variational_parameter_name(variable, "mean")
            if variable == "bias" and hasattr(self, parameter_name):
                weight_mean = self.variational_parameter_name("weight", "mean")
                assert hasattr(
                    self, weight_mean
                ), "Standard initialization of bias requires weight"
                fan_in, _ = init._calculate_fan_in_and_fan_out(
                    getattr(self, weight_mean)
                )
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(getattr(self, parameter_name), -bound, bound)
            elif hasattr(self, parameter_name):
                init.kaiming_uniform_(getattr(self, parameter_name), a=math.sqrt(5))

    @staticmethod
    def variational_parameter_name(variable: str, variational_parameter: str) -> str:
        """Obtain the attribute name of the variational parameter for the specified variable."""
        spec = ["", variable, variational_parameter]
        return "_".join(spec)

    def get_variational_parameters(self, variable: str) -> List[Tensor]:
        """Obtain all variational parameters for the specified variable."""
        vardist = self.variational_distribution[self.random_variables.index(variable)]
        return [
            getattr(self, self.variational_parameter_name(variable, param))
            for param in vardist.variational_parameters
        ]
