import math
from inspect import signature
from typing import TYPE_CHECKING, Callable, Dict, Tuple

from torch import Tensor
from torch.nn import init

from ..utils import PostInitCallMeta

if TYPE_CHECKING:
    from ..base import VIBaseModule  # pragma: no cover


class VariationalDistribution(metaclass=PostInitCallMeta):
    """Base class for variational distributions."""

    variational_parameters: Tuple[str, ...]
    _default_variational_parameters: Tuple[float, ...]
    sample: Callable[..., Tensor]
    log_prob: Callable[..., Tensor]

    def reset_parameters(
        self,
        module: "VIBaseModule",
        variable: str,
        fan_in: int,
        kaiming_scaling: bool = True,
    ) -> None:
        """
        Reset the variational parameters of module.

        Parameters equivalent to non-Bayesian weights (currently "mean", "mode", or
        "loc") are reset accordingly using Kaiming uniform initialization based on
        fan_in (cf. torch.init._calculate_fan_in_and_fan_out). Other parameters are
        initialized to the fixed values specified by class defaults.
        If kaiming_scaling is True, the defaults are scaled with scale * default. Any
        parameter beginning with "log" is assumed to be in log space and scaled with
        default + log(scale). The scale is 1 / sqrt(fan_in) for vectors and
        1 / sqrt(3*fan_in) for matrices.

        Parameters
        ----------
        module : VIModule
            Module to reset parameters.
        variable : str
            Name of the variable to reset.
        fan_in : int
            Size if  the input parameter map.
        kaiming_scaling : bool
            Whether th scale all parameters according to input map size. Default: True
        """
        for parameter, default in zip(
            self.variational_parameters, self._default_variational_parameters
        ):
            parameter_name = module.variational_parameter_name(variable, parameter)
            param = getattr(module, parameter_name)

            if parameter in ["mean", "mode", "loc"]:
                self._init_uniform(param, fan_in)
            elif not kaiming_scaling:
                init.constant_(param, default)
            else:
                is_log = parameter.startswith("log")
                self._init_constant(param, default, fan_in, is_log)

    @staticmethod
    def _init_constant(
        parameter: Tensor, default: float, fan_in: int, is_log: bool, eps: float = 1e-5
    ) -> None:
        scale = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        if is_log:
            init.constant_(parameter, default + math.log(scale + eps))
        else:
            init.constant_(parameter, scale * default)

    @staticmethod
    def _init_uniform(parameter: Tensor, fan_in: int) -> None:
        if parameter.dim() < 2:
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(parameter, -bound, bound)
        else:
            init.kaiming_uniform_(parameter, a=math.sqrt(5))

    def __post_init__(self) -> None:
        """Ensure instance has required attributes."""
        if not hasattr(self, "variational_parameters"):
            raise NotImplementedError("Subclasses must define variational_parameters")
        if not hasattr(self, "_default_variational_parameters"):
            raise NotImplementedError(
                "Subclasses must define _default_variational_parameters"
            )
        assert len(self.variational_parameters) == len(
            self._default_variational_parameters
        ), "Each variational parameter must be assigned a default value"
        if not hasattr(self, "sample"):
            raise NotImplementedError("Subclasses must define the sample method")
        assert len(self.variational_parameters) == len(
            signature(self.sample).parameters
        ), "Sample must accept exactly one Tensor for each variational parameter"
        if not hasattr(self, "log_prob"):
            raise NotImplementedError("Subclasses must define log_prob")
        assert (
            len(self.variational_parameters)
            == (len(signature(self.log_prob).parameters) - 1)
        ), "log_prob must accept an argument for each variational parameter plus the sample"

    def match_parameters(
        self, distribution_parameters: Tuple[str, ...]
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        Compare variational parameters to other parameter list.

        Parameters
        ----------
        distribution_parameters : Tuple[str]
            Tuple of parameter names to compare.

        Returns
        -------
        Tuple[Dict[str, int], Dict[str, int]]
            The first dictionary maps the names of the shared parameters to their index in self.variational_parameters.
            The second dictionary maps the names of parameters exclusive to self.variational_parameters to their index.
        """
        shared_params = {}
        diff_params = {}

        for i, var_param in enumerate(self.variational_parameters):
            if var_param in distribution_parameters:
                shared_params[var_param] = i
            else:
                diff_params[var_param] = i

        return shared_params, diff_params
