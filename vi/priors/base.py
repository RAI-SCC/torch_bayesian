import math
from inspect import signature
from typing import TYPE_CHECKING, Callable, Dict, Tuple
from warnings import warn

from torch import Tensor

from ..utils import PostInitCallMeta

if TYPE_CHECKING:
    from ..base import VIBaseModule  # pragma: no cover


def _reset_parameters_unimplemented(
    self: "Prior", module: "VIBaseModule", variable: str
) -> None:
    warn(
        f'Module [{type(self).__name__}] is missing the "reset_parameters" function and does not perform prior initialization'
    )


class Prior(metaclass=PostInitCallMeta):
    """
    Base for prior distributions.

    Parameters
    ----------
    distribution_parameters: Tuple[str, ...]
        Parameters characterizing the prior and can be set during prior based initialization.
    _required_parameters: Tuple[str, ...] = ()
        Parameters besides a sample needed to calculate log_prob.
    log_prob: Callable[..., Tensor]
        Function to calculate the log probability of the input sample.
    reset_parameters: Callable[[Prior, VIBaseModule], None]
        Function initializing the parameters of a VIBaseModule according to the prior distribution.
    """

    distribution_parameters: Tuple[str, ...]
    _required_parameters: Tuple[str, ...] = ()
    _scaling_parameters: Tuple[str, ...]
    _rescaled: bool = False
    log_prob: Callable[..., Tensor]
    reset_parameters: Callable[..., None] = _reset_parameters_unimplemented

    def __post_init__(self) -> None:
        """Ensure instance has required attributes."""
        if not hasattr(self, "distribution_parameters"):
            raise NotImplementedError("Subclasses must define distribution_parameters")
        if not hasattr(self, "log_prob"):
            raise NotImplementedError("Subclasses must define log_prob")
        assert (
            len(self._required_parameters)
            == (len(signature(self.log_prob).parameters) - 1)
        ), "log_prob must accept an argument for each required parameter plus the sample"
        if not hasattr(self, "_scaling_parameters"):
            self._scaling_parameters = self.distribution_parameters
        for parameter in self._scaling_parameters:
            assert hasattr(
                self, parameter
            ), f"Module [{type(self).__name__}] is missing exposed scaling parameter [{parameter}]"

    def kaiming_rescale(self, fan_in: int, eps: float = 1e-5) -> None:
        """Rescale prior based on layer width, for normalization."""
        if self._rescaled:
            pass
        else:
            self._rescaled = True
            scale = 1 / math.sqrt(fan_in) if fan_in > 0 else 0

            for parameter in self._scaling_parameters:
                param = getattr(self, parameter)
                if parameter.startswith("log"):
                    setattr(self, parameter, param + math.log(scale + eps))
                else:
                    setattr(self, parameter, param * scale)

    def match_parameters(
        self, variational_parameters: Tuple[str, ...]
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        Compare distribution parameters to other parameter list.

        Parameters
        ----------
        variational_parameters : Tuple[str]
            Tuple of parameter names to compare.

        Returns
        -------
        Tuple[Dict[str, int], Dict[str, int]]
            The first dictionary maps the names of the shared parameters to their index in self.distribution_parameters.
            The second dictionary maps the names of parameters exclusive to self.distribution_parameters to their index.
        """
        shared_params = {}
        diff_params = {}

        for i, dist_param in enumerate(self.distribution_parameters):
            if dist_param in variational_parameters:
                shared_params[dist_param] = i
            else:
                diff_params[dist_param] = i

        return shared_params, diff_params
