from inspect import signature
from typing import TYPE_CHECKING, Callable, Dict, Tuple

from torch import Tensor
from torch.nn import init

from ..utils import ForceRequiredAttributeDefinitionMeta

if TYPE_CHECKING:
    from ..base import VIBaseModule  # pragma: no cover


class VariationalDistribution(metaclass=ForceRequiredAttributeDefinitionMeta):
    """Base class for variational distributions."""

    variational_parameters: Tuple[str, ...]
    _default_variational_parameters: Tuple[float, ...]
    sample: Callable[..., Tensor]
    log_prob: Callable[..., Tensor]

    def reset_parameters(self, module: "VIBaseModule") -> None:
        """
        Reset the variational parameters of module.

        This ignores the parameter `mean` that is set by module.reset_mean.
        All other variational parameters are reset to their default value as specified in `_default_variational_parameters`.
        It may be overwritten by prior.reset_parameters(), if enabled.

        Parameters
        ----------
        module : VIModule
            VIModule to reset parameters.
        """
        for variable in module.random_variables:
            for parameter, default in zip(
                self.variational_parameters, self._default_variational_parameters
            ):
                if parameter == "mean":
                    # First moment should be reset by the module by implementing reset_mean
                    continue
                parameter_name = module.variational_parameter_name(variable, parameter)
                init.constant_(getattr(module, parameter_name), default)

    def check_required_attributes(self) -> None:
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
