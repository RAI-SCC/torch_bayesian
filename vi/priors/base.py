from inspect import signature
from typing import Callable, Dict, Tuple

from torch import Tensor

from ..utils import ForceRequiredAttributeDefinitionMeta


class Prior(metaclass=ForceRequiredAttributeDefinitionMeta):
    """Base for prior distributions."""

    distribution_parameters: Tuple[str, ...]
    _required_parameters: Tuple[str, ...] = ()
    log_prob: Callable[..., Tensor]

    def check_required_attributes(self) -> None:
        """Ensure instance has required attributes."""
        if not hasattr(self, "distribution_parameters"):
            raise NotImplementedError("Subclasses must define distribution_parameters")
        if not hasattr(self, "log_prop"):
            raise NotImplementedError("Subclasses must define log_prob")
        assert (
            len(self._required_parameters)
            == (len(signature(self.log_prob).parameters) + 1)
        ), "log_prob must accept an argument for each required parameter plus the sample"

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
