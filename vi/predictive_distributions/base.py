from typing import Callable, Tuple, Union

from torch import Tensor

from ..utils import ForceRequiredAttributeDefinitionMeta


class PredictiveDistribution(metaclass=ForceRequiredAttributeDefinitionMeta):
    """Base class for all predictive distributions."""

    predictive_parameters: Tuple[str, ...]
    predictive_parameters_from_samples: Callable[
        [Tensor], Union[Tensor, Tuple[Tensor, ...]]
    ]
    log_prob_from_parameters: Callable[
        [Tensor, Union[Tensor, Tuple[Tensor, ...]]], Tensor
    ]

    def check_required_attributes(self) -> None:
        """Ensure instance has required attributes."""
        if not hasattr(self, "predictive_parameters"):
            raise NotImplementedError("Subclasses must define predictive_parameters")
        if not hasattr(self, "predictive_parameters_from_samples"):
            raise NotImplementedError(
                "Subclasses must define predictive_parameters_from_samples"
            )
        if not hasattr(self, "log_prob_from_parameters"):
            raise NotImplementedError("Subclasses must define log_prob_from_parameters")

    def log_prob_from_samples(self, reference: Tensor, *samples: Tensor) -> Tensor:
        """Calculate the log probability for reference given a set of samples."""
        params = self.predictive_parameters_from_samples(*samples)
        return self.log_prob_from_parameters(reference, params)
