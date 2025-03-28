import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812

from .base import PredictiveDistribution


class CategoricalPredictiveDistribution(PredictiveDistribution):
    """Categorical predictive distribution used for classification tasks."""

    predictive_parameters = ("probs",)

    def __init__(self, input_type: str = "logits"):
        assert input_type in ["logits", "probs"], "input_type must be logits or probs"
        self._in_logits = input_type == "logits"

    def predictive_parameters_from_samples(
        self, samples: Tensor, eps: float = 1e-5
    ) -> Tensor:
        """Calculate predictive probabilities from samples."""
        if self._in_logits:
            return F.softmax(samples + eps, -1).mean(dim=0)
        else:
            normalized = samples / samples.sum(dim=-1, keepdim=True)
            return normalized.mean(dim=0)

    @staticmethod
    def log_prob_from_parameters(
        reference: Tensor, parameters: Tensor, eps: float = 1e-5
    ) -> Tensor:
        """
        Calculate log probability from parameters.

        Assumes reference and parameters are logits or probs, based on output_type.
        """
        parameters = torch.log(parameters + eps)
        value = reference.long().unsqueeze(-1)
        value, log_pmf = torch.broadcast_tensors(value, parameters)
        value = value[..., :1]
        return log_pmf.gather(-1, value).squeeze(-1)
