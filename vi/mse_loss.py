from typing import Optional
from warnings import warn

from torch import Tensor
from torch.nn import Module
from torch.nn import MSELoss

from .predictive_distributions import PredictiveDistribution
from .utils.common_types import _log_prob_return_format


class MeanSquaredErrorLoss(Module):
    """
    Mean Squared Error Loss.

    Calculates the mean of the squared differences between predicted and actual values.

    """

    def __init__(
        self,
    ) -> None:
        super().__init__()
    def forward(
        self,
        model_output: Tensor,
        target: Tensor
    ) -> Tensor:
        """
        Calculate the MSE loss from sampled evaluations and a target.

        Accepts a Tensor of N samples and a target to calculate the loss.

        Parameters
        ----------
        model_output: Tensor
            The model output in with return_log_probs = False. The Tensor is the
            sampled model prediction (Shape: (N, *).
        target: Tensor,
            Target prediction. Shape (*)

        Returns
        -------
        Tensor
            MSE loss. Shape: (1,)
        """
        # Average over the N sampled predictions
        mean_model_output = model_output.mean(dim=0)
        loss = MSELoss()
        return loss(mean_model_output, target)
