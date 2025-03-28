from torch import Tensor, nn

from .base import PredictiveDistribution


class NonBayesianPredictiveDistribution(PredictiveDistribution):
    """Predictive distribution for non-Bayesian forecasts."""

    predictive_parameters = ("mean",)

    def __init__(self, loss_type: str = "MSE") -> None:
        super().__init__()
        if loss_type in ["MSE", "L2"]:
            self.loss = nn.MSELoss()
        elif loss_type in ["MAE", "L1"]:
            self.loss = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    @staticmethod
    def predictive_parameters_from_samples(samples: Tensor) -> Tensor:
        """Calculate predictive mean from samples."""
        return samples.mean(dim=0)

    def log_prob_from_parameters(self, reference: Tensor, parameters: Tensor) -> Tensor:
        """Calculate the loss of the mean prediction with respect to reference."""
        return self.loss(parameters, reference)
