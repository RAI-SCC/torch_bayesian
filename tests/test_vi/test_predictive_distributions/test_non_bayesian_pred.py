from typing import Optional

import pytest
import torch
from torch import nn

from torch_bayesian.vi.predictive_distributions import NonBayesianPredictiveDistribution


@pytest.mark.parametrize(
    "loss_type, target",
    [
        ("MSE", nn.MSELoss),
        ("L2", nn.MSELoss),
        ("MAE", nn.L1Loss),
        ("L1", nn.L1Loss),
        ("Error", None),
    ],
)
def test_non_bayesian_predictive_distribution(
    loss_type: str, target: Optional[nn.Module], device: torch.device
) -> None:
    """Test Non-Bayesian Predictive Distribution."""
    if target is None:
        with pytest.raises(ValueError, match=f"Unsupported loss type: {loss_type}"):
            _ = NonBayesianPredictiveDistribution(loss_type)
        return

    predictive_distribution = NonBayesianPredictiveDistribution(loss_type)
    assert isinstance(predictive_distribution.loss, target)

    sample_shape = (4, 7)
    samples = torch.randn((1, *sample_shape), device=device)
    reference = torch.randn(sample_shape, device=device)

    target_mean = samples.mean(dim=0)
    target_loss = target()(target_mean, reference)

    predictive_mean = predictive_distribution.predictive_parameters_from_samples(
        samples
    )
    loss = predictive_distribution.log_prob_from_samples(reference, samples)

    assert target_mean.shape == predictive_mean.shape
    assert predictive_mean.device == device
    assert torch.allclose(target_mean, predictive_mean)

    assert target_loss.shape == loss.shape
    assert loss.device == device
    assert torch.allclose(target_loss, loss)
