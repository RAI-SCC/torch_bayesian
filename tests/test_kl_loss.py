from warnings import filterwarnings

import torch

from vi import KullbackLeiblerLoss
from vi.predictive_distributions import MeanFieldNormalPredictiveDistribution


def test_kl_loss() -> None:
    """Test Kullback-LeiblerLoss."""
    sample_nr = 8
    sample_shape = (5,)
    samples = torch.randn((sample_nr, *sample_shape))
    prior_log_prob = torch.randn((sample_nr,))
    variational_log_prob = torch.randn((sample_nr,))
    target = torch.randn(sample_shape)

    loss1 = KullbackLeiblerLoss(MeanFieldNormalPredictiveDistribution())
    filterwarnings("error")
    try:
        _ = loss1(samples, prior_log_prob, variational_log_prob, target)
        raise AssertionError
    except UserWarning as e:
        assert (
            str(e)
            == f"No dataset_size is provided. Number of samples ({sample_nr}) is used instead."
        )

    out1 = loss1(
        samples, prior_log_prob, variational_log_prob, target, dataset_size=sample_nr
    )
    assert out1.shape == ()

    loss2 = KullbackLeiblerLoss(
        MeanFieldNormalPredictiveDistribution(), dataset_size=sample_nr
    )
    out2 = loss2(samples, prior_log_prob, variational_log_prob, target)
    assert out1 == out2
    out3 = loss2(
        samples,
        prior_log_prob,
        variational_log_prob,
        target,
        dataset_size=2 * sample_nr,
    )
    assert out1 != out3

    loss3 = KullbackLeiblerLoss(
        MeanFieldNormalPredictiveDistribution(), dataset_size=2 * sample_nr
    )
    out4 = loss3(samples, prior_log_prob, variational_log_prob, target)
    assert out1 != out4
    assert out3 == out4

    loss4 = KullbackLeiblerLoss(
        MeanFieldNormalPredictiveDistribution(), dataset_size=sample_nr, heat=0.5
    )
    out5 = loss4(samples, prior_log_prob, variational_log_prob, target)
    assert out1 != out5

    filterwarnings("ignore", category=UserWarning)
    out6 = loss1(samples, prior_log_prob, variational_log_prob, target)
    assert out1 == out6
