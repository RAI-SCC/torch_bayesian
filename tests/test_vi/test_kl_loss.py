from warnings import filterwarnings

import torch
from pytest import warns

from torch_bayesian.vi import KullbackLeiblerLoss
from torch_bayesian.vi.predictive_distributions import (
    MeanFieldNormalPredictiveDistribution,
)


def test_kl_loss() -> None:
    """Test Kullback-LeiblerLoss."""
    sample_nr = 8
    batch_size = 4
    sample_shape = (5, 3)
    samples = torch.randn((sample_nr, batch_size, *sample_shape))
    log_probs = torch.randn((sample_nr, 2))
    target = torch.randn([batch_size, *sample_shape])

    model_return = samples, log_probs
    double_return = torch.cat([samples] * 2, dim=1), log_probs
    double_target = torch.cat([target] * 2, dim=0)

    loss1 = KullbackLeiblerLoss(MeanFieldNormalPredictiveDistribution())
    with warns(
        UserWarning,
        match=f"No dataset_size is provided. Number of samples \\({sample_nr}\\) is used instead.",
    ):
        _ = loss1(model_return, target)

    out1 = loss1(model_return, target, dataset_size=sample_nr)
    ref_data_fit = (
        -sample_nr
        * loss1.predictive_distribution.log_prob_from_samples(target, samples)
        .mean(0)
        .sum()
    )
    ref_kl_term = log_probs.mean(0)[1] - log_probs.mean(0)[0]
    assert out1 == (ref_data_fit + ref_kl_term)
    assert out1.shape == ()
    assert loss1.log is None
    assert not loss1._track

    loss2 = KullbackLeiblerLoss(
        MeanFieldNormalPredictiveDistribution(), dataset_size=sample_nr
    )
    out2 = loss2(model_return, target)
    assert out1 == out2
    out3 = loss2(model_return, target, dataset_size=2 * sample_nr)
    assert out1 != out3

    loss3 = KullbackLeiblerLoss(
        MeanFieldNormalPredictiveDistribution(), dataset_size=2 * sample_nr
    )
    out4 = loss3(model_return, target)
    assert out1 != out4
    assert out3 == out4

    loss4 = KullbackLeiblerLoss(
        MeanFieldNormalPredictiveDistribution(), dataset_size=sample_nr, heat=0.5
    )
    out5 = loss4(model_return, target)
    assert out1 != out5

    filterwarnings("ignore", category=UserWarning)
    out6 = loss1(model_return, target)
    assert out1 == out6

    loss5 = KullbackLeiblerLoss(
        MeanFieldNormalPredictiveDistribution(), dataset_size=sample_nr, track=True
    )

    assert loss5.log is not None
    assert loss5._track

    loss1.track()
    assert loss1.log is not None
    assert loss1._track

    for key in ["data_fitting", "prior_matching", "log_probs"]:
        assert loss1.log[key] == []
        assert loss5.log[key] == []

    loss1(model_return, target, dataset_size=sample_nr)
    loss5(model_return, target, dataset_size=sample_nr)

    for key in ["data_fitting", "prior_matching", "log_probs"]:
        assert len(loss1.log[key]) == 1
        assert len(loss5.log[key]) == 1
        comp = loss1.log[key][0] == loss5.log[key][0]
        if isinstance(comp, bool):
            assert comp
        else:
            assert comp.all()

    assert loss1.log["log_probs"][0][1] - loss1.log["log_probs"][0][0] == ref_kl_term
    assert loss1.log["data_fitting"][0] + loss1.log["prior_matching"][0] == out1

    double_out = loss1(double_return, double_target, dataset_size=sample_nr)
    assert torch.allclose(double_out, out1)
