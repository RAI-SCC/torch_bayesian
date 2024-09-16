from math import exp

import torch
from torch import Tensor

from vi import VILinear
from vi.priors import MeanFieldNormalPrior


def test_vilinear() -> None:
    """Test VILinear and prior initialization."""
    in_features = 3
    out_features = 5
    module1 = VILinear(in_features, out_features, return_log_probs=False)
    assert module1.in_features == in_features
    assert module1.out_features == out_features
    assert module1.random_variables == ("weight", "bias")

    sample1 = torch.randn(4, in_features)
    out = module1(sample1, samples=10)
    assert out.shape == (10, 4, out_features)
    out.sum().backward()

    module1._has_sampling_responsibility = False
    out = module1(sample1)
    assert out.shape == (4, out_features)

    sample2 = torch.randn(3, 2, in_features)
    multisample: Tensor = module1.sampled_forward(sample2, samples=4)
    assert multisample.shape == (4, 3, 2, out_features)
    for s in multisample[1:]:
        assert not (s == multisample[0]).all()
    multisample.sum().backward()

    # test bias == False
    in_features = 4
    out_features = 3
    module2 = VILinear(in_features, out_features, bias=False, return_log_probs=False)
    assert module2.random_variables == ("weight",)
    assert not hasattr(module2, "_bias_mean")

    sample2 = torch.randn(6, in_features)
    out = module2(sample2, samples=4)
    assert out.shape == (4, 6, out_features)

    # test return_log_probs == True
    in_features = 2
    out_features = 5
    module3 = VILinear(in_features, out_features, return_log_probs=True)

    sample3 = torch.randn(4, 7, in_features)
    out, lps = module3(sample3, samples=5)
    assert out.shape == (5, 4, 7, out_features)
    assert lps.shape == (5, 2)

    module3._has_sampling_responsibility = False
    out, lps = module3(sample3)
    assert out.shape == (4, 7, out_features)
    assert lps.shape == (2,)
    out.sum().backward()

    multisample2 = module3.sampled_forward(sample3, samples=10)
    assert multisample2[0].shape == (10, 4, 7, out_features)
    assert multisample2[1].shape == (10, 2)
    multisample2[0].sum().backward()

    # test prior_init
    in_features = 7
    out_features = 3
    module4 = VILinear(
        in_features, out_features, prior_initialization=True, return_log_probs=False
    )

    weight_mean = module4._weight_mean.clone()
    weight_log_std = module4._weight_log_std.clone()
    bias_mean = module4._bias_mean.clone()
    bias_log_std = module4._bias_log_std.clone()

    assert (weight_mean == torch.zeros_like(weight_mean)).all()
    assert (weight_log_std == torch.zeros_like(weight_log_std)).all()
    assert (bias_mean == torch.zeros_like(bias_mean)).all()
    assert (bias_log_std == torch.zeros_like(bias_log_std)).all()

    module4.reset_parameters()
    assert (weight_mean == module4._weight_mean).all()
    assert (weight_log_std == module4._weight_log_std).all()
    assert (bias_mean == module4._bias_mean).all()
    assert (bias_log_std == module4._bias_log_std).all()

    in_features = 6
    out_features = 5
    prior = MeanFieldNormalPrior(mean=1.0, std=exp(5.0))
    module5 = VILinear(
        in_features,
        out_features,
        prior=prior,
        prior_initialization=True,
        return_log_probs=False,
    )

    weight_mean = module5._weight_mean.clone()
    weight_log_std = module5._weight_log_std.clone()
    bias_mean = module5._bias_mean.clone()
    bias_log_std = module5._bias_log_std.clone()

    assert (weight_mean == torch.ones_like(weight_mean)).all()
    assert (weight_log_std == torch.full_like(weight_log_std, 5.0)).all()
    assert (bias_mean == torch.ones_like(bias_mean)).all()
    assert (bias_log_std == torch.full_like(bias_log_std, 5.0)).all()


def test_fastpath() -> None:
    """Test Gaussian fastpath of VILinear."""
    in_features = 7
    out_features = 9

    module1 = VILinear(in_features, out_features)
    sample1 = torch.randn(4, in_features)

    out1, _ = module1(sample1, samples=10000)
    module1.return_log_probs(False)
    out2 = module1(sample1, samples=10000)
    out3 = torch.vmap(module1._gaussian_stable_fast_forward, randomness="different")(
        sample1.expand(10000, 4, in_features)
    )

    mean1 = out1.mean(0)
    mean2 = out2.mean(0)
    mean3 = out3.mean(0)

    assert torch.allclose(mean1, mean2, rtol=1e-1)
    assert torch.allclose(mean2, mean3, rtol=1e-1)

    std1 = out1.std(0)
    std2 = out2.std(0)
    std3 = out3.std(0)

    assert torch.allclose(std1, std2, rtol=1e-1)
    assert torch.allclose(std2, std3, rtol=1e-1)
