import torch

from vi import VIConv1d, VIConv2d, VIConv3d
from vi.priors import MeanFieldNormalPrior
from vi.variational_distributions import MeanFieldNormalVarDist


# Since these are basically copied from torch 2.4 testing is limited
def test_viconv1d() -> None:
    """Test VIConv1d."""
    args = dict(
        in_channels=2,
        out_channels=4,
        kernel_size=3,
        stride=2,
        padding=1,
        dilation=2,
        groups=2,
        bias=False,
        padding_mode="reflect",
        variational_distribution=MeanFieldNormalVarDist(),
        prior=MeanFieldNormalPrior(),
    )

    sample = torch.randn((6, args["in_channels"], 7))
    test1 = VIConv1d(**args, return_log_probs=True)  # type: ignore
    out1, lps1 = test1(sample, samples=5)
    assert out1.shape == (5, 6, args["out_channels"], 3)
    assert lps1.shape == (5, 2)

    for key in args:
        if key == "bias":
            assert not hasattr(test1, "_bias_mean")
            assert not hasattr(test1, "_bias_log_std")
        elif key in [
            "variational_distribution",
            "prior",
            "kernel_size",
            "stride",
            "padding",
            "dilation",
        ]:
            assert test1.__dict__[key][0] is args[key]
            assert len(test1.__dict__[key]) == 1
        else:
            assert test1.__dict__[key] is args[key]

    args["padding_mode"] = "zeros"
    args["bias"] = True
    test2 = VIConv1d(**args, return_log_probs=False)  # type: ignore

    out2 = test2(sample, samples=5)
    assert out2.shape == (5, 6, args["out_channels"], 3)


def test_viconv2d() -> None:
    """Test VIConv2d."""
    args = dict(
        in_channels=2,
        out_channels=4,
        kernel_size=3,
        stride=2,
        padding=1,
        dilation=2,
        groups=2,
        bias=False,
        padding_mode="reflect",
        variational_distribution=MeanFieldNormalVarDist(),
        prior=MeanFieldNormalPrior(),
    )

    sample = torch.randn((6, args["in_channels"], 7, 4))
    test1 = VIConv2d(**args, return_log_probs=True)  # type: ignore
    out1, lps1 = test1(sample, samples=5)
    assert out1.shape == (5, 6, args["out_channels"], 3, 1)
    assert lps1.shape == (5, 2)

    for key in args:
        if key == "bias":
            assert not hasattr(test1, "_bias_mean")
            assert not hasattr(test1, "_bias_log_std")
        elif key in [
            "variational_distribution",
            "prior",
            "kernel_size",
            "stride",
            "padding",
            "dilation",
        ]:
            assert test1.__dict__[key][0] is args[key]
            if key in ["variational_distribution", "prior"]:
                assert len(test1.__dict__[key]) == 1
            else:
                assert len(test1.__dict__[key]) == 2
        else:
            assert test1.__dict__[key] is args[key]

    args["padding_mode"] = "zeros"
    args["bias"] = True
    test2 = VIConv2d(**args, return_log_probs=False)  # type: ignore

    out2 = test2(sample, samples=5)
    assert out2.shape == (5, 6, args["out_channels"], 3, 1)


def test_viconv3d() -> None:
    """Test VIConv3d."""
    args = dict(
        in_channels=2,
        out_channels=4,
        kernel_size=3,
        stride=2,
        padding=1,
        dilation=2,
        groups=2,
        bias=False,
        padding_mode="reflect",
        variational_distribution=MeanFieldNormalVarDist(),
        prior=MeanFieldNormalPrior(),
    )

    sample = torch.randn((6, args["in_channels"], 7, 4, 9))
    test1 = VIConv3d(**args, return_log_probs=True)  # type: ignore
    out1, lps1 = test1(sample, samples=5)
    assert out1.shape == (5, 6, args["out_channels"], 3, 1, 4)
    assert lps1.shape == (5, 2)

    for key in args:
        if key == "bias":
            assert not hasattr(test1, "_bias_mean")
            assert not hasattr(test1, "_bias_log_std")
        elif key in [
            "variational_distribution",
            "prior",
            "kernel_size",
            "stride",
            "padding",
            "dilation",
        ]:
            assert test1.__dict__[key][0] is args[key]
            if key in ["variational_distribution", "prior"]:
                assert len(test1.__dict__[key]) == 1
            else:
                assert len(test1.__dict__[key]) == 3
        else:
            assert test1.__dict__[key] is args[key]

    args["padding_mode"] = "zeros"
    args["bias"] = True
    test2 = VIConv3d(**args, return_log_probs=False)  # type: ignore

    out2 = test2(sample, samples=5)
    assert out2.shape == (5, 6, args["out_channels"], 3, 1, 4)
