import torch

from vi import VIConv1d, VIConv2d, VIConv3d
from vi.conv import _VIConvNd
from vi.priors import MeanFieldNormalPrior
from vi.variational_distributions import MeanFieldNormalVarDist


# Since these are basically copied from torch 2.4 testing is limited
def test_viconvnd() -> None:
    """Test baseclass _VIConvNd."""
    args = dict(
        in_channels=1,
        out_channels=3,
        kernel_size=(3,),
        stride=(2,),
        padding="blub",
        dilation=(1,),
        transposed=False,
        output_padding=(0,),
        groups=0,
        bias=True,
        padding_mode="blub",
        variational_distribution=MeanFieldNormalVarDist(),
        prior=MeanFieldNormalPrior(),
    )

    # Test errors
    try:
        _ = _VIConvNd(**args)  # type: ignore
        raise AssertionError
    except ValueError as e:
        assert str(e) == "groups must be a positive integer"
    args["groups"] = 2

    try:
        _ = _VIConvNd(**args)  # type: ignore
        raise AssertionError
    except ValueError as e:
        assert str(e) == "in_channels must be divisible by groups"
    args["in_channels"] = 2

    try:
        _ = _VIConvNd(**args)  # type: ignore
        raise AssertionError
    except ValueError as e:
        assert str(e) == "out_channels must be divisible by groups"
    args["groups"] = 1

    try:
        _ = _VIConvNd(**args)  # type: ignore
        raise AssertionError
    except ValueError as e:
        assert str(e).startswith("Invalid padding string 'blub'")
    args["padding"] = "same"

    try:
        _ = _VIConvNd(**args)  # type: ignore
        raise AssertionError
    except ValueError as e:
        assert str(e) == "padding='same' is not supported for strided convolutions"
    args["stride"] = (1,)

    try:
        _ = _VIConvNd(**args)  # type: ignore
        raise AssertionError
    except ValueError as e:
        assert str(e).startswith("padding_mode must be one of")
    args["padding_mode"] = "reflect"

    # Test correct argument parsing
    test1 = _VIConvNd(*args.values())  # type: ignore

    for key in args:
        if key == "bias":
            continue
        if key == "variational_distribution" or key == "prior":
            assert isinstance(test1.__dict__[key][0], type(args[key]))
            assert isinstance(test1.__dict__[key][1], type(args[key]))
        else:
            assert test1.__dict__[key] is args[key]

    assert test1._reversed_padding_repeated_twice == [1, 1]
    assert test1._weight_mean.shape == (3, 2, 3)
    assert test1._weight_log_std.shape == (3, 2, 3)
    assert test1._bias_mean.shape == (3,)
    assert test1._bias_log_std.shape == (3,)
    args["out_channels"] = 8
    args["kernel_size"] = (3, 5)
    args["transposed"] = True
    args["padding"] = (2,)
    args["groups"] = 2
    args["bias"] = False

    # Test passing of prior_initialization and return_log_probs
    test2 = _VIConvNd(
        *args.values(),  # type: ignore
        prior_initialization=not test1._prior_init,
        return_log_probs=not test1._return_log_probs,
    )
    assert test1._prior_init != test2._prior_init
    assert test1._return_log_probs != test2._return_log_probs
    assert test2._reversed_padding_repeated_twice == (2, 2)
    print(test2._weight_mean.shape)
    assert test2._weight_mean.shape == (2, 4, 3, 5)
    assert test2._weight_log_std.shape == (2, 4, 3, 5)
    assert not hasattr(test2, "_bias_mean")
    assert not hasattr(test2, "_bias_log_std")

    # Test __setstate__
    del test2.padding_mode
    test2.__setstate__(())
    assert test2.padding_mode == "zeros"


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
            assert isinstance(test1.__dict__[key][0], type(args[key]))
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
            if key in ["variational_distribution", "prior"]:
                assert isinstance(test1.__dict__[key][0], type(args[key]))
                assert len(test1.__dict__[key]) == 1
            else:
                assert test1.__dict__[key][0] is args[key]
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
            if key in ["variational_distribution", "prior"]:
                assert isinstance(test1.__dict__[key][0], type(args[key]))
                assert len(test1.__dict__[key]) == 1
            else:
                assert test1.__dict__[key][0] is args[key]
                assert len(test1.__dict__[key]) == 3
        else:
            assert test1.__dict__[key] is args[key]

    args["padding_mode"] = "zeros"
    args["bias"] = True
    test2 = VIConv3d(**args, return_log_probs=False)  # type: ignore

    out2 = test2(sample, samples=5)
    assert out2.shape == (5, 6, args["out_channels"], 3, 1, 4)
