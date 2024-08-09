import torch
from torch import Tensor

from vi import VIModule


def test_normal_sample() -> None:
    """Test _normal_sample."""
    mean = torch.randn((3, 4))
    std = torch.zeros_like(mean)
    sample = VIModule._normal_sample(mean, std)
    assert sample.shape == mean.shape
    assert (sample == mean).all()

    std = torch.ones_like(mean)
    sample = VIModule._normal_sample(mean, std)
    assert not (sample == mean).all()


def test_expand_to_samples() -> None:
    """Test _expand_to_samples."""
    shape1 = (3, 4)
    sample1 = torch.randn(shape1)
    out1 = VIModule._expand_to_samples(sample1, samples=5)
    assert out1.shape == (5,) + shape1
    for s in out1:
        assert (s == sample1).all()

    shape2 = (5,)
    sample2 = torch.randn(shape2)
    out2 = VIModule._expand_to_samples(sample2, samples=1)
    assert out2.shape == (1,) + shape2
    for s in out2:
        assert (s == sample2).all()


def test_sampled_forward() -> None:
    """Test _sampled_forward."""

    class Test(VIModule):
        def __init__(self, ref: Tensor) -> None:
            super().__init__()
            self.ref = ref

        def forward(self, x: Tensor) -> Tensor:
            assert x.shape == self.ref.shape
            return x - self.ref

    shape1 = (3, 4)
    sample1 = torch.randn(shape1)
    test1 = Test(ref=sample1)
    assert (
        test1.sampled_forward(sample1, samples=5) == torch.zeros((5,) + shape1)
    ).all()

    shape2 = (5,)
    sample2 = torch.randn(shape2)
    test2 = Test(ref=sample2)
    assert (
        test2.sampled_forward(sample2, samples=1) == torch.zeros((1,) + shape2)
    ).all()
