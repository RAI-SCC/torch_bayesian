from collections import OrderedDict
from typing import Tuple, Union

import pytest
import torch
from torch import Tensor
from torch.nn import ReLU

from torch_bayesian.vi import VILinear, VIModule, VIResidualConnection, VISequential


def test_sequential() -> None:
    """Test VISequential."""
    in_features = 2
    hidden_features = 4
    out_features = 3

    broke_module_dict = OrderedDict(
        module1=VILinear(in_features, hidden_features, return_log_probs=False),
        activation=ReLU(),
        module2=VILinear(hidden_features, out_features),
    )

    with pytest.raises(AssertionError, match="return_log_probs *"):
        _ = VISequential(broke_module_dict)

    module_dict = OrderedDict(
        module1=VILinear(in_features, hidden_features),
        activation=ReLU(),
        module2=VILinear(hidden_features, out_features),
    )

    model1 = VISequential(module_dict)
    assert model1._return_log_probs

    module_list = list(module_dict.values())
    module_list[0].return_log_probs(False)
    module_list[2].return_log_probs(False)

    model2 = VISequential(*module_list)
    assert not model2._return_log_probs

    for m1, m2 in zip(model1.modules(), model2.modules()):
        if isinstance(m1, VISequential) or isinstance(m1, ReLU):
            continue
        assert (m1._weight_mean == m2._weight_mean).all()

    sample = torch.randn(2, in_features)

    model1.return_log_probs()
    model2.return_log_probs()
    out1, _ = model1(sample, samples=5)
    out2, _ = model2(sample, samples=5)
    assert out1.shape == (5, 2, out_features)
    assert out2.shape == (5, 2, out_features)

    model1.return_log_probs(False)
    model2.return_log_probs(False)
    out1 = model1(sample, samples=4)
    out2 = model2(sample, samples=4)
    assert out1.shape == (4, 2, out_features)
    assert out2.shape == (4, 2, out_features)

    module3 = VISequential(ReLU(), ReLU())
    assert not module3._return_log_probs


def test_residual_connection() -> None:
    """Test VIResidualConnection."""

    class Test(VIModule):
        def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
            if self._return_log_probs:
                return x, torch.tensor([0.0, 1.0])
            else:
                return x

    class Test2(VIModule):
        def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
            if self._return_log_probs:
                return x.reshape((3, 6)), torch.tensor([0.0, 1.0])
            else:
                return x.reshape((2, 9))

    module = VIResidualConnection(Test())
    broken_module = VIResidualConnection(Test2())
    module.return_log_probs()
    sample1 = torch.randn(6, 3)
    out1, lps1 = module(sample1, samples=3)
    plp1 = lps1[:, 0]
    vlp1 = lps1[:, 1]
    try:
        broken_module(sample1, samples=3)
        raise AssertionError
    except RuntimeError as e:
        assert (
            str(e)
            == "Output shape (torch.Size([3, 6])) of residual connection must match input shape (torch.Size([6, 3]))"
        )

    assert torch.allclose(out1.mean(0), 2 * sample1)
    assert (plp1 == torch.zeros_like(plp1)).all()
    assert (vlp1 == torch.ones_like(vlp1)).all()

    module.return_log_probs(False)
    broken_module.return_log_probs(False)
    sample2 = torch.randn(3, 6)
    out2 = module(sample2, samples=5)
    assert torch.allclose(out2.mean(0), 2 * sample2)
    try:
        broken_module(sample2, samples=5)
        raise AssertionError
    except RuntimeError as e:
        assert (
            str(e)
            == "Output shape (torch.Size([2, 9])) of residual connection must match input shape (torch.Size([3, 6]))"
        )
