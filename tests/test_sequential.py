from collections import OrderedDict
from typing import Tuple, Union

import torch
from torch import Tensor
from torch.nn import ReLU

from vi import VILinear, VIModule, VIResidualConnection, VISequential


def test_sequential() -> None:
    """Test VISequential."""
    in_features = 2
    hidden_features = 4
    out_features = 3

    module_dict = OrderedDict(
        module1=VILinear(in_features, hidden_features),
        activation=ReLU(),
        module2=VILinear(hidden_features, out_features),
    )

    module_list = list(module_dict.values())

    model1 = VISequential(module_dict)
    model2 = VISequential(*module_list)
    for m1, m2 in zip(model1.modules(), model2.modules()):
        if isinstance(m1, VISequential) or isinstance(m1, ReLU):
            continue
        assert (m1._weight_mean == m2._weight_mean).all()

    sample = torch.randn(2, in_features)

    model1.return_log_prob()
    model2.return_log_prob()
    out1, (plp1, vlp1) = model1(sample, samples=5)
    out2, (plp2, vlp2) = model2(sample, samples=5)
    assert out1.shape == (5, 2, out_features)
    assert out2.shape == (5, 2, out_features)

    model1.return_log_prob(False)
    model2.return_log_prob(False)
    out1 = model1(sample, samples=4)
    out2 = model2(sample, samples=4)
    assert out1.shape == (4, 2, out_features)
    assert out2.shape == (4, 2, out_features)


def test_residual_connection() -> None:
    """Test VIResidualConnection."""

    class Test(VIModule):
        def forward(
            self, x: Tensor
        ) -> Union[Tensor, Tuple[Tensor, Tuple[Tensor, Tensor]]]:
            if self._return_log_prob:
                return x, (torch.tensor(0.0), torch.tensor(1.0))
            else:
                return x

    class Test2(VIModule):
        def forward(
            self, x: Tensor
        ) -> Union[Tensor, Tuple[Tensor, Tuple[Tensor, Tensor]]]:
            if self._return_log_prob:
                return x.reshape((3, 6)), (torch.tensor(0.0), torch.tensor(1.0))
            else:
                return x.reshape((2, 9))

    module = VIResidualConnection(Test())
    broken_module = VIResidualConnection(Test2())
    module.return_log_prob()
    sample1 = torch.randn(6, 3)
    out1, (plp1, vlp1) = module(sample1, samples=3)
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

    module.return_log_prob(False)
    broken_module.return_log_prob(False)
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
