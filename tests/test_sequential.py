from collections import OrderedDict

import torch
from torch.nn import ReLU

from vi import VILinear, VISequential


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
    out1, plp1, vlp1 = model1(sample, samples=5)
    out2, plp2, vlp2 = model2(sample, samples=5)
    assert out1.shape == (5, 2, out_features)
    assert out2.shape == (5, 2, out_features)

    model1.return_log_prob(False)
    model2.return_log_prob(False)
    out1 = model1(sample, samples=4)
    out2 = model2(sample, samples=4)
    assert out1.shape == (4, 2, out_features)
    assert out2.shape == (4, 2, out_features)
