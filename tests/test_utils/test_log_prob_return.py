import torch

from vi.utils import to_log_prob_return_format


def test_to_log_prob_return_format() -> None:
    """Test conversion to log prob return format."""
    out = torch.randn((3, 5))
    plp = torch.randn(7)
    vlp = torch.randn((9, 4, 16))

    lp_out = to_log_prob_return_format(out, plp, vlp)
    assert torch.equal(lp_out[0], out)
    assert torch.equal(lp_out[1][0], plp)
    assert torch.equal(lp_out[1][1], vlp)
