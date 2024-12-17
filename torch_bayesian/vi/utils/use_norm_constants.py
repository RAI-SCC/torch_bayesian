from torch_bayesian.vi import _globals


def use_norm_constants(mode: bool = True) -> None:
    """
    Set global flag _USE_NORM_CONSTANTS.

    This flag makes all distribution add normalization constants during log_prob
    calculation. These constants are mathematically accurate, but not needed and
    seemingly counterproductive for training, possibly due to float accuracy.
    """
    _globals._USE_NORM_CONSTANTS = mode
