from torch import Tensor

from vi.priors import Prior


def test_parameter_checking() -> None:
    """Test enforcement or required parameters for subclasses of Prior."""

    # distribution_parameters assertion
    class Test1(Prior):
        pass

    try:
        Test1()
        raise AssertionError
    except NotImplementedError as e:
        assert str(e) == "Subclasses must define distribution_parameters"

    # log_prob assertion
    class Test2(Prior):
        distribution_parameters = ("mean", "log_std")

    try:
        Test2()
        raise AssertionError
    except NotImplementedError as e:
        assert str(e) == "Subclasses must define log_prob"

    # log_prob signature assertion
    class Test3(Prior):
        distribution_parameters = ("mean", "log_std")
        _required_parameters = ("mean",)

        def log_prob(self, x: Tensor) -> Tensor:
            pass

    try:
        Test3()
        raise AssertionError
    except AssertionError as e:
        assert (
            str(e)
            == "log_prob must accept an argument for each required parameter plus the sample"
        )

    class Test4(Prior):
        distribution_parameters = ("mean", "log_std")

        def log_prob(self, x: Tensor, mean: Tensor) -> Tensor:
            pass

    try:
        Test4()
        raise AssertionError
    except AssertionError as e:
        assert (
            str(e)
            == "log_prob must accept an argument for each required parameter plus the sample"
        )

    # Test correct initialization
    class Test5(Prior):
        distribution_parameters = ("mean", "log_std")
        _required_parameters = ("mean",)

        def log_prob(self, x: Tensor, mean: Tensor) -> Tensor:
            pass

    _ = Test5()

    class Test6(Prior):
        distribution_parameters = ("mean", "log_std")

        def log_prob(self, x: Tensor) -> Tensor:
            pass

    _ = Test6()


def test_match_parameters() -> None:
    """Test Prior.match_parameters()."""

    class Test(Prior):
        distribution_parameters = ("mean", "log_std")
        _required_parameters = ("mean",)

        def log_prob(self, x: Tensor, mean: Tensor) -> Tensor:
            pass

    test = Test()

    ref1 = ("mean",)
    shared, diff = test.match_parameters(ref1)
    assert shared == {"mean": 0}
    assert diff == {"log_std": 1}

    ref2 = ("log_std", "mean", "skew")
    shared, diff = test.match_parameters(ref2)
    assert shared == {"mean": 0, "log_std": 1}
    assert diff == {}
