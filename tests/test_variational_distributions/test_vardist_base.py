from torch import Tensor

from vi.variational_distributions import VariationalDistribution


def test_parameter_checking() -> None:
    """Test enforcement or required parameters for subclasses of VariationalDistribution."""

    # variational_parameters assertion
    class Test(VariationalDistribution):
        pass

    try:
        Test()
        raise AssertionError
    except NotImplementedError as e:
        assert str(e) == "Subclasses must define variational_parameters"

    # _default_variational_parameters assertion
    class Test1(VariationalDistribution):
        variational_parameters = ("mean", "std")

    try:
        Test1()
        raise AssertionError
    except NotImplementedError as e:
        assert str(e) == "Subclasses must define _default_variational_parameters"

    # length matching of variational_parameters and default parameters
    class Test2(VariationalDistribution):
        variational_parameters = ("mean", "std")
        _default_variational_parameters = (0.0,)

    try:
        Test2()
        raise AssertionError
    except AssertionError as e:
        assert str(e) == "Each variational parameter must be assigned a default value"

    # sample assertion
    class Test3(VariationalDistribution):
        variational_parameters = ("mean", "std")
        _default_variational_parameters = (0.0, 1.0)

    try:
        Test3()
        raise AssertionError
    except NotImplementedError as e:
        assert str(e) == "Subclasses must define the sample method"

    # sample assertion
    class Test4(VariationalDistribution):
        variational_parameters = ("mean", "std")
        _default_variational_parameters = (0.0, 1.0)

        def sample(self, mean: Tensor) -> Tensor:
            return mean

    try:
        Test4()
        raise AssertionError
    except AssertionError as e:
        assert (
            str(e)
            == "Sample must accept exactly one Tensor for each variational parameter"
        )

    class Test5(VariationalDistribution):
        variational_parameters = ("mean", "std")
        _default_variational_parameters = (0.0, 1.0)

        def sample(self, mean: Tensor, std: Tensor) -> Tensor:
            return mean + std

    try:
        Test5()
        raise AssertionError
    except NotImplementedError as e:
        assert str(e) == "Subclasses must define log_prob"

    class Test6(VariationalDistribution):
        variational_parameters = ("mean", "std")
        _default_variational_parameters = (0.0, 1.0)

        def sample(self, mean: Tensor, std: Tensor) -> Tensor:
            return mean + std

        def log_prob(self, mean: Tensor, std: Tensor) -> Tensor:
            return mean + std

    try:
        Test6()
        raise AssertionError
    except AssertionError as e:
        assert (
            str(e)
            == "log_prob must accept an argument for each variational parameter plus the sample"
        )

    class Test7(VariationalDistribution):
        variational_parameters = ("mean", "std")
        _default_variational_parameters = (0.0, 1.0)

        def sample(self, mean: Tensor, std: Tensor) -> Tensor:
            return mean + std

        def log_prob(self, sample: Tensor, mean: Tensor, std: Tensor) -> Tensor:
            return sample + mean + std

    _ = Test7()


def test_match_parameters() -> None:
    """Test VariationalDistribution.match_parameters()."""

    class Test(VariationalDistribution):
        variational_parameters = ("mean", "std")
        _default_variational_parameters = (0.0, 1.0)

        def sample(self, mean: Tensor, std: Tensor) -> Tensor:
            return mean + std

        def log_prob(self, sample: Tensor, mean: Tensor, std: Tensor) -> Tensor:
            return sample + mean + std

    test = Test()

    ref1 = ("mean",)
    shared, diff = test.match_parameters(ref1)
    assert shared == {"mean": 0}
    assert diff == {"std": 1}

    ref2 = ("std", "mean", "skew")
    shared, diff = test.match_parameters(ref2)
    assert shared == {"mean": 0, "std": 1}
    assert diff == {}
