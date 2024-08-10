from vi.utils import ForceRequiredAttributeDefinitionMeta


def test_force_required_attribute_definition_meta() -> None:
    """Test ForceRequiredAttributeDefinitionMeta."""

    class Test(metaclass=ForceRequiredAttributeDefinitionMeta):
        a: int

        def check_required_attributes(self) -> None:
            if not hasattr(self, "a"):
                raise NotImplementedError("Test error")

    try:
        _ = Test()
        raise AssertionError
    except NotImplementedError as e:
        assert str(e) == "Test error"

    class Test2(Test):
        def __init__(self, a: int) -> None:
            self.a = a

    Test2(3)
