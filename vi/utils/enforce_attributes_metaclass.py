from typing import Any, Type, TypeVar

T = TypeVar("T")


class ForceRequiredAttributeDefinitionMeta(type):
    """Metaclass to enforce definition of all attributes."""

    def __call__(cls: Type[T], *args: Any, **kwargs: Any) -> T:
        """Run __call__ as usual, then check for required attributes."""
        class_object = type.__call__(cls, *args, **kwargs)
        class_object.check_required_attributes()
        return class_object
