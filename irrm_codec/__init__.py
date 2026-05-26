"""IRRM-CODEC package."""

__all__ = ["ForwardModel", "InverseModel"]


def __getattr__(name):
    if name == "ForwardModel":
        from .forward_model import ForwardModel

        return ForwardModel
    if name == "InverseModel":
        from .inverse_model import InverseModel

        return InverseModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
