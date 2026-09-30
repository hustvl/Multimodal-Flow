"""Built-in generation adapters for the registry-backed physical API."""

from __future__ import annotations

from mf.registries import register_generation


def _physical_forward(pipeline: object, request: object, **kwargs: object) -> object:
    routing_layout = kwargs.pop("routing_layout", None)
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"physical_forward does not accept options: {unknown}")
    return pipeline.forward_physical(
        request.physical_layout,
        routing_layout=routing_layout,
    )


register_generation("physical_forward", _physical_forward, replace=True)


__all__ = []
