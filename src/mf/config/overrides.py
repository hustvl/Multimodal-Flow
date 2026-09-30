from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from copy import deepcopy
from types import UnionType
from typing import Annotated, Union, get_args, get_origin

import yaml
from pydantic import BaseModel


class OverrideError(ValueError):
    pass


def parse_override(text: str) -> tuple[tuple[str, ...], object]:
    if "=" not in text:
        raise OverrideError(f"override must use path=value syntax: {text!r}")

    path_text, raw = text.split("=", 1)
    path = tuple(path_text.split("."))
    if not path_text or any(not component for component in path):
        raise OverrideError(
            f"override path must contain non-empty components: {text!r}"
        )

    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise OverrideError(
            f"invalid YAML override value for {path_text!r}: {raw!r}"
        ) from exc

    return path, value


def _nested_model_schemas(annotation: object) -> tuple[type[BaseModel], ...]:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return (annotation,)
    origin = get_origin(annotation)
    if origin is Annotated:
        return _nested_model_schemas(get_args(annotation)[0])
    if origin not in (Union, UnionType):
        return ()
    arguments = tuple(
        argument for argument in get_args(annotation) if argument is not type(None)
    )
    models = tuple(
        argument
        for argument in arguments
        if isinstance(argument, type) and issubclass(argument, BaseModel)
    )
    return models if len(models) == len(arguments) else ()


def _validate_override_path(
    schema: type[BaseModel],
    path: tuple[str, ...],
) -> None:
    current_schemas = (schema,)

    for index, component in enumerate(path):
        fields = tuple(
            field
            for candidate in current_schemas
            if (field := candidate.model_fields.get(component)) is not None
        )
        if not fields:
            raise OverrideError(f"unknown override path: {'.'.join(path)}")
        if index == len(path) - 1:
            return

        nested = tuple(
            dict.fromkeys(
                model
                for field in fields
                for model in _nested_model_schemas(field.annotation)
            )
        )
        if not nested:
            raise OverrideError(f"cannot traverse override path: {'.'.join(path)}")
        current_schemas = nested


def apply_overrides(
    data: Mapping[str, object],
    overrides: Sequence[str],
    schema: type[BaseModel],
) -> dict[str, object]:
    resolved = deepcopy(dict(data))

    for text in overrides:
        path, value = parse_override(text)
        _validate_override_path(schema, path)
        current: object = resolved

        for component in path[:-1]:
            if not isinstance(current, MutableMapping):
                raise OverrideError(f"cannot traverse override path: {'.'.join(path)}")
            if component not in current:
                current[component] = {}
            current = current[component]

        if not isinstance(current, MutableMapping):
            raise OverrideError(f"cannot traverse override path: {'.'.join(path)}")

        current[path[-1]] = value

    return resolved
