from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from mf.config.overrides import apply_overrides
from mf.config.schema import MFConfig


class _UniqueKeyLoader(yaml.SafeLoader):
    source_path: Path


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            mark = key_node.start_mark
            raise ValueError(
                f"unhashable YAML key at {loader.source_path}:{mark.line + 1}:{mark.column + 1}"
            ) from error
        if duplicate:
            mark = key_node.start_mark
            raise ValueError(
                f"duplicate YAML key {key!r} at {loader.source_path}:"
                f"{mark.line + 1}:{mark.column + 1}"
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_yaml_document(path: Path) -> object:
    loader = _UniqueKeyLoader(path.read_text(encoding="utf-8"))
    loader.source_path = path
    try:
        return loader.get_single_data()
    finally:
        loader.dispose()


def _merge_mappings(
    base: Mapping[str, object],
    override: Mapping[str, object],
) -> dict[str, object]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(value, Mapping) and value.get("__replace__") is True:
            replacement = dict(value)
            replacement.pop("__replace__")
            merged[key] = replacement
            continue
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_mappings(current, value)
        else:
            merged[key] = value
    return merged


def _configuration_root(path: Path) -> Path:
    for parent in (path.parent, *path.parents):
        if parent.name == "configs":
            return parent.resolve()
    return path.parent.parent.resolve()


def _load_mapping(
    path: Path,
    stack: tuple[Path, ...] = (),
    root: Path | None = None,
) -> dict[str, object]:
    if path.is_symlink():
        raise ValueError(f"configuration must not be a symlink: {path}")
    resolved = path.expanduser().resolve()
    root = _configuration_root(resolved) if root is None else root
    if not resolved.is_relative_to(root):
        raise ValueError(
            f"configuration inheritance escapes configuration root: {resolved}"
        )
    if resolved in stack:
        chain = " -> ".join(str(item) for item in (*stack, resolved))
        raise ValueError(f"configuration inheritance cycle: {chain}")

    raw = load_yaml_document(resolved)
    if not isinstance(raw, Mapping):
        raise ValueError(f"configuration root must be a mapping: {resolved}")
    child = dict(raw)
    parent_ref = child.pop("extends", None)
    if parent_ref is None:
        return child
    if not isinstance(parent_ref, str) or not parent_ref:
        raise ValueError("configuration extends must be a non-empty path string")

    parent_path = Path(parent_ref)
    if parent_path.is_absolute():
        raise ValueError("configuration extends must use a relative path")
    parent_path = resolved.parent / parent_path
    parent = _load_mapping(parent_path, (*stack, resolved), root)
    return _merge_mappings(parent, child)


def resolve_config_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_file() or path.is_symlink():
        return path
    if not path.is_absolute() and path.parent == Path("configs"):
        for parent in Path(__file__).resolve().parents:
            installed = parent / "share/multimodal-flow" / path
            if installed.is_file():
                return installed
    raise FileNotFoundError(f"configuration file not found: {path}")


def load_config(path: Path, overrides: Sequence[str]) -> MFConfig:
    path = resolve_config_path(path)
    raw = _load_mapping(path)
    resolved = apply_overrides(raw, overrides, MFConfig)
    return MFConfig.model_validate(resolved)
