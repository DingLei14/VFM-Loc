from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config file must contain a dict at top level: {path}")
    return data


def _resolve_bases(path: Path, data: Dict[str, Any]) -> Dict[str, Any]:
    base_entry = data.pop("_base_", None)
    if base_entry is None:
        return data

    if isinstance(base_entry, str):
        base_paths = [base_entry]
    elif isinstance(base_entry, list):
        base_paths = base_entry
    else:
        raise TypeError(f"_base_ must be str or list[str], got {type(base_entry)}")

    merged: Dict[str, Any] = {}
    for base_name in base_paths:
        base_path = (path.parent / base_name).resolve()
        base_cfg = _load_yaml(base_path)
        base_cfg = _resolve_bases(base_path, base_cfg)
        merged = _deep_merge(merged, base_cfg)
    return _deep_merge(merged, data)


PATH_KEYS = {
    "config_path",
    "repo_root",
    "data_root",
    "output_dir",
    "gps_dict_path",
    "checkpoint",
    "checkpoints",
    "weights",
    "bpe_path",
}


def _resolve_path(value: Any, config_dir: Path) -> Any:
    if not isinstance(value, str):
        return value
    if value.startswith(("http://", "https://")):
        return value
    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate)
    direct = (config_dir / candidate).resolve()
    if direct.exists():
        return str(direct)
    return str((ROOT / candidate).resolve())


def _resolve_paths_inplace(node: Any, config_dir: Path, parent_key: str | None = None) -> Any:
    if isinstance(node, dict):
        return {
            key: _resolve_paths_inplace(value, config_dir, parent_key=key)
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [_resolve_paths_inplace(value, config_dir, parent_key=parent_key) for value in node]
    if parent_key in PATH_KEYS:
        return _resolve_path(node, config_dir)
    return node


def load_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path).resolve()
    data = _load_yaml(path)
    data = _resolve_bases(path, data)
    data["config_path"] = str(path)
    data["repo_root"] = str(ROOT)
    return _resolve_paths_inplace(data, path.parent)
