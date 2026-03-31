from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch

from .config import ROOT


def ensure_vendored_package(package_name: str, relative_root: str) -> None:
    try:
        __import__(package_name)
        return
    except Exception:
        package_root = ROOT / relative_root
        if package_root.exists():
            package_root_str = str(package_root)
            if package_root_str not in sys.path:
                sys.path.insert(0, package_root_str)


def resolve_checkpoint_path(path: Optional[str], extra_candidates: Optional[Iterable[str]] = None) -> Optional[str]:
    candidates = []
    if path:
        candidates.append(Path(path))
        candidates.append(ROOT / path)
        candidates.append(ROOT / "pretrained_weights" / path)
        candidates.append(ROOT / "pretrained_weights" / Path(path).name)
    for candidate in extra_candidates or []:
        p = Path(candidate)
        candidates.append(p)
        candidates.append(ROOT / candidate)
        candidates.append(ROOT / "pretrained_weights" / candidate)

    seen = set()
    for candidate in candidates:
        resolved = str(candidate.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.exists():
            return resolved
    return None


def setup_system(seed: int, cudnn_benchmark: bool = True, cudnn_deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic


def select_device(device: Optional[str]) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.val = float(value)
        self.sum += float(value) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_state_dict(model: torch.nn.Module, checkpoint_path: str, strict: bool = False) -> None:
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state_dict, dict):
        if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict and isinstance(state_dict["model"], dict):
            state_dict = state_dict["model"]
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(state_dict)}")
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key[len("module.") :]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=strict)


def canonical_query_labels(labels: torch.Tensor) -> torch.Tensor:
    if labels.ndim == 1:
        return labels
    return labels[:, 0]


def label_sets(labels: torch.Tensor) -> list[set[int]]:
    if labels.ndim == 1:
        return [{int(value)} for value in labels.cpu().tolist()]
    return [
        {int(v) for v in row if int(v) >= 0}
        for row in labels.cpu().tolist()
    ]
