from __future__ import annotations

import argparse
import os


def _sanitize_thread_env() -> None:
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        value = os.environ.get(key)
        if value is None:
            continue
        if not value.isdigit() or int(value) <= 0:
            os.environ[key] = "1"


_sanitize_thread_env()

import torch

from vfm_loc.config import load_config
from vfm_loc.datasets import build_eval_datasets, build_eval_loaders, get_model_transforms
from vfm_loc.engine import format_metrics
from vfm_loc.models import build_model
from vfm_loc.zero_shot import evaluate_zero_shot


def main():
    parser = argparse.ArgumentParser(description="Evaluate VFM-Loc zero-shot release.")
    parser.add_argument("--config", required=True, help="Path to eval config yaml.")
    args = parser.parse_args()

    config = load_config(args.config)
    config.setdefault("evaluation", {})
    device = config["evaluation"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    config["evaluation"]["device"] = device
    config["model"]["device"] = device

    model = build_model(config["model"]).to(device)
    if torch.cuda.device_count() > 1 and len(config["evaluation"].get("gpu_ids", [0])) > 1:
        model = torch.nn.DataParallel(model, device_ids=config["evaluation"]["gpu_ids"])

    preprocess = model.module.get_preprocess_config() if isinstance(model, torch.nn.DataParallel) else model.get_preprocess_config()
    transforms = get_model_transforms(config, preprocess)
    query_ds, reference_ds = build_eval_datasets(config, transforms)
    query_loader, reference_loader = build_eval_loaders(config, query_ds, reference_ds)

    metrics = evaluate_zero_shot(config, model, query_loader, reference_loader)
    print(format_metrics(metrics))


if __name__ == "__main__":
    main()
