from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm

from .utils import label_sets


def _amp_context(device: str, enabled: bool):
    return autocast(device_type="cuda" if str(device).startswith("cuda") else "cpu", enabled=enabled)


def predict(config: Dict, model, dataloader, normalize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    evaluation = config.get("evaluation", {})
    device = evaluation["device"]
    verbose = bool(evaluation.get("verbose", True))
    mixed_precision = bool(config.get("evaluation", {}).get("mixed_precision", True))
    model.eval()
    features, labels = [], []
    progress = tqdm(dataloader, total=len(dataloader), disable=not verbose)
    for img, batch_labels in progress:
        img = img.to(device, non_blocking=True)
        with _amp_context(device, mixed_precision):
            feats = model(img)
            if normalize and feats.ndim == 2:
                feats = F.normalize(feats, dim=-1)
        features.append(feats.to(torch.float32))
        labels.append(batch_labels)
    progress.close()
    return torch.cat(features, dim=0), torch.cat(labels, dim=0).to(device)


@torch.no_grad()
def evaluate_retrieval(
    query_features: torch.Tensor,
    reference_features: torch.Tensor,
    query_labels: torch.Tensor,
    reference_labels: torch.Tensor,
    ranks: Iterable[int],
    step_size: int = 256,
) -> Dict[str, float]:
    ranks = sorted(set(int(rank) for rank in ranks))
    top1p = max(1, reference_features.size(0) // 100)
    max_k = max(max(ranks), top1p)
    positives = label_sets(query_labels)
    ref_labels_list = reference_labels.cpu().tolist()
    ref_label_set = set(ref_labels_list)
    matchable_queries = sum(bool(pos.intersection(ref_label_set)) for pos in positives)
    if matchable_queries == 0:
        raise RuntimeError("No query labels overlap with gallery labels. Check dataset ID matching.")
    correct = {rank: 0 for rank in ranks}
    correct[top1p] = 0
    ap_sum = 0.0
    hit_rate = 0.0
    total = query_features.size(0)

    for start in range(0, total, step_size):
        end = min(total, start + step_size)
        sim = query_features[start:end] @ reference_features.T
        sorted_idx = torch.argsort(sim, dim=1, descending=True)
        topk_idx = sorted_idx[:, :max_k]
        topk_labels = reference_labels[topk_idx].cpu().tolist()

        for offset, ranked_indices in enumerate(sorted_idx.cpu().tolist()):
            query_pos = positives[start + offset]
            ranked_labels = [ref_labels_list[idx] for idx in ranked_indices]
            matches = [label in query_pos for label in ranked_labels]
            for rank in ranks:
                if any(matches[:rank]):
                    correct[rank] += 1
            if any(matches[:top1p]):
                correct[top1p] += 1

            relevant_positions = [idx + 1 for idx, matched in enumerate(matches) if matched]
            if relevant_positions:
                precision_sum = 0.0
                for position in relevant_positions:
                    precision_sum += sum(matches[:position]) / position
                ap_sum += precision_sum / len(relevant_positions)

            local_topk = topk_labels[offset]
            negatives = [label for label in local_topk if label not in query_pos]
            if not negatives:
                hit_rate += 1.0

    metrics = {f"recall@{rank}": correct[rank] / total * 100.0 for rank in ranks}
    metrics["recall@top1%"] = correct[top1p] / total * 100.0
    metrics["mAP"] = ap_sum / total * 100.0
    metrics["hit_rate"] = hit_rate / total * 100.0
    return metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    ordered_keys = [key for key in ["recall@1", "recall@5", "recall@10", "recall@top1%", "mAP", "hit_rate"] if key in metrics]
    return " - ".join(f"{key}: {metrics[key]:.2f}" for key in ordered_keys)
