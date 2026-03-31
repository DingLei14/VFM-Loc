from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from .engine import evaluate_retrieval, predict
from .utils import canonical_query_labels


@torch.no_grad()
def pca_fit(features: torch.Tensor):
    x = features.to(torch.float32)
    mean = x.mean(dim=0)
    centered = x - mean
    cov = (centered.T @ centered) / max(1, centered.shape[0] - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    return mean, eigvecs[:, order]


@torch.no_grad()
def pca_project(features: torch.Tensor, mean: torch.Tensor, proj: torch.Tensor, dim: int):
    return (features.to(torch.float32) - mean) @ proj[:, : min(dim, proj.shape[1])]


@torch.no_grad()
def mean_by_id(features: torch.Tensor, labels: torch.Tensor):
    labels_cpu = labels.cpu()
    uniq = torch.unique(labels_cpu)
    uniq, _ = torch.sort(uniq)
    pooled = []
    for uid in uniq.tolist():
        pooled.append(features[(labels_cpu == uid).to(features.device)].mean(dim=0))
    return uniq.to(features.device), torch.stack(pooled, dim=0)


@torch.no_grad()
def procrustes_align(query_anchors: torch.Tensor, reference_anchors: torch.Tensor):
    qn = F.normalize(query_anchors, dim=-1)
    rn = F.normalize(reference_anchors, dim=-1)
    matrix = qn.T @ rn
    u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    return u @ vh


def build_rmac_weights(levels, device, dtype, alpha: float = 2.0):
    weights = []
    for level in levels:
        per_region = 1.0 / float(level**alpha)
        weights.extend([per_region] * (level * level))
    return torch.tensor(weights, device=device, dtype=dtype)


@torch.no_grad()
def evaluate_zero_shot(config: Dict, model, query_loader, reference_loader):
    evaluation = config["evaluation"]
    alignment = config.get("alignment", {})
    normalize_global = bool(evaluation.get("normalize_features", True))
    query_features, query_labels = predict(config, model, query_loader, normalize=normalize_global)
    reference_features, reference_labels = predict(config, model, reference_loader, normalize=normalize_global)

    if query_features.ndim == 3:
        levels = config["model"].get("rmac_levels", [1, 2])
        alpha = float(alignment.get("rmac_alpha", 2.0))
        pca_dim = int(alignment.get("pca_dim", min(256, query_features.shape[-1])))

        q_flat = query_features.reshape(-1, query_features.shape[-1])
        r_flat = reference_features.reshape(-1, reference_features.shape[-1])
        q_mean, q_proj = pca_fit(q_flat)
        r_mean, r_proj = pca_fit(r_flat)
        qp = pca_project(q_flat, q_mean, q_proj, pca_dim).reshape(query_features.shape[0], query_features.shape[1], pca_dim)
        rp = pca_project(r_flat, r_mean, r_proj, pca_dim).reshape(reference_features.shape[0], reference_features.shape[1], pca_dim)

        weights = build_rmac_weights(levels, device=qp.device, dtype=qp.dtype, alpha=alpha)
        weights_sum = weights.sum().clamp_min(1e-6)
        query_anchor = (qp * weights.view(1, -1, 1)).sum(dim=1) / weights_sum
        reference_anchor = (rp * weights.view(1, -1, 1)).sum(dim=1) / weights_sum

        query_ids = canonical_query_labels(query_labels)
        ref_ids = reference_labels
        q_unique, q_mean_anchor = mean_by_id(query_anchor, query_ids)
        r_unique, r_mean_anchor = mean_by_id(reference_anchor, ref_ids)
        common = sorted(set(q_unique.cpu().tolist()).intersection(set(r_unique.cpu().tolist())))
        if alignment.get("use_procrustes", True) and len(common) >= 2:
            q_map = {int(label.item()): idx for idx, label in enumerate(q_unique.cpu())}
            r_map = {int(label.item()): idx for idx, label in enumerate(r_unique.cpu())}
            q_pairs = torch.stack([q_mean_anchor[q_map[item]] for item in common], dim=0)
            r_pairs = torch.stack([r_mean_anchor[r_map[item]] for item in common], dim=0)
            rotation = procrustes_align(q_pairs, r_pairs)
            qp = qp @ rotation

        query_vectors = F.normalize((F.normalize(qp, dim=-1) * weights.view(1, -1, 1)).sum(dim=1) / weights_sum, dim=-1)
        reference_vectors = F.normalize((F.normalize(rp, dim=-1) * weights.view(1, -1, 1)).sum(dim=1) / weights_sum, dim=-1)
    else:
        pca_dim = int(alignment.get("pca_dim", min(256, query_features.shape[-1])))
        q_mean, q_proj = pca_fit(query_features)
        r_mean, r_proj = pca_fit(reference_features)
        query_vectors = pca_project(query_features, q_mean, q_proj, pca_dim)
        reference_vectors = pca_project(reference_features, r_mean, r_proj, pca_dim)

        if alignment.get("use_procrustes", True):
            query_ids = canonical_query_labels(query_labels)
            q_unique, q_anchors = mean_by_id(query_vectors, query_ids)
            r_unique, r_anchors = mean_by_id(reference_vectors, reference_labels)
            common = sorted(set(q_unique.cpu().tolist()).intersection(set(r_unique.cpu().tolist())))
            if len(common) >= 2:
                q_map = {int(label.item()): idx for idx, label in enumerate(q_unique.cpu())}
                r_map = {int(label.item()): idx for idx, label in enumerate(r_unique.cpu())}
                q_pairs = torch.stack([q_anchors[q_map[item]] for item in common], dim=0)
                r_pairs = torch.stack([r_anchors[r_map[item]] for item in common], dim=0)
                rotation = procrustes_align(q_pairs, r_pairs)
                query_vectors = query_vectors @ rotation

        query_vectors = F.normalize(query_vectors, dim=-1)
        reference_vectors = F.normalize(reference_vectors, dim=-1)

    return evaluate_retrieval(
        query_features=query_vectors,
        reference_features=reference_vectors,
        query_labels=query_labels,
        reference_labels=reference_labels,
        ranks=evaluation.get("ranks", [1, 5, 10]),
        step_size=int(evaluation.get("step_size", 256)),
    )
