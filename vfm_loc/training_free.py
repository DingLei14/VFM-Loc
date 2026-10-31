from __future__ import annotations

from typing import Dict, Optional, Tuple

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
    kept = []
    for uid in uniq.tolist():
        if int(uid) < 0:
            continue
        kept.append(uid)
        pooled.append(features[(labels_cpu == uid).to(features.device)].mean(dim=0))
    if not pooled:
        empty = features.new_zeros((0, features.shape[-1]))
        return uniq.new_tensor([], dtype=uniq.dtype).to(features.device), empty
    return torch.tensor(kept, device=features.device, dtype=uniq.dtype), torch.stack(pooled, dim=0)


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
def _fit_procrustes_rotation(
    query_anchor: torch.Tensor,
    reference_anchor: torch.Tensor,
    query_labels: torch.Tensor,
    reference_labels: torch.Tensor,
    dim: int,
    *,
    use_procrustes: bool,
) -> Tuple[torch.Tensor, int]:
    q_unique, q_mean_anchor = mean_by_id(query_anchor, canonical_query_labels(query_labels))
    r_unique, r_mean_anchor = mean_by_id(reference_anchor, reference_labels)
    common = sorted(set(q_unique.cpu().tolist()).intersection(set(r_unique.cpu().tolist())))
    if use_procrustes and len(common) >= 2:
        q_map = {int(label.item()): idx for idx, label in enumerate(q_unique.cpu())}
        r_map = {int(label.item()): idx for idx, label in enumerate(r_unique.cpu())}
        q_pairs = torch.stack([q_mean_anchor[q_map[item]] for item in common], dim=0)
        r_pairs = torch.stack([r_mean_anchor[r_map[item]] for item in common], dim=0)
        return procrustes_align(q_pairs, r_pairs), len(common)
    return torch.eye(dim, device=query_anchor.device, dtype=query_anchor.dtype), len(common)


@torch.no_grad()
def align_features(
    query_features: torch.Tensor,
    reference_features: torch.Tensor,
    query_labels: torch.Tensor,
    reference_labels: torch.Tensor,
    config: Dict,
    *,
    pretrained: Optional[Dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """统一对齐流程，与 evaluate_training_free / fit_alignment / eval_pretrained 共用。

    pretrained is None:
        在当前特征上拟合 PCA + Procrustes（oracle，等同 eval.py）。
    pretrained 提供 q_mean/q_proj/r_mean/r_proj/rotation:
        使用预存 PCA + rotation_pca，不在 test 上重算。
    """
    alignment = config.get("alignment", {})
    use_procrustes = bool(alignment.get("use_procrustes", True))

    if query_features.ndim == 3:
        levels = config["model"].get("rmac_levels", [1, 2])
        alpha = float(alignment.get("rmac_alpha", 2.0))
        feature_dim = query_features.shape[-1]
        pca_dim = int(
            (pretrained or {}).get("pca_dim")
            or alignment.get("pca_dim", min(256, feature_dim))
        )

        q_flat = query_features.reshape(-1, feature_dim)
        r_flat = reference_features.reshape(-1, feature_dim)

        if pretrained is None:
            q_mean, q_proj = pca_fit(q_flat)
            r_mean, r_proj = pca_fit(r_flat)
        else:
            device = query_features.device
            q_mean = pretrained["q_mean"].to(device)
            q_proj = pretrained["q_proj"].to(device)
            r_mean = pretrained["r_mean"].to(device)
            r_proj = pretrained["r_proj"].to(device)
            rotation = pretrained["rotation"].to(device)
            if rotation.shape != (pca_dim, pca_dim):
                raise ValueError(
                    f"预存 rotation shape {tuple(rotation.shape)} 与 pca_dim={pca_dim} 不一致；"
                    f"PCA 与 rotation 必须成套迁移，请重新运行 fit_alignment.py。"
                )

        qp = pca_project(q_flat, q_mean, q_proj, pca_dim).reshape(
            query_features.shape[0], query_features.shape[1], pca_dim
        )
        rp = pca_project(r_flat, r_mean, r_proj, pca_dim).reshape(
            reference_features.shape[0], reference_features.shape[1], pca_dim
        )

        weights = build_rmac_weights(levels, device=qp.device, dtype=qp.dtype, alpha=alpha)
        weights_sum = weights.sum().clamp_min(1e-6)

        if pretrained is None:
            query_anchor = (qp * weights.view(1, -1, 1)).sum(dim=1) / weights_sum
            reference_anchor = (rp * weights.view(1, -1, 1)).sum(dim=1) / weights_sum
            rotation, _ = _fit_procrustes_rotation(
                query_anchor,
                reference_anchor,
                query_labels,
                reference_labels,
                pca_dim,
                use_procrustes=use_procrustes,
            )
        qp = qp @ rotation

        query_vectors = F.normalize(
            (F.normalize(qp, dim=-1) * weights.view(1, -1, 1)).sum(dim=1) / weights_sum, dim=-1
        )
        reference_vectors = F.normalize(
            (F.normalize(rp, dim=-1) * weights.view(1, -1, 1)).sum(dim=1) / weights_sum, dim=-1
        )
        return query_vectors, reference_vectors

    pca_dim = int(
        (pretrained or {}).get("pca_dim")
        or alignment.get("pca_dim", min(256, query_features.shape[-1]))
    )

    if pretrained is None:
        q_mean, q_proj = pca_fit(query_features)
        r_mean, r_proj = pca_fit(reference_features)
    else:
        device = query_features.device
        q_mean = pretrained["q_mean"].to(device)
        q_proj = pretrained["q_proj"].to(device)
        r_mean = pretrained["r_mean"].to(device)
        r_proj = pretrained["r_proj"].to(device)
        rotation = pretrained["rotation"].to(device)
        if rotation.shape != (pca_dim, pca_dim):
            raise ValueError(
                f"预存 rotation shape {tuple(rotation.shape)} 与 pca_dim={pca_dim} 不一致。"
            )

    query_vectors = pca_project(query_features, q_mean, q_proj, pca_dim)
    reference_vectors = pca_project(reference_features, r_mean, r_proj, pca_dim)

    if pretrained is None:
        rotation, _ = _fit_procrustes_rotation(
            query_vectors,
            reference_vectors,
            query_labels,
            reference_labels,
            pca_dim,
            use_procrustes=use_procrustes,
        )
    query_vectors = query_vectors @ rotation

    query_vectors = F.normalize(query_vectors, dim=-1)
    reference_vectors = F.normalize(reference_vectors, dim=-1)
    return query_vectors, reference_vectors


@torch.no_grad()
def fit_alignment_params(
    query_features: torch.Tensor,
    reference_features: torch.Tensor,
    query_labels: torch.Tensor,
    reference_labels: torch.Tensor,
    config: Dict,
    meta: Optional[Dict] = None,
) -> Dict:
    """在当前数据上拟合对齐参数，流程与 align_features(oracle) 完全一致。"""
    alignment = config.get("alignment", {})
    use_procrustes = bool(alignment.get("use_procrustes", True))
    params: Dict = {"ndim": query_features.ndim}
    if meta:
        params["meta"] = meta

    if query_features.ndim == 3:
        levels = config["model"].get("rmac_levels", [1, 2])
        alpha = float(alignment.get("rmac_alpha", 2.0))
        feature_dim = query_features.shape[-1]
        pca_dim = int(alignment.get("pca_dim", min(256, feature_dim)))

        q_flat = query_features.reshape(-1, feature_dim)
        r_flat = reference_features.reshape(-1, feature_dim)
        q_mean, q_proj = pca_fit(q_flat)
        r_mean, r_proj = pca_fit(r_flat)

        qp = pca_project(q_flat, q_mean, q_proj, pca_dim).reshape(
            query_features.shape[0], query_features.shape[1], pca_dim
        )
        rp = pca_project(r_flat, r_mean, r_proj, pca_dim).reshape(
            reference_features.shape[0], reference_features.shape[1], pca_dim
        )

        weights = build_rmac_weights(levels, device=qp.device, dtype=qp.dtype, alpha=alpha)
        weights_sum = weights.sum().clamp_min(1e-6)
        query_anchor = (qp * weights.view(1, -1, 1)).sum(dim=1) / weights_sum
        reference_anchor = (rp * weights.view(1, -1, 1)).sum(dim=1) / weights_sum
        rotation, n_common = _fit_procrustes_rotation(
            query_anchor,
            reference_anchor,
            query_labels,
            reference_labels,
            pca_dim,
            use_procrustes=use_procrustes,
        )
        params.update(
            {
                "pca_dim": pca_dim,
                "feature_dim": feature_dim,
                "q_mean": q_mean.cpu(),
                "q_proj": q_proj.cpu(),
                "r_mean": r_mean.cpu(),
                "r_proj": r_proj.cpu(),
                "rotation": rotation.cpu(),
                "num_common_ids": n_common,
            }
        )
        return params

    pca_dim = int(alignment.get("pca_dim", min(256, query_features.shape[-1])))
    feature_dim = query_features.shape[-1]
    q_mean, q_proj = pca_fit(query_features)
    r_mean, r_proj = pca_fit(reference_features)
    query_vectors = pca_project(query_features, q_mean, q_proj, pca_dim)
    reference_vectors = pca_project(reference_features, r_mean, r_proj, pca_dim)
    rotation, n_common = _fit_procrustes_rotation(
        query_vectors,
        reference_vectors,
        query_labels,
        reference_labels,
        pca_dim,
        use_procrustes=use_procrustes,
    )
    params.update(
        {
            "pca_dim": pca_dim,
            "feature_dim": feature_dim,
            "q_mean": q_mean.cpu(),
            "q_proj": q_proj.cpu(),
            "r_mean": r_mean.cpu(),
            "r_proj": r_proj.cpu(),
            "rotation": rotation.cpu(),
            "num_common_ids": n_common,
        }
    )
    return params


@torch.no_grad()
def evaluate_training_free(config: Dict, model, query_loader, reference_loader):
    evaluation = config["evaluation"]
    alignment = config.get("alignment", {})
    normalize_global = bool(evaluation.get("normalize_features", True))
    query_features, query_labels = predict(config, model, query_loader, normalize=normalize_global)
    reference_features, reference_labels = predict(config, model, reference_loader, normalize=normalize_global)

    query_vectors, reference_vectors = align_features(
        query_features,
        reference_features,
        query_labels,
        reference_labels,
        config,
        pretrained=None,
    )

    return evaluate_retrieval(
        query_features=query_vectors,
        reference_features=reference_vectors,
        query_labels=query_labels,
        reference_labels=reference_labels,
        ranks=evaluation.get("ranks", [1, 5, 10]),
        step_size=int(evaluation.get("step_size", 256)),
    )
