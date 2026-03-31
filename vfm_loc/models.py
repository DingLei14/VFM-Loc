from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import timm
import torch
import torch.nn as nn

from .utils import ensure_vendored_package, load_state_dict, resolve_checkpoint_path, select_device


class TimmModel(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True, img_size: int = 384) -> None:
        super().__init__()
        if "vit" in model_name:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size)
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def get_preprocess_config(self) -> Dict[str, List[float]]:
        data_config = timm.data.resolve_model_data_config(self.model)
        return {
            "mean": data_config["mean"],
            "std": data_config["std"],
        }

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        if hasattr(self.model, "set_grad_checkpointing"):
            self.model.set_grad_checkpointing(enable)

    def forward(self, query: torch.Tensor, reference: Optional[torch.Tensor] = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if reference is None:
            return self.model(query)
        return self.model(query), self.model(reference)


class DinoV3Embedder(nn.Module):
    def __init__(
        self,
        model_name: str,
        weights: Optional[str] = None,
        img_size: int = 224,
        pool: str = "gem",
        use_rmac: bool = True,
        rmac_levels: Optional[List[int]] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        ensure_vendored_package("dinov3", "third_party/dinov3")
        from dinov3.hub.backbones import (  # type: ignore
            dinov3_convnext_base,
            dinov3_convnext_large,
            dinov3_convnext_small,
            dinov3_convnext_tiny,
            dinov3_vit7b16,
            dinov3_vitb16,
            dinov3_vith16plus,
            dinov3_vitl16,
            dinov3_vitl16plus,
            dinov3_vits16,
        )

        model_map = {
            "dinov3_vits16": dinov3_vits16,
            "dinov3_vitb16": dinov3_vitb16,
            "dinov3_vitl16": dinov3_vitl16,
            "dinov3_vith16plus": dinov3_vith16plus,
            "dinov3_vitl16plus": dinov3_vitl16plus,
            "dinov3_vit7b16": dinov3_vit7b16,
            "dinov3_convnext_tiny": dinov3_convnext_tiny,
            "dinov3_convnext_small": dinov3_convnext_small,
            "dinov3_convnext_base": dinov3_convnext_base,
            "dinov3_convnext_large": dinov3_convnext_large,
        }
        if model_name not in model_map:
            raise ValueError(f"Unknown DINOv3 model: {model_name}")

        self.model_name = model_name
        self.img_size = int(img_size)
        self.pool = pool
        self.use_rmac = use_rmac
        self.rmac_levels = rmac_levels or [1, 2]
        self.gem_p = 3.0
        self.device_str = select_device(device)

        resolved_weights = resolve_checkpoint_path(weights)
        if resolved_weights:
            self.model = model_map[model_name](pretrained=False)
            load_state_dict(self.model, resolved_weights, strict=False)
        else:
            self.model = model_map[model_name](pretrained=True)

    def get_preprocess_config(self) -> Dict[str, List[float] | int]:
        return {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "resolution": self.img_size,
        }

    @torch.inference_mode()
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if self.device_str.startswith("cuda") and not img.is_cuda:
            img = img.cuda(non_blocking=True)

        features = self.model(img, is_training=True)
        patch_tokens = features["x_norm_patchtokens"]
        cls_token = features["x_norm_clstoken"]
        tokens = torch.cat([cls_token.unsqueeze(1), patch_tokens], dim=1)

        if self.pool == "cls":
            pooled = tokens[:, 0]
        elif self.pool == "avg":
            pooled = tokens.mean(dim=1)
        elif self.pool == "gem":
            x = tokens.clamp_min(1e-6).pow(self.gem_p)
            pooled = x.mean(dim=1).pow(1.0 / self.gem_p)
        else:
            raise ValueError(f"Unknown pool: {self.pool}")

        if not self.use_rmac:
            return pooled

        bsz, num_tokens, dim = tokens.shape
        grid = int((num_tokens - 1) ** 0.5)
        patches = tokens[:, 1:, :].view(bsz, grid, grid, dim).permute(0, 3, 1, 2)
        regions = []
        for level in self.rmac_levels:
            win_h = max(1, grid // level)
            win_w = max(1, grid // level)
            for i in range(level):
                for j in range(level):
                    y0 = i * win_h
                    x0 = j * win_w
                    y1 = grid if i == level - 1 else min(grid, (i + 1) * win_h)
                    x1 = grid if j == level - 1 else min(grid, (j + 1) * win_w)
                    patch = patches[:, :, y0:y1, x0:x1]
                    if self.pool == "gem":
                        region = patch.clamp_min(1e-6).pow(self.gem_p).mean(dim=(-2, -1)).pow(1.0 / self.gem_p)
                    else:
                        region = patch.mean(dim=(-2, -1))
                    regions.append(region)
        return torch.stack(regions, dim=1)


def build_model(model_cfg: Dict[str, object]) -> nn.Module:
    model_type = str(model_cfg.get("type", "timm")).lower()
    device = str(model_cfg.get("device")) if model_cfg.get("device") else None

    if model_type == "timm":
        model = TimmModel(
            model_name=str(model_cfg["name"]),
            pretrained=bool(model_cfg.get("pretrained", True)),
            img_size=int(model_cfg.get("img_size", 384)),
        )
        checkpoint_path = resolve_checkpoint_path(model_cfg.get("checkpoint"))
        if checkpoint_path:
            load_state_dict(model, checkpoint_path, strict=False)
        return model

    if model_type == "dinov3":
        return DinoV3Embedder(
            model_name=str(model_cfg["name"]),
            weights=model_cfg.get("weights"),
            img_size=int(model_cfg.get("img_size", 224)),
            pool=str(model_cfg.get("pool", "gem")),
            use_rmac=bool(model_cfg.get("use_rmac", True)),
            rmac_levels=list(model_cfg.get("rmac_levels", [1, 2])),
            device=device,
        )

    raise ValueError(f"Unknown model type: {model_type}")
