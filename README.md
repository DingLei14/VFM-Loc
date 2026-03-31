# VFM-Loc

PyTorch release for the paper **VFM-Loc: Zero-Shot Cross-View Geo-Localization via Aligning Discriminative Visual Hierarchies**.

This repository provides a cleaner GitHub-oriented release at the project root for the **training-free, zero-shot** VFM-Loc pipeline:

- root-level evaluation entry script
- YAML-based dataset and inference configuration
- unified loading of pretrained backbones from `pretrained_weights/`
- vendored DINOv3 code under the root repository
- support for flexible zero-shot evaluation across multiple CVGL datasets

## Paper

**Lu J, Sang Z, Wei H, et al.**  
**VFM-Loc: Zero-Shot Cross-View Geo-Localization via Aligning Discriminative Visual Hierarchies**  
*arXiv preprint arXiv:2603.13855, 2026*

## Highlights

- `eval.py`: zero-shot VFM-Loc evaluation entry
- `configs/eval/`: reusable config templates for CVUSA, CVACT, VIGOR and University-1652
- `vfm_loc/`: simplified release package containing dataset loaders, model wrappers, retrieval engine and zero-shot alignment logic
- `third_party/dinov3/`: vendored DINOv3 code used by the release implementation

## Repository Layout

```text
.
├── configs/
│   ├── base/
│   └── eval/
├── eval.py
├── pretrained_weights/
├── third_party/
│   └── dinov3/
├── vfm_loc/
└── ...
```

## Installation

```bash
pip install -r requirements.txt
```

## Data And Weights

Default dataset setting is **University-1652** with cloud path:

```text
/root/autodl-tmp/University/
```

Pretrained checkpoints should be placed under `pretrained_weights/`.

Example:

```text
pretrained_weights/
└── dinov3/
    └── dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

All dataset roots, checkpoint paths, image sizes, pooling modes, PCA dimension and retrieval settings are controlled by YAML files in `configs/`.

### Data Directory Guide

Set your own dataset root by editing the config file below.

Default file:

```text
configs/base/eval_base.yaml
```

Current setting:

```yaml
dataset:
  data_root: /root/autodl-tmp/University/
```

Example replacement:

```yaml
dataset:
  data_root: /path/to/your/University/
```

You can also override the dataset path in a dataset-specific config such as:

- `configs/eval/u1652_dinov3_zero_shot.yaml`
- `configs/eval/cvusa_dinov3_zero_shot.yaml`
- `configs/eval/cvact_dinov3_zero_shot.yaml`
- `configs/eval/vigor_same_dinov3_zero_shot.yaml`

In short, replace `/root/autodl-tmp/University/` with your own absolute dataset directory before running evaluation.

## Evaluation

Example zero-shot evaluation with DINOv3:

```bash
python eval.py --config configs/eval/u1652_dinov3_zero_shot.yaml
```

Other provided examples:

- `configs/eval/cvusa_dinov3_zero_shot.yaml`
- `configs/eval/cvact_dinov3_zero_shot.yaml`
- `configs/eval/vigor_same_dinov3_zero_shot.yaml`
- `configs/eval/u1652_dinov3_zero_shot.yaml`

The zero-shot pipeline includes:

- frozen VFM feature extraction
- GeM / R-MAC style hierarchical descriptor aggregation
- domain-wise PCA
- orthogonal Procrustes alignment
- cross-view retrieval in the aligned embedding space

## Config Notes

Important config sections:

- `dataset`: dataset root, split, task mode, image sizes
- `model`: backbone type, pretrained weight path, pooling, R-MAC settings
- `evaluation`: ranks, batch size, step size
- `alignment`: PCA dimension, Procrustes switch, scale weighting

## Notes

- This release is intentionally focused on the paper's **training-free** setting.
- The release implementation no longer depends on `VFM_CVGL/core/dinov3`.
- `eval.py` at the repository root is the recommended entry point.
- The **LO-UCV** dataset used in the paper **will be released**.
- Additional backbone support such as **Radio** models will be added in a future update.

## Citation

If you find this repository useful, please cite:

```bibtex
@article{lu2026vfmloc,
  title   = {VFM-Loc: Zero-Shot Cross-View Geo-Localization via Aligning Discriminative Visual Hierarchies},
  author  = {Jun Lu and Zehao Sang and Haoqi Wei and Xiangyun Liu and Kun Zhu and Haitao Guo and Zhihui Gong and Lei Ding},
  journal = {arXiv preprint arXiv:2603.13855},
  year    = {2026}
}
```
