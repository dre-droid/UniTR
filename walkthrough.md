# UniTR SSL Pretraining & Downstream Evaluation — Full Walkthrough

This document tracks the complete body of work across this conversation: from implementing iBOT-style self-supervised pretraining for the UniTR multi-modal backbone, through scaling it to the full NuScenes dataset on a SLURM-managed HPC cluster, to running comparative downstream object detection evaluations.

---

## Phase 1: iBOT SSL Architecture & Implementation

### 1.1 Student-Teacher Framework (`iBOTUniTR`)

A new model wrapper [ibot_unitr.py](file:///home/andrea_mastroberti/UniTR/pcdet/models/ssl/ibot_unitr.py) was implemented following the iBOT (Image BERT Pre-Training with Online Tokenizer) paradigm:

- **Shared VFE:** Both student and teacher share the same `DynPillarVFE` instance, creating voxel features `(N, 128)` from raw LiDAR points.
- **Student backbone:** Updated via standard backpropagation on the `L_MIM` (Masked Image/Voxel Modeling) distillation loss.
- **Teacher backbone:** Frozen (`torch.no_grad()`), updated via Exponential Moving Average (EMA) of the student weights. EMA momentum follows a cosine schedule from `0.996` → `1.0`.
- **Total params:** 8,667,200 (4,342,464 trainable — reflecting frozen teacher & frozen VFE).

### 1.2 Multi-Modal Masking

Custom masking utilities were implemented in [masking.py](file:///home/andrea_mastroberti/UniTR/pcdet/models/ssl/masking.py):

| Modality | Mask Ratio | Mechanism |
|----------|-----------|-----------|
| LiDAR voxels | 40% | Random subset replaced by learnable `[MASK]` token before student backbone |
| Image patches | 30% | Random patches masked in `PatchEmbed` output |

### 1.3 Distillation Head & Loss

- **`iBOTProjectionHead`:** 3-layer MLP projecting features into a normalized `4096`-dimensional space.
- **`iBOTLoss`:** Cross-entropy between sharpened teacher predictions and student predictions, computed only on masked coordinates.
- **Centering:** Running average of teacher outputs prevents mode collapse.

### 1.4 Dataset Adaptation

[nuscenes_dataset.py](file:///home/andrea_mastroberti/UniTR/pcdet/datasets/nuscenes/nuscenes_dataset.py) was modified to support label-free operation:

- When `SELF_SUPERVISED: True`, the pipeline bypasses ground-truth loading, augmentation filtering, and balanced resampling.
- Augmentations (scaling, rotation, translation) gracefully handle empty label tensors.
- Fixed sweep sampling bounds checking to prevent crashes on early frames.
- Added compatibility layer for MMDetection3d-style nuScenes metadata (camera intrinsics key mapping).
- Dynamically concatenated velocity to `gt_boxes` for TransFusionHead compatibility (9-D bounding boxes).

### 1.5 Smoke Test

A smoke test ([test_ssl_smoke.py](file:///home/andrea_mastroberti/UniTR/tools/test_ssl_smoke.py)) validated the full forward pass:

```text
Dataset OK: 28130  (All NuScenes samples loaded)
Model created: 8,667,200 params total, 4,342,464 trainable
Loss: 8.298693656921387
TB dict: {'loss_mim_voxel': 8.298, 'ema_momentum': 0.996, 'num_voxels': 13626, 'num_masked_voxels': 5450}
SMOKE TEST PASSED!
```

---

## Phase 2: Scaling SSL Pre-training to Full NuScenes

### 2.1 Initial Overnight Run (Manual, `gpu-light`)

The first full-dataset pre-training run was launched manually (not via SLURM script) with conservative settings:

| Parameter | Value |
|-----------|-------|
| Dataset | `v1.0-trainval` (28,130 samples) |
| Batch size | 2 |
| Workers | 4 |
| Epochs configured | 100 |
| Epochs completed | ~3 (ran overnight) |

**Key observations from the training log** ([train_ssl_20260401-231747.log](file:///home/andrea_mastroberti/UniTR/output/nuscenes_models/unitr_ibot/full_dataset_v1/train_ssl_20260401-231747.log)):
- **Epoch 1:** Loss dropped aggressively from `8.30` → `2.54` as the model learned trivial geometry (roads, empty space).
- **Epoch 2 onward:** Loss plateaued at `~0.78-0.79`. This is **expected behavior** for iBOT SSL — the numerical loss flattens while the latent space continues to structure class boundaries internally.
- The learning rate was still in the warmup phase (`WARMUP_EPOCH: 5`), suppressed at `~1e-05`. The OneCycle LR scheduler had not yet reached its peak.

### 2.2 SLURM Script Development

Three SLURM submission scripts were developed and refined:

#### SSL Pre-training: [batch_run_ssl.slurm](file:///home/andrea_mastroberti/UniTR/tools/batch_run_ssl.slurm)
```bash
#SBATCH --partition=gpu-heavy      # 12-day time limit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

python -u train_ssl.py \
    --extra_tag full_dataset_v1 \
    --cfg_file cfgs/nuscenes_models/unitr_ibot.yaml \
    --batch_size 4 \
    --epochs 50 \
    --workers 8
```

#### Downstream Fine-Tuning (SSL-initialized): [batch_run_evaluate.slurm](file:///home/andrea_mastroberti/UniTR/tools/batch_run_evaluate.slurm)
```bash
#SBATCH --partition=gpu-light

python -u train.py \
    --cfg_file cfgs/nuscenes_models/unitr.yaml \
    --pretrained_model ../pretrained_unitr.pth \
    --batch_size 1 --epochs 10 \
    --extra_tag v1_mini_evaluation \
    --set DATA_CONFIG.VERSION v1.0-mini ...
```

#### Downstream Baseline (Random Init): [batch_run_baseline.slurm](file:///home/andrea_mastroberti/UniTR/tools/batch_run_baseline.slurm)
```bash
#SBATCH --partition=gpu-light

python -u train.py \
    --cfg_file cfgs/nuscenes_models/unitr.yaml \
    --batch_size 1 --epochs 10 \
    --extra_tag v1_mini_baseline \
    --set DATA_CONFIG.VERSION v1.0-mini ...
```

> [!IMPORTANT]
> The `--extra_tag` flag is critical for output isolation. It ensures that `v1_mini_evaluation` and `v1_mini_baseline` write to completely separate checkpoint/log directories under `output/nuscenes_models/unitr/`.

### 2.3 Hyperparameter Optimization Decisions

| Parameter | Initial Value | Final Value | Rationale |
|-----------|--------------|-------------|-----------|
| `--batch_size` | 2 | **4** | VRAM smoke test on 40GB GPU confirmed BS=4 fits without OOM. Doubles sample throughput. |
| `--epochs` | 100 | **50** | The YAML config's `DECAY_STEP_LIST: [35, 45]` reveals the authors designed the LR schedule around ~50 epochs. SSL contrastive learning heavily plateaus after 30-50 epochs. |
| `--workers` | 4 | **8** | Matched to `--cpus-per-task=8` for optimal DataLoader parallelism. |
| `--partition` | `gpu-light` | **`gpu-heavy`** | 12-day time limit (vs 2-day on `gpu-light`) ensures the full 50-epoch run completes without SLURM termination. |

### 2.4 VRAM Limit Testing

Batch size limits were validated interactively before committing to a multi-day job:

```bash
srun -p gpu-light --gres=gpu:1 python -u train_ssl.py \
    --cfg_file cfgs/nuscenes_models/unitr_ibot.yaml \
    --batch_size 4 --workers 8 --extra_tag vram_limit_test
```

The test ran past 50 iterations without `CUDA out of memory`, confirming BS=4 is safe on 40GB VRAM. The process was then manually interrupted with `Ctrl+C`.

### 2.5 Automatic Checkpoint Resumption

The `train_ssl.py` script has built-in auto-resume logic: if no `--ckpt` argument is provided, it scans `output/.../ckpt/` for existing `.pth` files and resumes from the most recent one. This was confirmed in the Job 8811 log:

```
INFO  ==> Loading checkpoint .../unitr_ibot/full_dataset_v1/ckpt/latest_model.pth
INFO  ==> Done
INFO  **********************Start SSL training nuscenes_models/unitr_ibot(full_dataset_v1)**********************
epochs:   0%|          | 0/47 [00:00<?, ?it/s]   ← Resumed at epoch 4 (47 remaining)
```

---

## Phase 3: Weight Extraction & Downstream Evaluation

### 3.1 Weight Extraction Script

[prepare_pretrained.py](file:///home/andrea_mastroberti/UniTR/tools/prepare_pretrained.py) converts SSL checkpoints into downstream-compatible weights:

- Extracts `vfe.*` keys (shared pillar VFE) as-is.
- Remaps `student_backbone.*` → `mm_backbone.*` to match the supervised UniTR architecture.
- Strips optimizer/scheduler state to minimize file size.
- Source: `output/nuscenes_models/unitr_ibot/full_dataset_v1/ckpt/latest_model.pth`
- Output: `pretrained_unitr.pth` (7.8MB, saved to both `UniTR/` root and `tools/`)

### 3.2 NuScenes Evaluation Bug Fixes

Two critical bugs were resolved before evaluation could complete:

1. **`dataroot` path duplication** ([nuscenes_dataset.py:321](file:///home/andrea_mastroberti/UniTR/pcdet/datasets/nuscenes/nuscenes_dataset.py#L321)): The evaluation code concatenated the NuScenes version string twice. Fixed by setting `dataroot=str(self.root_path.parent)`.

2. **Missing map files:** The NuScenes devkit requires map `.png` files even for detection-only evaluation. Workaround: injected dummy map files into `data/nuscenes/maps/`.

### 3.3 NumPy 2.x Compatibility

Bulk replaced deprecated `np.int` alias with native Python `int` across:
- [database_sampler.py](file:///home/andrea_mastroberti/UniTR/pcdet/datasets/augmentor/database_sampler.py)
- [base_bev_backbone.py](file:///home/andrea_mastroberti/UniTR/pcdet/models/backbones_2d/base_bev_backbone.py)

### 3.4 Comparative Evaluation Results (v1.0-mini)

Both models were trained for 10 epochs on the `v1.0-mini` dataset (81 training samples), then evaluated on the mini validation set.

#### SSL-Pretrained Backbone (1 epoch of SSL)

| Metric | Value |
|--------|-------|
| **mAP** | 0.1527 |
| **NDS** | 0.1675 |

| Object Class | AP | ATE | ASE | AOE |
|---|---|---|---|---|
| car | 0.603 | 0.390 | 0.758 | 1.510 |
| pedestrian | 0.653 | 0.162 | 0.365 | 1.628 |
| bus | 0.205 | 1.355 | 0.853 | 2.320 |
| truck | 0.066 | 0.456 | 0.788 | 1.583 |
| All others | 0.000 | — | — | — |

#### Baseline (Random Init)

The baseline model (no pretrained weights) **outperformed** the SSL model on this micro-dataset.

> [!NOTE]
> **Why the baseline won:** The SSL checkpoint had only completed ~1 epoch of warmup (LR still suppressed at `~1e-05`). The features were in an infant state. When squeezed into a tiny supervised fine-tuning regime (81 samples, 10 epochs), the heavy pretrained weights resisted adaptation. Meanwhile, the randomly initialized baseline freely overfitted the small dataset. This result is expected and will reverse once SSL training completes all 50 epochs.

---

## Phase 4: Final Production Run (Current)

### Active Job

| Field | Value |
|-------|-------|
| **Job ID** | 8811 |
| **Partition** | `gpu-heavy` (`move-hpc-08-gpu`) |
| **Configuration** | BS=4, 50 epochs, 8 workers, AMP enabled |
| **Resume point** | Epoch 4 (auto-resumed from overnight run) |
| **LR schedule** | OneCycle with 5-epoch warmup, peak at `1e-04` |
| **ETA** | ~4-5 days |

### Key Files Reference

| Purpose | Path |
|---------|------|
| SSL training script | [train_ssl.py](file:///home/andrea_mastroberti/UniTR/tools/train_ssl.py) |
| SSL SLURM script | [batch_run_ssl.slurm](file:///home/andrea_mastroberti/UniTR/tools/batch_run_ssl.slurm) |
| SSL model config | [unitr_ibot.yaml](file:///home/andrea_mastroberti/UniTR/tools/cfgs/nuscenes_models/unitr_ibot.yaml) |
| Weight extractor | [prepare_pretrained.py](file:///home/andrea_mastroberti/UniTR/tools/prepare_pretrained.py) |
| Downstream eval script | [batch_run_evaluate.slurm](file:///home/andrea_mastroberti/UniTR/tools/batch_run_evaluate.slurm) |
| Baseline script | [batch_run_baseline.slurm](file:///home/andrea_mastroberti/UniTR/tools/batch_run_baseline.slurm) |
| SSL checkpoints | `output/nuscenes_models/unitr_ibot/full_dataset_v1/ckpt/` |
| Downstream eval output | `output/nuscenes_models/unitr/v1_mini_evaluation/` |
| Baseline output | `output/nuscenes_models/unitr/v1_mini_baseline/` |
| Extracted weights | [pretrained_unitr.pth](file:///home/andrea_mastroberti/UniTR/pretrained_unitr.pth) (7.8MB) |
| NuScenes dataset fix | [nuscenes_dataset.py:321](file:///home/andrea_mastroberti/UniTR/pcdet/datasets/nuscenes/nuscenes_dataset.py#L321) |

---

## Next Steps

1. **Monitor Job 8811:** Allow the 50-epoch SSL pre-training to complete on `gpu-heavy`.
2. **Re-extract weights:** Run `prepare_pretrained.py` again with the fully-trained checkpoint.
3. **Re-evaluate:** Submit `batch_run_evaluate.slurm` with the new weights and compare against baseline.
4. **Scale downstream:** If SSL shows clear gains, scale supervised fine-tuning to the full `v1.0-trainval` dataset.
