# RobustCAM — Stable Grad-CAM Explanations via Augmentation Fusion

RobustCAM addresses Grad-CAM's spatial instability by computing explanations across
multiple semantic-preserving augmented views, aligning heatmaps via inverse geometric
warping, and fusing them with mean or median aggregation. The framework is evaluated
in two settings: general-purpose ImageNet classification and lung CT classification.

---

## How it works

For each input image, RobustCAM applies 6 augmentations (identity, horizontal flip,
rotation +15°, colour jitter, Gaussian blur, greyscale), computes Grad-CAM on each,
warps geometric views back to the original coordinate frame, and fuses the aligned
heatmaps pixel-wise. The cross-augmentation standard deviation gives a per-pixel
uncertainty map.

---

## Experiment 1 — ImageNet / VGG16

**Goal:** Validate that augmentation fusion improves explanation stability on a
general-purpose benchmark before applying the framework to medical images.

**Setup:** 50-image curated ImageNet subset (5 domains × 10 images), pretrained VGG16
(no fine-tuning), Grad-CAM at 3 layers (Layer 14, 20, 30).

**Results at Layer 30 (deepest, most semantic):**

| Method | Mean Variance ↓ | Top-k IoU ↑ | Spearman ρ ↑ |
|---|---|---|---|
| Standard Grad-CAM | 0.142 ± 0.031 | 0.524 ± 0.089 | 0.612 ± 0.112 |
| RobustCAM (mean)  | 0.058 ± 0.015 | 0.781 ± 0.052 | 0.871 ± 0.041 |
| RobustCAM (median)| **0.051 ± 0.012** | **0.798 ± 0.048** | **0.889 ± 0.038** |
| **Improvement**   | **64%**        | **52%**         | **45%**          |

All improvements: p < 0.001, Cohen's d > 1.2. Stability improves monotonically with
depth. The 6-augmentation suite is the recommended default; extending to 8 views yields
marginal gains at 35% higher runtime.

Run the legacy single-image pipeline (VGG16):

```cmd
conda activate robustcam
python src/pipeline.py --image "data\Normal cases\Normal case (1).jpg" --layers 14 20 30 --num-aug 6 --fusion-method mean
```

Note: this pipeline produces ImageNet class predictions, not CT class predictions.

---

## Experiment 2 — Lung CT / IQ-OTH/NCCD

**Goal:** Transfer RobustCAM to a clinical domain, fine-tune multiple architectures,
and evaluate with a comprehensive 9-metric faithfulness suite supplemented by LIME
and SHAP cross-validation.

### Dataset

| Class | Folder | Images | Train | Val | Test |
|---|---|---|---|---|---|
| 0 Normal    | `Normal cases`    | 416   | 251 | 83  | 75  |
| 1 Benign    | `Bengin cases`    | 120   | 74  | 24  | 24  |
| 2 Malignant | `Malignant cases` | 561   | 333 | 112 | 121 |
| **Total**   |                   | **1097** | **658** | **219** | **220** |

Note: "Bengin cases" is the actual folder name (misspelling is in the original dataset).
60/20/20 split, seed=42, weighted cross-entropy (Benign class weight ≈ 4.6×).

### Multi-architecture fine-tuning

Five architectures from Panboonyuen 2026 were fine-tuned with a 3-class head (frozen
backbone, Adam lr=1e-3, batch 32, 25 epochs). Published baseline: ResNet-50 accuracy=0.85.

| Architecture | Val Acc | Test W-F1 | Best Epoch | Hook Layers | XAI Methods |
|---|---|---|---|---|---|
| ResNet-50       | 0.881 | 0.865 | 18 | layer3, layer4 | Grad-CAM + LIME + SHAP |
| ResNet-101      | **0.927** | 0.868 | 16 | layer3, layer4 | Grad-CAM + LIME + SHAP |
| DenseNet-161    | 0.904 | 0.881 | 12 | features.denseblock3/4 | Grad-CAM only |
| EfficientNet-B0 | 0.918 | **0.911** | 23 | features.6/7/8 | Grad-CAM + LIME + SHAP |
| ViT-B/16        | 0.900 | 0.898 | 22 | encoder_layer_10/11 | Grad-CAM only* |

*ViT-B/16 produces near-uniform Grad-CAM heatmaps (confirmed in both ImageNet and CT
settings). Classification performance is competitive; meaningful localisation requires
attention-based explanation methods.

Full results: `results/tables/architecture_comparison.csv` and training curves in
`results/figures/training/`.

### Grad-CAM vs. Robust-CAM faithfulness (CNN architectures)

| Architecture | GC Consist↑ | RC Consist↑ | GC Faith↑ | RC Faith↑ | RC Spear↑ |
|---|---|---|---|---|---|
| ResNet-50       | 0.351 | **0.471** | 0.214 | 0.191 | 0.707 |
| ResNet-101      | 0.458 | **0.534** | 0.245 | 0.241 | 0.752 |
| EfficientNet-B0 | 0.173 | **0.495** | 0.155 | 0.150 | 0.663 |
| DenseNet-161    | 0.303 | **0.482** | 0.158 | 0.146 | 0.723 |

RobustCAM raises explanation consistency (Consist IoU) across all CNN architectures.
The largest gain is EfficientNet-B0, where single-view Grad-CAM is intrinsically
unstable (0.173 → 0.495, +186%). A small faithfulness reduction is expected and
consistent: the fused map covers a slightly broader region.

### Cross-method comparison (LIME and SHAP)

| Architecture | GC Faith | LIME Faith | SHAP Faith | GC Fid | SHAP Fid |
|---|---|---|---|---|---|
| ResNet-50       | 0.214 | 0.208 | 0.305 | −0.077 |  0.000 |
| ResNet-101      | 0.245 | 0.181 | 0.302 | −0.018 |  0.000 |
| EfficientNet-B0 | 0.155 | 0.152 | **0.499** | −0.050 | 0.050 |

SHAP achieves the highest faithfulness across all methods. Negative Grad-CAM/LIME
fidelity means the model classifies correctly with the highlighted region removed —
CT classification relies on distributed structural context, not isolated focal lesions.

Run the batch evaluation pipeline:

```cmd
# Grad-CAM + Robust-CAM only (fast)
conda activate robustcam
python src/eval_pipeline.py --split test --max-images 5 --no-lime --no-shap

# Full evaluation with LIME + SHAP
python src/eval_pipeline.py --split test --run-name full_eval
```

---

## 9-Metric faithfulness suite

| Group | Metric | Symbol | Source |
|---|---|---|---|
| A | Perturbation faithfulness | Faith↑ | Panboonyuen 2026 Eq. 5 |
| A | Localization accuracy | LocAcc↑ | Panboonyuen 2026 Eq. 4 (NaN: no CT pixel masks) |
| A | Consistency (IoU) | Consist↑ | Panboonyuen 2026 Eq. 6 |
| B | Fidelity | Fid↑ | Akgündoğdu 2025 Eq. 11-15 |
| B | Stability | Stab↑ | Akgündoğdu 2025 Eq. 16-17 |
| B | Consistency (Pearson) | Cons↑ | Akgündoğdu 2025 Eq. 18 |
| C | Mean per-pixel variance | Var↓ | RobustCAM (Robust-CAM only) |
| C | Mean top-k IoU | IoU_k↑ | RobustCAM (Robust-CAM only) |
| C | Mean Spearman ρ | Spear↑ | RobustCAM (Robust-CAM only) |

Group C metrics require multiple augmented views — reported as NaN for single-view
methods (Grad-CAM baseline, LIME, SHAP).

---

## Setup

```cmd
conda create -n robustcam python=3.12 -y
conda activate robustcam
pip install -r requirements.txt
```

All scripts must use the `robustcam` environment.

---

## Project layout

```
src/
├── train.py                    ← fine-tune any architecture on IQ-OTH/NCCD
├── iq_othncc_dataset.py        ← dataset loader, 60/20/20 split, seed=42
├── faithfulness_metrics.py     ← 9-metric suite, compute_all_metrics()
├── lime_service.py             ← LIME explanation wrapper
├── shap_service.py             ← SHAP via GradientExplainer
├── xai_fusion.py               ← pixel-wise voting mask (Grad-CAM + LIME + SHAP)
├── visualize_xai.py            ← qualitative XAI comparison figures per class
├── eval_pipeline.py            ← batch evaluation: metrics, figures, CSV export
├── cascading_randomization.py  ← Adebayo 2018 sanity check (layer randomization)
├── model_service.py            ← ModelService + HookManager, 5-arch support
├── data_service.py             ← image loading and ImageNet preprocessing
├── gradcam_service.py          ← Grad-CAM computation (arch-agnostic)
├── augmentation_service.py     ← 6 augmentation types + meta dict for warp
├── robust_cam.py               ← fuse_mean/median, warp_heatmap_back, stability metrics
├── mlflow_service.py           ← MLflow experiment logging
├── pipeline.py                 ← legacy VGG16/ImageNet CLI (Experiment 1)
└── vgg_structure.py            ← VGG16 layer index helper (legacy)

checkpoints/
└── resnet50_iqothnc.pth        ← epoch=18, val_acc=0.8813

results/
├── figures/
│   ├── training/               ← loss/accuracy curves for all 5 architectures
│   ├── qualitative/            ← per-image XAI overlays and comparison panels
│   └── quantitative/           ← metric bar charts (faith, fidelity, stability, consistency)
└── tables/
    ├── architecture_comparison.csv   ← Experiment 2 multi-arch classification results
    ├── metrics_table_test.csv        ← 9-metric comparison across XAI methods
    └── classification_report.csv    ← per-class precision / recall / F1
```

---

## MLflow tracking

```cmd
mlflow ui --backend-store-uri ./mlruns --port 5000
```
