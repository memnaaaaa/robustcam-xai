"""
src/cascading_randomization.py

Cascading randomization sanity check (Adebayo et al. 2018).
Progressively reinitializes ResNet50 layer groups top-down and shows how Grad-CAM and
Robust-CAM heatmaps degrade — visually confirming that explanations are sensitive to
learned weights, not just image edges.

Coarse mode (default): fc → layer4 → layer3 → layer2 → layer1  (6 states)
Fine-grained mode (--fine-grained): randomizes individual Bottleneck blocks top-down
  fc → layer4.2 → layer4.1 → layer4.0 → layer3.5 → ... → layer1.0  (17 states)

Outputs (all in --out-dir):
  {class}_{stem}_grid.png          heatmap grid per sample image
  {class}_{stem}_big_grid.png      single consolidated grid (1-image mode, fine-grained)
  degradation_spearman_rho.png     Spearman rho vs fully-trained state, both methods
  spearman_rho_table.csv           CSV of rho values per image / method / state
"""

import os
import sys
import argparse
import copy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from model_service import ModelService
from gradcam_service import GradCAMService
from data_service import DataService
from augmentation_service import AugmentationService
from iq_othncc_dataset import IQOTHNCCDDataset
import robust_cam

# ── Randomization states ────────────────────────────────────────────────────
# Coarse: whole layer groups top-down
COARSE_LAYER_ORDER = ["fc", "layer4", "layer3", "layer2", "layer1"]
COARSE_STATE_LABELS = [
    "Trained",
    "rand fc",
    "rand fc+L4",
    "rand fc+L4+L3",
    "rand fc+L4+L3+L2",
    "All Random",
]

# Fine-grained: individual Bottleneck blocks, top-down
# ResNet50: layer4(3 blocks), layer3(6 blocks), layer2(4 blocks), layer1(3 blocks)
FINE_LAYER_ORDER = (
    ["fc"]
    + [f"layer4.{i}" for i in [2, 1, 0]]
    + [f"layer3.{i}" for i in [5, 4, 3, 2, 1, 0]]
    + [f"layer2.{i}" for i in [3, 2, 1, 0]]
    + [f"layer1.{i}" for i in [2, 1, 0]]
)

def _make_fine_labels(layer_order: list[str]) -> list[str]:
    labels = ["Trained"]
    randomized = []
    for layer in layer_order:
        randomized.append(layer)
        if len(randomized) == 1:
            labels.append(f"+ {layer}")
        elif len(randomized) <= 3:
            labels.append("+ " + layer)
        else:
            # abbreviate: show last added only
            labels.append(f"...+{layer}")
    return labels

FINE_STATE_LABELS = _make_fine_labels(FINE_LAYER_ORDER)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _select_samples(data_root: str, n_per_class: int) -> list:
    """Pick first n_per_class test samples per class (deterministic)."""
    ds = IQOTHNCCDDataset(data_root, split="test")
    all_samples = ds.get_all_samples()
    by_class: dict[int, list] = {}
    for path, label, class_name in all_samples:
        by_class.setdefault(label, []).append((path, label, class_name))
    selected = []
    for label in sorted(by_class.keys()):
        selected.extend(by_class[label][:n_per_class])
    return selected


def _reinit_layer_group(model, layer_name: str):
    """
    Reinitialize all parameters in a named module (supports dotted paths like 'layer4.2').
    """
    # resolve dotted path
    parts = layer_name.split(".")
    group = model
    for part in parts:
        group = getattr(group, part)
    for module in group.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()


def _spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    rho, _ = spearmanr(a.flatten(), b.flatten())
    return float(rho) if not np.isnan(rho) else 0.0


# ── Heatmap computation ───────────────────────────────────────────────────────

def _compute_gradcam_heatmap(
    ms: ModelService,
    tensor,
    layer_name: str = "layer4",
) -> np.ndarray:
    """Compute a single Grad-CAM heatmap (float32 [0,1])."""
    ms.register_hooks_by_name([layer_name])
    class_idx, activations, gradients = ms.run(tensor)
    act = activations.get(layer_name)
    grad = gradients.get(layer_name)
    if act is None or grad is None:
        return np.zeros((7, 7), dtype=np.float32)
    svc = GradCAMService()
    return svc.compute_raw_heatmap(act, grad)


def _compute_robustcam_heatmap(
    ms: ModelService,
    pil_image: Image.Image,
    aug_service: AugmentationService,
    ds: DataService,
    layer_name: str = "layer4",
    n_views: int = 6,
) -> np.ndarray:
    """Compute Robust-CAM fused heatmap from n_views augmented views."""
    aug_dict = aug_service.apply(pil_image)
    items = list(aug_dict.items())[:n_views]
    heatmaps = []
    for name, (aug_img, meta) in items:
        tensor = ds.preprocess(aug_img).to(ms.device)
        ms.register_hooks_by_name([layer_name])
        class_idx, activations, gradients = ms.run(tensor)
        act = activations.get(layer_name)
        grad = gradients.get(layer_name)
        if act is None or grad is None:
            continue
        svc = GradCAMService()
        hm = svc.compute_raw_heatmap(act, grad)
        hm_warped = robust_cam.warp_heatmap_back(hm, meta, (hm.shape[0], hm.shape[1]))
        heatmaps.append(hm_warped)
    if not heatmaps:
        return np.zeros((7, 7), dtype=np.float32)
    return robust_cam.fuse_mean(heatmaps)


# ── Visualization ─────────────────────────────────────────────────────────────

def _build_sample_grid(
    pil_image: Image.Image,
    state_results: list[dict],  # list of {"gradcam": hm, "robustcam": hm | None}
    state_labels: list[str],
    include_robustcam: bool,
) -> plt.Figure:
    """
    Build a grid figure for one sample image.
    Rows: Grad-CAM overlays | Robust-CAM overlays | Delta-from-trained maps
    Columns: one per randomization state (6 total)
    """
    n_states = len(state_labels)
    n_rows = 3 if include_robustcam else 2  # delta row always shown
    fig, axes = plt.subplots(n_rows, n_states, figsize=(3 * n_states, 3 * n_rows))

    orig_np = np.array(pil_image.resize((224, 224)))
    svc = GradCAMService()

    gc0 = state_results[0]["gradcam"]
    rc0 = state_results[0].get("robustcam") if include_robustcam else None

    for col, (result, label) in enumerate(zip(state_results, state_labels)):
        gc_hm = result["gradcam"]
        gc_hm_resized = svc.resize_heatmap_to_image(gc_hm, 224, 224)
        gc_overlay = svc.overlay(orig_np, gc_hm_resized)

        axes[0, col].imshow(gc_overlay)
        axes[0, col].set_title(label, fontsize=8)
        axes[0, col].axis("off")

        if include_robustcam:
            rc_hm = result.get("robustcam")
            if rc_hm is not None:
                rc_hm_resized = svc.resize_heatmap_to_image(rc_hm, 224, 224)
                rc_overlay = svc.overlay(orig_np, rc_hm_resized)
                axes[1, col].imshow(rc_overlay)
            else:
                axes[1, col].imshow(orig_np)
            axes[1, col].axis("off")

        # Delta row: abs difference vs state-0 (Grad-CAM)
        delta_row = 2 if include_robustcam else 1
        gc0_resized = svc.resize_heatmap_to_image(gc0, 224, 224)
        delta_gc = np.abs(gc_hm_resized - gc0_resized)
        axes[delta_row, col].imshow(delta_gc, cmap="hot", vmin=0, vmax=1)
        axes[delta_row, col].axis("off")

    # Row labels
    axes[0, 0].set_ylabel("Grad-CAM", fontsize=9)
    if include_robustcam:
        axes[1, 0].set_ylabel("Robust-CAM", fontsize=9)
    axes[n_rows - 1, 0].set_ylabel("Δ from Trained\n(Grad-CAM)", fontsize=9)

    fig.suptitle("Cascading Randomization Sanity Check", fontsize=11, y=1.01)
    fig.tight_layout()
    return fig


def _build_big_grid(
    pil_image: Image.Image,
    state_results: list[dict],
    state_labels: list[str],
    include_robustcam: bool,
    cols_per_row: int = 9,
) -> plt.Figure:
    """
    One consolidated figure for a single image with many states.
    States wrap into multiple row-groups of cols_per_row.
    Each row-group has: Grad-CAM row + (optionally) Robust-CAM row + delta row.
    """
    n_states = len(state_labels)
    n_groups = (n_states + cols_per_row - 1) // cols_per_row
    rows_per_group = 3 if include_robustcam else 2
    total_rows = n_groups * rows_per_group

    fig, axes = plt.subplots(
        total_rows, cols_per_row,
        figsize=(2.4 * cols_per_row, 2.6 * total_rows),
    )
    # ensure 2D axes array
    if total_rows == 1:
        axes = axes[np.newaxis, :]
    if cols_per_row == 1:
        axes = axes[:, np.newaxis]

    orig_np = np.array(pil_image.resize((224, 224)))
    svc = GradCAMService()
    gc0 = svc.resize_heatmap_to_image(state_results[0]["gradcam"], 224, 224)

    for state_idx, (result, label) in enumerate(zip(state_results, state_labels)):
        group = state_idx // cols_per_row
        col = state_idx % cols_per_row
        base_row = group * rows_per_group

        gc_hm = svc.resize_heatmap_to_image(result["gradcam"], 224, 224)
        gc_overlay = svc.overlay(orig_np, gc_hm)
        axes[base_row, col].imshow(gc_overlay)
        axes[base_row, col].set_title(label, fontsize=6.5, pad=2)
        axes[base_row, col].axis("off")

        if include_robustcam:
            rc_raw = result.get("robustcam")
            if rc_raw is not None:
                rc_hm = svc.resize_heatmap_to_image(rc_raw, 224, 224)
                axes[base_row + 1, col].imshow(svc.overlay(orig_np, rc_hm))
            else:
                axes[base_row + 1, col].imshow(orig_np)
            axes[base_row + 1, col].axis("off")

        delta_row = base_row + (2 if include_robustcam else 1)
        delta = np.abs(gc_hm - gc0)
        axes[delta_row, col].imshow(delta, cmap="hot", vmin=0, vmax=1)
        axes[delta_row, col].axis("off")

    # hide unused axes in the last row-group
    used_in_last = n_states % cols_per_row
    if used_in_last:
        last_group = n_groups - 1
        for col in range(used_in_last, cols_per_row):
            base_row = last_group * rows_per_group
            for r in range(rows_per_group):
                axes[base_row + r, col].set_visible(False)

    # row-group labels on leftmost column
    row_labels_gc = ["Grad-CAM"] * n_groups
    row_labels_rc = ["Robust-CAM"] * n_groups
    row_labels_delta = ["Δ from\nTrained"] * n_groups
    for g in range(n_groups):
        base_row = g * rows_per_group
        axes[base_row, 0].set_ylabel(row_labels_gc[g], fontsize=8)
        if include_robustcam:
            axes[base_row + 1, 0].set_ylabel(row_labels_rc[g], fontsize=8)
        axes[base_row + rows_per_group - 1, 0].set_ylabel(row_labels_delta[g], fontsize=8)

    fig.suptitle(
        "Cascading Randomization — Fine-Grained Block-Level Sanity Check",
        fontsize=12, y=1.002,
    )
    fig.tight_layout(pad=0.4)
    return fig


def _build_degradation_plot(
    state_labels: list[str],
    gc_rho: list[list[float]],   # [n_samples][n_states]
    rc_rho: list[list[float]] | None,
    sample_names: list[str],
    class_names: list[str],
) -> plt.Figure:
    """Line plot of Spearman ρ vs randomization state for all samples."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = list(range(len(state_labels)))
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]

    for i, (gc_row, name, cls) in enumerate(zip(gc_rho, sample_names, class_names)):
        c = colors[i % len(colors)]
        ax.plot(x, gc_row, color=c, linewidth=1.2, alpha=0.7,
                label=f"GC {cls}/{name}")

    if rc_rho is not None:
        for i, (rc_row, name, cls) in enumerate(zip(rc_rho, sample_names, class_names)):
            c = colors[i % len(colors)]
            ax.plot(x, rc_row, color=c, linewidth=1.2, alpha=0.7,
                    linestyle="--", label=f"RC {cls}/{name}")

    # mean lines
    gc_mean = np.mean(gc_rho, axis=0)
    ax.plot(x, gc_mean, color="black", linewidth=2.5, label="GC mean")
    if rc_rho is not None:
        rc_mean = np.mean(rc_rho, axis=0)
        ax.plot(x, rc_mean, color="black", linewidth=2.5, linestyle="--", label="RC mean")

    ax.set_xticks(x)
    ax.set_xticklabels(state_labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Spearman ρ vs Trained")
    ax.set_ylim(0, 1.05)
    ax.set_title("Cascading Randomization — Spearman ρ Degradation")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def run_cascading_randomization(
    data_root: str = "data",
    checkpoint: str = "checkpoints/resnet50_iqothnc.pth",
    out_dir: str = "results/figures/cascading_randomization",
    n_per_class: int = 2,
    aug_views: int = 6,
    no_robustcam: bool = False,
    fine_grained: bool = False,
    seed: int = 42,
):
    _ensure_dir(out_dir)

    # choose coarse vs fine-grained state definitions
    if fine_grained:
        layer_order = FINE_LAYER_ORDER
        state_labels = FINE_STATE_LABELS
    else:
        layer_order = COARSE_LAYER_ORDER
        state_labels = COARSE_STATE_LABELS

    print("Selecting samples...")
    samples = _select_samples(data_root, n_per_class)
    print(f"  {len(samples)} images selected ({n_per_class} per class)")

    ds = DataService()
    aug_service = AugmentationService(seed=seed)

    print("Loading model...")
    ms = ModelService(arch="resnet50", checkpoint_path=checkpoint)

    # Pre-load all images
    sample_data = []
    for path, label, class_name in samples:
        tensor, pil_image = ds.get_image_tensor(path)
        tensor = tensor.to(ms.device)
        sample_data.append((path, label, class_name, tensor, pil_image))

    # state_heatmaps[state_idx] = list of {"gradcam": hm, "robustcam": hm|None}
    state_heatmaps: list[list[dict]] = []

    for state_idx, state_label in enumerate(state_labels):
        if state_idx > 0:
            group = layer_order[state_idx - 1]
            print(f"  Reinitializing {group}...")
            _reinit_layer_group(ms.model, group)
            ms.model.eval()

        print(f"State {state_idx}: {state_label}")
        per_image = []
        for path, label, class_name, tensor, pil_image in sample_data:
            gc_hm = _compute_gradcam_heatmap(ms, tensor, "layer4")
            rc_hm = None
            if not no_robustcam:
                rc_hm = _compute_robustcam_heatmap(
                    ms, pil_image, aug_service, ds, "layer4", n_views=aug_views
                )
            per_image.append({"gradcam": gc_hm, "robustcam": rc_hm})
        state_heatmaps.append(per_image)

    # Compute Spearman ρ vs state-0 for each sample
    n_samples = len(sample_data)
    gc_rho = [[0.0] * len(state_labels) for _ in range(n_samples)]
    rc_rho = [[0.0] * len(state_labels) for _ in range(n_samples)] if not no_robustcam else None

    svc = GradCAMService()
    for img_idx in range(n_samples):
        gc0 = svc.resize_heatmap_to_image(state_heatmaps[0][img_idx]["gradcam"], 224, 224)
        rc0 = None
        if not no_robustcam and state_heatmaps[0][img_idx]["robustcam"] is not None:
            rc0 = svc.resize_heatmap_to_image(state_heatmaps[0][img_idx]["robustcam"], 224, 224)

        for s_idx in range(len(state_labels)):
            gc_s = svc.resize_heatmap_to_image(state_heatmaps[s_idx][img_idx]["gradcam"], 224, 224)
            gc_rho[img_idx][s_idx] = _spearman_rho(gc0, gc_s)
            if rc_rho is not None and rc0 is not None:
                rc_s_raw = state_heatmaps[s_idx][img_idx].get("robustcam")
                if rc_s_raw is not None:
                    rc_s = svc.resize_heatmap_to_image(rc_s_raw, 224, 224)
                    rc_rho[img_idx][s_idx] = _spearman_rho(rc0, rc_s)

    # Per-sample grid figures
    print("Saving grid figures...")
    for img_idx, (path, label, class_name, tensor, pil_image) in enumerate(sample_data):
        state_results = [state_heatmaps[s][img_idx] for s in range(len(state_labels))]
        stem = os.path.splitext(os.path.basename(path))[0].replace(" ", "_")

        if fine_grained and n_per_class == 1:
            # single consolidated big grid
            fig = _build_big_grid(pil_image, state_results, state_labels, not no_robustcam)
            out_path = os.path.join(out_dir, f"{class_name}_{stem}_big_grid.png")
        else:
            fig = _build_sample_grid(pil_image, state_results, state_labels, not no_robustcam)
            out_path = os.path.join(out_dir, f"{class_name}_{stem}_grid.png")

        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path}")

    # Degradation plot
    print("Saving degradation plot...")
    sample_names = [
        os.path.splitext(os.path.basename(p))[0][:12]
        for p, *_ in sample_data
    ]
    class_names_list = [cls for _, _, cls, _, _ in sample_data]
    fig2 = _build_degradation_plot(
        state_labels, gc_rho, rc_rho, sample_names, class_names_list
    )
    rho_plot_path = os.path.join(out_dir, "degradation_spearman_rho.png")
    fig2.savefig(rho_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved {rho_plot_path}")

    # CSV
    rows = []
    for img_idx, (path, label, class_name, _, _) in enumerate(sample_data):
        stem = os.path.splitext(os.path.basename(path))[0]
        for method, rho_matrix in [("gradcam", gc_rho), ("robustcam", rc_rho)]:
            if rho_matrix is None:
                continue
            row = {"image": stem, "class": class_name, "method": method}
            for s_idx, s_label in enumerate(state_labels):
                row[s_label] = round(rho_matrix[img_idx][s_idx], 4)
            rows.append(row)

    csv_path = os.path.join(out_dir, "spearman_rho_table.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}")
    print(f"\nDone. All outputs in: {out_dir}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Cascading randomization sanity check for Grad-CAM and Robust-CAM."
    )
    p.add_argument("--data-root",    default="data")
    p.add_argument("--checkpoint",   default="checkpoints/resnet50_iqothnc.pth")
    p.add_argument("--out-dir",      default="results/figures/cascading_randomization")
    p.add_argument("--n-per-class",  default=2, type=int)
    p.add_argument("--aug-views",    default=6, type=int)
    p.add_argument("--no-robustcam",  action="store_true")
    p.add_argument("--fine-grained",  action="store_true",
                   help="Randomize individual Bottleneck blocks (17 states)")
    p.add_argument("--seed",          default=42, type=int)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_cascading_randomization(
        data_root=args.data_root,
        checkpoint=args.checkpoint,
        out_dir=args.out_dir,
        n_per_class=args.n_per_class,
        aug_views=args.aug_views,
        no_robustcam=args.no_robustcam,
        fine_grained=args.fine_grained,
        seed=args.seed,
    )
