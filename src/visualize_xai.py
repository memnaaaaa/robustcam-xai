"""
visualize_xai.py — Generate Grad-CAM / Robust-CAM / LIME overlay images for the report.

For each selected image, writes:
  - results/figures/qualitative/<class>_<stem>_gradcam_overlay.png
  - results/figures/qualitative/<class>_<stem>_robustcam_overlay.png
  - results/figures/qualitative/<class>_<stem>_lime_overlay.png
  - results/figures/qualitative/<class>_<stem>_xai_comparison.png
      Layout: 1 row × 4 cols — Original CT | Grad-CAM | Robust-CAM | LIME

Usage:
    python src/visualize_xai.py --images-per-class 1 --lime-samples 200
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import argparse

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from augmentation_service import AugmentationService
from data_service import DataService
from gradcam_service import GradCAMService
from iq_othncc_dataset import IQOTHNCCDDataset
from lime_service import LIMEService
from model_service import ModelService
from robust_cam import fuse_mean, warp_heatmap_back


CLASS_NAMES = {0: "Normal", 1: "Benign", 2: "Malignant"}


# ── helpers ───────────────────────────────────────────────────────────────────

def save_individual_overlay(overlay_rgb: np.ndarray, save_path: str) -> None:
    """Save RGB numpy array as PNG (OpenCV expects BGR)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))


def save_comparison_panel(
    pil_img,
    gradcam_overlay: np.ndarray,
    robustcam_overlay: np.ndarray,
    lime_overlay: np.ndarray,
    pred_class: int,
    confidence: float,
    save_path: str,
) -> None:
    """
    1 row × 4 cols: Original CT | Grad-CAM | Robust-CAM | LIME.
    Figure size 16×4 in, 150 dpi.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    images = [
        (np.array(pil_img.resize((224, 224))), "Original CT"),
        (gradcam_overlay, "Grad-CAM"),
        (robustcam_overlay, "Robust-CAM\n(augmentation fusion)"),
        (lime_overlay, "LIME"),
    ]
    for ax, (img, title) in zip(axes, images):
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle(
        f"Predicted: {CLASS_NAMES[pred_class]}  (conf={confidence:.2f})",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Robust-CAM augmentation sweep ─────────────────────────────────────────────

def run_robust_cam(
    pil_img,
    ms: ModelService,
    ds: DataService,
    gc: GradCAMService,
    aug_svc: AugmentationService,
) -> np.ndarray:
    """
    Runs Grad-CAM over all augmented views of pil_img, warps each heatmap
    back to original coordinates, and returns the mean-fused heatmap (float32 [224,224]).
    """
    aug_dict = aug_svc.apply(pil_img)  # {name: (aug_pil, meta)}
    warped_heatmaps = []
    ms.register_hooks_by_name(["layer4"])

    for name, (aug_pil, meta) in aug_dict.items():
        try:
            aug_tensor = ds.preprocess(aug_pil).to(ms.device)
            _, acts, grads = ms.run(aug_tensor)
            hm = gc.compute_raw_heatmap(acts["layer4"], grads["layer4"])
            hm = cv2.resize(hm, (224, 224))
            warped = warp_heatmap_back(hm, meta, (224, 224))
            warped_heatmaps.append(warped)
        except Exception as e:
            print(f"  [Warning] Robust-CAM aug '{name}' failed: {e}")

    if not warped_heatmaps:
        # Fallback: return a zero heatmap if all augmentations failed
        return np.zeros((224, 224), dtype=np.float32)

    return fuse_mean(warped_heatmaps)


# ── per-image processing ──────────────────────────────────────────────────────

def process_image(
    img_path: str,
    label: int,
    ms: ModelService,
    ds: DataService,
    gc: GradCAMService,
    aug_svc: AugmentationService,
    lime_svc: LIMEService,
    results_dir: str,
    lime_num_samples: int,
) -> None:
    stem = os.path.splitext(os.path.basename(img_path))[0].replace(" ", "_")
    class_prefix = CLASS_NAMES[label].lower()
    qual_dir = os.path.join(results_dir, "figures", "qualitative")

    # ── Load + baseline Grad-CAM ──────────────────────────────────────────────
    tensor, pil_img = ds.get_image_tensor(img_path)
    tensor = tensor.to(ms.device)

    ms.register_hooks_by_name(["layer4"])
    pred_class, acts, grads = ms.run(tensor)
    hm_gradcam = gc.compute_raw_heatmap(acts["layer4"], grads["layer4"])
    hm_gradcam = cv2.resize(hm_gradcam, (224, 224))

    with torch.no_grad():
        logits = ms.model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    confidence = probs[pred_class].item()

    img_np = np.array(pil_img.resize((224, 224)))
    gradcam_overlay = gc.overlay(img_np, hm_gradcam)

    # ── Robust-CAM augmentation sweep ────────────────────────────────────────
    hm_robustcam = run_robust_cam(pil_img, ms, ds, gc, aug_svc)
    robustcam_overlay = gc.overlay(img_np, hm_robustcam)

    # ── LIME ─────────────────────────────────────────────────────────────────
    predict_fn = lime_svc.build_predict_fn(ms, ds)
    hm_lime = lime_svc.explain(
        pil_img, predict_fn, target_class=pred_class, image_size=224
    )
    lime_overlay = gc.overlay(img_np, hm_lime)

    # ── Save individual overlays ──────────────────────────────────────────────
    for method_name, overlay in [
        ("gradcam",    gradcam_overlay),
        ("robustcam",  robustcam_overlay),
        ("lime",       lime_overlay),
    ]:
        save_individual_overlay(
            overlay,
            os.path.join(qual_dir, f"{class_prefix}_{stem}_{method_name}_overlay.png"),
        )

    # ── Save 4-panel comparison figure ───────────────────────────────────────
    save_comparison_panel(
        pil_img,
        gradcam_overlay,
        robustcam_overlay,
        lime_overlay,
        pred_class,
        confidence,
        os.path.join(qual_dir, f"{class_prefix}_{stem}_xai_comparison.png"),
    )

    print(
        f"  [{CLASS_NAMES[label]}] {stem}: "
        f"pred={CLASS_NAMES[pred_class]}, conf={confidence:.3f}"
    )


# ── main entry ────────────────────────────────────────────────────────────────

def run(
    data_root: str = "data",
    checkpoint_path: str = "checkpoints/resnet50_iqothnc.pth",
    results_dir: str = "results",
    images_per_class: int = 1,
    lime_num_samples: int = 500,
    seed: int = 42,
) -> None:
    ms = ModelService(arch="resnet50", checkpoint_path=checkpoint_path)
    ds = DataService()
    gc = GradCAMService()
    aug_svc = AugmentationService(seed=seed)
    lime_svc = LIMEService(num_samples=lime_num_samples, random_state=seed)

    # Collect images_per_class per label from the test split
    dataset = IQOTHNCCDDataset(data_root, split="test", seed=seed)
    test_samples = dataset.get_all_samples()  # list of (path, label, class_name)

    by_class: dict[int, list[tuple[str, int]]] = {0: [], 1: [], 2: []}
    for path, label, _ in test_samples:
        if len(by_class[label]) < images_per_class:
            by_class[label].append((path, label))

    for label in [0, 1, 2]:
        for img_path, lbl in by_class[label]:
            process_image(
                img_path, lbl, ms, ds, gc, aug_svc, lime_svc,
                results_dir, lime_num_samples,
            )

    qual_dir = os.path.join(results_dir, "figures", "qualitative")
    print(f"\nAll outputs saved to {qual_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Generate Grad-CAM / Robust-CAM / LIME overlays for the report"
    )
    p.add_argument("--data-root",        default="data")
    p.add_argument("--checkpoint",       default="checkpoints/resnet50_iqothnc.pth")
    p.add_argument("--results-dir",      default="results")
    p.add_argument("--images-per-class", default=1, type=int)
    p.add_argument("--lime-samples",     default=500, type=int)
    p.add_argument("--seed",             default=42, type=int)
    args = p.parse_args()
    run(
        data_root=args.data_root,
        checkpoint_path=args.checkpoint,
        results_dir=args.results_dir,
        images_per_class=args.images_per_class,
        lime_num_samples=args.lime_samples,
        seed=args.seed,
    )
