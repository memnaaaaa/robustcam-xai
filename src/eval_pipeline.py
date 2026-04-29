# src/eval_pipeline.py
# Phase 5: Batch evaluation pipeline for RobustCAM.
# Produces results/tables/metrics_table_<split>.csv, bar charts, classification report,
# and per-image qualitative panels. Logs everything to MLflow.

import os
import sys
import warnings
import argparse
import time

# Force UTF-8 stdout/stderr on Windows to avoid emoji encoding errors in dependencies
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Windows
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report as sk_classification_report

# ── ensure src/ is on path when run from project root ──────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from model_service import ModelService
from data_service import DataService
from gradcam_service import GradCAMService
from augmentation_service import AugmentationService
from robust_cam import fuse_mean, fuse_median, compute_uncertainty, warp_heatmap_back
from faithfulness_metrics import compute_all_metrics
from lime_service import LIMEService
from shap_service import SHAPService
from xai_fusion import compute_voting_mask, voting_mask_to_colormap
from iq_othncc_dataset import IQOTHNCCDDataset
from mlflow_service import MLflowService

_ARCH_DEFAULT_LAYERS: dict[str, list[str]] = {
    "resnet50":        ["layer3", "layer4"],
    "resnet101":       ["layer3", "layer4"],
    "densenet161":     ["features.denseblock3", "features.denseblock4"],
    "efficientnet_b0": ["features.6", "features.7", "features.8"],
    "vit_b_16":        ["encoder.layers.encoder_layer_10",
                        "encoder.layers.encoder_layer_11"],
    "vgg16":           ["features.14", "features.20", "features.30"],
}


# ─── small helpers ─────────────────────────────────────────────────────────────

def _get_softmax_probs(ms: ModelService, tensor: torch.Tensor) -> np.ndarray:
    """Return 3-class softmax probabilities as a float32 array."""
    with torch.no_grad():
        logits, _ = ms.forward(tensor.to(ms.device))
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy().astype(np.float32)
    return probs


def _make_gradcam_fn(ms: ModelService, gc: GradCAMService, layer: str, target_class: int):
    """
    Returns a function that accepts a torch.Tensor [1,3,224,224] and returns a
    float32 [224,224] Grad-CAM heatmap. Used for stability / consistency metrics.
    """
    def fn(t: torch.Tensor) -> np.ndarray:
        _, acts, grads = ms.run(t.to(ms.device), target_class=target_class)
        if layer not in acts or acts[layer] is None:
            return np.zeros((224, 224), dtype=np.float32)
        hm = gc.compute_raw_heatmap(acts[layer], grads[layer])
        return gc.resize_heatmap_to_image(hm, 224, 224)
    return fn


def _overlay_hm(gc: GradCAMService, pil_img: Image.Image, hm: np.ndarray) -> np.ndarray:
    """Produce a 224×224 RGB overlay (numpy uint8) from a PIL image + float32 heatmap."""
    img_np = np.array(pil_img.convert("RGB").resize((224, 224)))
    hm_224 = gc.resize_heatmap_to_image(hm, 224, 224)
    return gc.overlay(img_np, hm_224, alpha=0.4)


def _log_metrics_mlflow(mlf: MLflowService, prefix: str, metrics: dict) -> None:
    """Log metric dict to MLflow via MLflowService, skipping NaN values."""
    mlf.log_metrics_dict(metrics, prefix=prefix)


def _save_panel(
    pil_img: Image.Image,
    hm_gc: np.ndarray,
    hm_rc: np.ndarray,
    uncertainty: np.ndarray,
    hm_lime: np.ndarray,
    hm_shap: np.ndarray,
    voting_cm: np.ndarray,
    softmax_probs: np.ndarray,
    title: str,
    save_path: str,
    gc_svc: GradCAMService,
) -> None:
    """
    Save a 2-row × 4-column qualitative XAI panel to save_path.
    Row 1: Original CT | Grad-CAM | Robust-CAM | Uncertainty
    Row 2: LIME        | SHAP     | Voting mask | Confidence bar
    """
    class_names = ["Normal", "Benign", "Malignant"]
    img224 = np.array(pil_img.convert("RGB").resize((224, 224)))

    ov_gc = _overlay_hm(gc_svc, pil_img, hm_gc)
    ov_rc = _overlay_hm(gc_svc, pil_img, hm_rc)
    unc_norm = (uncertainty / (uncertainty.max() + 1e-8) * 255).astype(np.uint8)
    unc_color = cv2.applyColorMap(unc_norm, cv2.COLORMAP_MAGMA)
    unc_color = cv2.cvtColor(unc_color, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(title, fontsize=12)

    # Row 1
    axes[0, 0].imshow(img224);        axes[0, 0].set_title("Original CT")
    axes[0, 1].imshow(ov_gc);         axes[0, 1].set_title("Grad-CAM")
    axes[0, 2].imshow(ov_rc);         axes[0, 2].set_title("Robust-CAM")
    axes[0, 3].imshow(unc_color);     axes[0, 3].set_title("Uncertainty")

    # Row 2 — LIME, SHAP, Voting mask
    axes[1, 0].imshow(_overlay_hm(gc_svc, pil_img, hm_lime));  axes[1, 0].set_title("LIME")
    axes[1, 1].imshow(_overlay_hm(gc_svc, pil_img, hm_shap));  axes[1, 1].set_title("SHAP")
    axes[1, 2].imshow(voting_cm);                               axes[1, 2].set_title("Voting mask")

    # Row 2 — Confidence bar
    colors = ["#4CAF50" if i == int(np.argmax(softmax_probs)) else "#90CAF9"
              for i in range(len(class_names))]
    axes[1, 3].barh(class_names, softmax_probs, color=colors)
    axes[1, 3].set_xlim(0, 1)
    axes[1, 3].set_xlabel("Confidence")
    axes[1, 3].set_title("Class confidence")

    for ax in axes.ravel():
        if ax != axes[1, 3]:
            ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _save_bar_charts(
    agg: dict[str, dict[str, float]],
    results_dir: str,
    arch_tag: str = "",
) -> None:
    """
    Save quantitative bar chart PNGs to results/figures/quantitative/.
    agg: { method_name: { metric_key: mean_value, ... }, ... }
    arch_tag: appended to each filename, e.g. "resnet50" → metric_comparison_faith_resnet50.png
    """
    quant_dir = os.path.join(results_dir, "figures", "quantitative")
    os.makedirs(quant_dir, exist_ok=True)

    suffix = f"_{arch_tag}" if arch_tag else ""
    methods = list(agg.keys())

    def _bar(metric_keys: list[str], base: str, ylabel: str, title: str) -> None:
        fig, axes = plt.subplots(1, len(metric_keys), figsize=(5 * len(metric_keys), 4),
                                 squeeze=False)
        for col_idx, metric in enumerate(metric_keys):
            ax = axes[0, col_idx]
            vals = [agg[m].get(metric, float("nan")) for m in methods]
            colors = ["#FF7043" if np.isnan(v) else "#42A5F5" for v in vals]
            bars = ax.bar(methods, vals, color=colors, edgecolor="black", linewidth=0.7)
            ax.set_title(metric)
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, max(1.0, max((v for v in vals if not np.isnan(v)), default=1.0)) * 1.15)
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=8)
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        fig.suptitle(f"{title} [{arch_tag}]" if arch_tag else title, fontsize=11)
        plt.tight_layout()
        fig.savefig(os.path.join(quant_dir, f"{base}{suffix}.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)

    _bar(["faith"],    "metric_comparison_faith",    "score", "Perturbation Faithfulness (↑)")
    _bar(["fidelity"], "metric_comparison_fidelity", "score", "XAI Fidelity (↑)")
    _bar(["stability"], "metric_comparison_stability", "score", "XAI Stability (↑)")
    _bar(["consist_iou", "consist_pearson"], "metric_comparison_consistency", "score",
         "Explanation Consistency (↑)")
    _bar(["mean_variance", "mean_iou_topk", "mean_spearman"],
         "robustcam_stability_metrics", "score", "Robust-CAM Augmentation Stability")


# ─── main pipeline ─────────────────────────────────────────────────────────────

def run_eval_pipeline(
    data_root: str = "data",
    checkpoint_path: str = "checkpoints/resnet50_iqothnc.pth",
    arch: str = "resnet50",
    layers: list[str] = None,
    split: str = "test",
    num_aug: int = 6,
    fusion_method: str = "mean",
    run_lime: bool = True,
    run_shap: bool = True,
    n_shap_background: int = 10,
    lime_num_samples: int = 500,
    max_images: int = None,
    results_dir: str = "results",
    experiment_name: str = "RobustCAM_ResNet50_IQ_OTH_NCCD",
    run_name: str = None,
    skip_panels: bool = False,
) -> None:
    """
    Batch evaluation pipeline. Runs Grad-CAM, Robust-CAM, LIME, SHAP, and voting mask,
    computing all 9 faithfulness metrics per image per method.

    Produces:
        results/tables/metrics_table_<split>.csv
        results/tables/classification_report.csv
        results/figures/qualitative/*.png
        results/figures/quantitative/*.png
    All metrics logged to MLflow.
    """
    if layers is None:
        layers = _ARCH_DEFAULT_LAYERS.get(arch, ["layer3", "layer4"])

    # ── ensure output dirs exist ───────────────────────────────────────────────
    qual_dir   = os.path.join(results_dir, "figures", "qualitative")
    quant_dir  = os.path.join(results_dir, "figures", "quantitative")
    tables_dir = os.path.join(results_dir, "tables")
    for d in [qual_dir, quant_dir, tables_dir]:
        os.makedirs(d, exist_ok=True)

    # ── val_acc from checkpoint ────────────────────────────────────────────────
    val_acc_ckpt = float("nan")
    if os.path.exists(checkpoint_path):
        try:
            ckpt_info = torch.load(checkpoint_path, map_location="cpu")
            val_acc_ckpt = float(ckpt_info.get("val_acc", float("nan")))
        except Exception as e:
            print(f"[Warning] Could not read val_acc from checkpoint: {e}")

    # ── instantiate services once ──────────────────────────────────────────────
    print("\n[eval_pipeline] Initializing services...")
    ms  = ModelService(arch=arch, checkpoint_path=checkpoint_path)
    ms.register_hooks_by_name(layers)
    ds  = DataService()
    gc  = GradCAMService()
    aug_svc = AugmentationService(seed=42)

    dataset    = IQOTHNCCDDataset(data_root, split=split, seed=42)
    train_ds   = IQOTHNCCDDataset(data_root, split="train", seed=42)
    all_samples = dataset.get_all_samples()
    if max_images is not None:
        all_samples = all_samples[:max_images]

    lime_svc     = None
    lime_pred_fn = None
    if run_lime:
        print("[eval_pipeline] Building LIME predict_fn...")
        lime_svc     = LIMEService(num_samples=lime_num_samples, random_state=42)
        lime_pred_fn = lime_svc.build_predict_fn(ms, ds)

    shap_svc  = None
    bg_tensor = None
    if run_shap:
        print("[eval_pipeline] Building SHAP background tensor...")
        train_paths = [p for p, _, _ in train_ds.get_all_samples()]
        bg_tensor   = SHAPService.build_background_tensor(ds, train_paths,
                                                          n_background=n_shap_background)
        shap_svc    = SHAPService(model_service=ms, background_tensor=bg_tensor)

    # ── MLflow setup ──────────────────────────────────────────────────────────
    mlf = MLflowService(experiment_name=experiment_name)
    mlf.start_run(
        run_name=run_name,
        params={
            "arch": arch,
            "split": split,
            "layers": str(layers),
            "num_aug": num_aug,
            "fusion_method": fusion_method,
            "lime_num_samples": lime_num_samples,
            "n_shap_background": n_shap_background,
            "max_images": max_images if max_images is not None else len(all_samples),
            "checkpoint_path": checkpoint_path,
        },
    )

    # ── metric accumulators ───────────────────────────────────────────────────
    method_names = [
        "Grad-CAM", "Robust-CAM (mean)", "Robust-CAM (median)",
        "LIME", "SHAP", "Voting mask",
    ]

    # {method: {metric_key: [values...]}}
    accum: dict[str, dict[str, list]] = {m: {} for m in method_names}

    y_true: list[int] = []
    y_pred: list[int] = []

    primary_layer = layers[-1]  # e.g., "layer4"

    try:
        for img_idx, (path, label, class_name) in enumerate(all_samples):
            stem = os.path.splitext(os.path.basename(path))[0]
            prefix = f"{class_name.lower()}_{stem}"
            print(f"\n[{img_idx+1}/{len(all_samples)}] {class_name} | {stem}")
            t0 = time.time()

            # ── load image & baseline Grad-CAM ────────────────────────────────
            pil_img = ds.load_image(path)
            tensor  = ds.preprocess(pil_img).to(ms.device)   # [1,3,224,224]

            pred_class, acts, grads = ms.run(tensor)
            y_true.append(label)
            y_pred.append(pred_class)

            hm_gc = gc.resize_heatmap_to_image(
                gc.compute_raw_heatmap(acts[primary_layer], grads[primary_layer]), 224, 224
            )

            softmax_probs = _get_softmax_probs(ms, tensor)
            print(f"  pred={pred_class} (true={label}), conf={softmax_probs[pred_class]:.3f}")

            # Grad-CAM heatmap_fn for stability/consistency
            gradcam_fn = _make_gradcam_fn(ms, gc, primary_layer, pred_class)

            # ── Robust-CAM sweep ──────────────────────────────────────────────
            aug_views    = aug_svc.apply(pil_img)
            aug_heatmaps = []
            for aug_name, (aug_pil, meta) in aug_views.items():
                aug_tensor = ds.preprocess(aug_pil).to(ms.device)
                _, aug_acts, aug_grads = ms.run(aug_tensor, target_class=pred_class)
                if primary_layer not in aug_acts or aug_acts[primary_layer] is None:
                    continue
                aug_hm_raw = gc.compute_raw_heatmap(aug_acts[primary_layer], aug_grads[primary_layer])
                warped     = warp_heatmap_back(aug_hm_raw, meta, (224, 224))
                aug_heatmaps.append(warped.astype(np.float32))

            hm_rc_mean   = fuse_mean(aug_heatmaps)
            hm_rc_median = fuse_median(aug_heatmaps)
            uncertainty  = compute_uncertainty(aug_heatmaps)

            # ── LIME ──────────────────────────────────────────────────────────
            hm_lime = np.zeros((224, 224), dtype=np.float32)
            if run_lime and lime_svc is not None:
                try:
                    hm_lime = gc.resize_heatmap_to_image(
                        lime_svc.explain(pil_img, lime_pred_fn, pred_class), 224, 224
                    )
                    print("  LIME done")
                except Exception as e:
                    print(f"  [Warning] LIME failed: {e}")

            # ── SHAP ──────────────────────────────────────────────────────────
            hm_shap = np.zeros((224, 224), dtype=np.float32)
            if run_shap and shap_svc is not None:
                try:
                    hm_shap = gc.resize_heatmap_to_image(
                        shap_svc.explain(ms, tensor, target_class=pred_class), 224, 224
                    )
                    print("  SHAP done")
                except Exception as e:
                    print(f"  [Warning] SHAP failed: {e}")

            # ── Voting mask ───────────────────────────────────────────────────
            voting_mask = compute_voting_mask(hm_gc, hm_lime, hm_shap)
            voting_cm   = voting_mask_to_colormap(voting_mask)
            hm_voting   = (voting_mask.astype(np.float32) / 3.0).astype(np.float32)

            # ── compute all 9 metrics per method ─────────────────────────────
            metrics_by_method: dict[str, dict] = {}

            # Grad-CAM baseline
            metrics_by_method["Grad-CAM"] = compute_all_metrics(
                ms, tensor, hm_gc, pred_class,
                heatmap_fn=gradcam_fn,
                aug_heatmaps=None,
                fused_heatmap=None,
            )

            # Robust-CAM (mean)
            metrics_by_method["Robust-CAM (mean)"] = compute_all_metrics(
                ms, tensor, hm_rc_mean, pred_class,
                heatmap_fn=gradcam_fn,
                aug_heatmaps=aug_heatmaps,
                fused_heatmap=hm_rc_mean,
            )

            # Robust-CAM (median)
            metrics_by_method["Robust-CAM (median)"] = compute_all_metrics(
                ms, tensor, hm_rc_median, pred_class,
                heatmap_fn=gradcam_fn,
                aug_heatmaps=aug_heatmaps,
                fused_heatmap=hm_rc_median,
            )

            # Only compute metrics for LIME/SHAP/Voting when the methods actually ran;
            # a zeros heatmap would zero the whole image at 80th-percentile threshold,
            # corrupting faith/fidelity for every other method in the comparison table.
            _nan_metrics = {k: float("nan") for k in [
                "faith", "loc_acc", "consist_iou", "fidelity", "stability",
                "consist_pearson", "mean_variance", "mean_iou_topk", "mean_spearman",
            ]}

            if run_lime and lime_svc is not None and hm_lime.max() > 0:
                metrics_by_method["LIME"] = compute_all_metrics(
                    ms, tensor, hm_lime, pred_class,
                    heatmap_fn=None, aug_heatmaps=None, fused_heatmap=None,
                )
            else:
                metrics_by_method["LIME"] = dict(_nan_metrics)

            if run_shap and shap_svc is not None and hm_shap.max() > 0:
                metrics_by_method["SHAP"] = compute_all_metrics(
                    ms, tensor, hm_shap, pred_class,
                    heatmap_fn=None, aug_heatmaps=None, fused_heatmap=None,
                )
            else:
                metrics_by_method["SHAP"] = dict(_nan_metrics)

            if run_lime and run_shap and hm_lime.max() > 0 and hm_shap.max() > 0:
                metrics_by_method["Voting mask"] = compute_all_metrics(
                    ms, tensor, hm_voting, pred_class,
                    heatmap_fn=None, aug_heatmaps=None, fused_heatmap=None,
                )
            else:
                metrics_by_method["Voting mask"] = dict(_nan_metrics)

            # ── accumulate & log per-image metrics ────────────────────────────
            for method, mdict in metrics_by_method.items():
                for k, v in mdict.items():
                    accum[method].setdefault(k, []).append(v)
                _log_metrics_mlflow(mlf, f"{prefix}_{method.lower().replace(' ', '_')}", mdict)

            # ── qualitative panel ─────────────────────────────────────────────
            if not skip_panels:
                panel_path = os.path.join(qual_dir, f"{class_name.lower()}_{stem}_xai_panel.png")
                _save_panel(
                    pil_img, hm_gc, hm_rc_mean, uncertainty,
                    hm_lime, hm_shap, voting_cm,
                    softmax_probs,
                    title=f"{class_name} | pred={pred_class} | conf={softmax_probs[pred_class]:.3f}",
                    save_path=panel_path,
                    gc_svc=gc,
                )
                print(f"  panel saved → {os.path.basename(panel_path)}")

            elapsed = time.time() - t0
            print(f"  [{img_idx+1}/{len(all_samples)}] done ({elapsed:.1f}s)")

        # ── post-loop: aggregate metrics ──────────────────────────────────────
        print("\n[eval_pipeline] Aggregating metrics...")

        agg: dict[str, dict[str, float]] = {}
        for method in method_names:
            agg[method] = {}
            for metric_key, vals in accum.get(method, {}).items():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    mean_val = float(np.nanmean(vals)) if vals else float("nan")
                agg[method][metric_key] = mean_val

        # log aggregate metrics to MLflow via MLflowService
        for method, mdict in agg.items():
            clean_method = method.lower().replace(" ", "_").replace("(", "").replace(")", "")
            mlf.log_metrics_dict(mdict, prefix=f"mean_{clean_method}")

        if not np.isnan(val_acc_ckpt):
            mlf.log_scalar("val_acc_from_checkpoint", val_acc_ckpt)

        # ── export CSV table ──────────────────────────────────────────────────
        metric_cols = [
            "faith", "loc_acc", "consist_iou",
            "fidelity", "stability", "consist_pearson",
            "mean_variance", "mean_iou_topk", "mean_spearman",
        ]
        rows = []
        for method in method_names:
            row = {"Method": method}
            for col in metric_cols:
                v = agg.get(method, {}).get(col, float("nan"))
                row[col] = round(v, 6) if not np.isnan(v) else float("nan")
            rows.append(row)

        df_metrics = pd.DataFrame(rows, columns=["Method"] + metric_cols)
        csv_path = os.path.join(tables_dir, f"metrics_table_{arch}_{split}.csv")
        df_metrics.to_csv(csv_path, index=False)
        print(f"\nMetrics table written → {csv_path}")
        print(df_metrics.to_string(index=False))

        # ── classification report ─────────────────────────────────────────────
        if y_true and y_pred:
            class_names_list = ["Normal", "Benign", "Malignant"]
            report_dict = sk_classification_report(
                y_true, y_pred,
                labels=list(range(len(class_names_list))),
                target_names=class_names_list,
                output_dict=True,
                zero_division=0,
            )
            report_rows = []
            for cls in class_names_list + ["macro avg", "weighted avg"]:
                if cls in report_dict:
                    r = report_dict[cls]
                    report_rows.append({
                        "class": cls,
                        "precision": round(r.get("precision", float("nan")), 4),
                        "recall":    round(r.get("recall",    float("nan")), 4),
                        "f1-score":  round(r.get("f1-score",  float("nan")), 4),
                        "support":   int(r.get("support", 0)),
                    })
            df_report = pd.DataFrame(report_rows)
            report_path = os.path.join(tables_dir, f"classification_report_{arch}.csv")
            df_report.to_csv(report_path, index=False)
            print(f"\nClassification report → {report_path}")
            print(df_report.to_string(index=False))

        # ── bar charts ────────────────────────────────────────────────────────
        _save_bar_charts(agg, results_dir, arch_tag=arch)
        print(f"\nBar charts written → {quant_dir}/")

    finally:
        mlf.end_run()

    print("\n[eval_pipeline] Done.")


# ─── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="RobustCAM Phase 5 evaluation pipeline")
    p.add_argument("--data-root",    default="data")
    p.add_argument("--checkpoint",   default="checkpoints/resnet50_iqothnc.pth",
                   dest="checkpoint_path")
    p.add_argument("--arch",         default="resnet50")
    p.add_argument("--layers",       default=["layer3", "layer4"], nargs="+")
    p.add_argument("--split",        default="test",
                   choices=["train", "val", "test", "all"])
    p.add_argument("--num-aug",      default=6, type=int, dest="num_aug")
    p.add_argument("--fusion",       default="mean", choices=["mean", "median"],
                   dest="fusion_method")
    p.add_argument("--max-images",   default=None, type=int, dest="max_images")
    p.add_argument("--results-dir",  default="results", dest="results_dir")
    p.add_argument("--lime-samples", default=500, type=int, dest="lime_num_samples")
    p.add_argument("--shap-background", default=10, type=int, dest="n_shap_background")
    p.add_argument("--no-lime",      action="store_true", dest="no_lime")
    p.add_argument("--no-shap",      action="store_true", dest="no_shap")
    p.add_argument("--skip-panels",  action="store_true", dest="skip_panels",
                   help="Skip saving qualitative XAI panel figures (metrics only)")
    p.add_argument("--run-name",     default=None, dest="run_name")
    p.add_argument("--experiment",   default="RobustCAM_ResNet50_IQ_OTH_NCCD",
                   dest="experiment_name")
    args = p.parse_args()

    run_eval_pipeline(
        data_root=args.data_root,
        checkpoint_path=args.checkpoint_path,
        arch=args.arch,
        layers=args.layers,
        split=args.split,
        num_aug=args.num_aug,
        fusion_method=args.fusion_method,
        run_lime=not args.no_lime,
        run_shap=not args.no_shap,
        n_shap_background=args.n_shap_background,
        lime_num_samples=args.lime_num_samples,
        max_images=args.max_images,
        results_dir=args.results_dir,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        skip_panels=args.skip_panels,
    )
