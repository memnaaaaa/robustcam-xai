# src/faithfulness_metrics.py
# Unified 9-metric evaluation suite for XAI explanation faithfulness.
#
# Group A — Panboonyuen 2026 faithfulness metrics (Faith, LocAcc, Consist_IoU)
# Group B — Akgündoğdu 2025 explainability quality metrics (Fid, Stab, Cons_Pearson)
# Group C — RobustCAM augmentation stability (Var, IoU_k, Spear) — imported from robust_cam.py

import numpy as np
import torch
from scipy.stats import pearsonr

from robust_cam import global_stability_metrics


# ─── helpers ──────────────────────────────────────────────────────────────────

def _binarize(heatmap: np.ndarray, threshold_percentile: float) -> np.ndarray:
    """Binarize float32 [H,W] heatmap at the given percentile. Returns uint8 {0,1}."""
    thresh = np.percentile(heatmap, threshold_percentile)
    return (heatmap >= thresh).astype(np.uint8)


def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Intersection-over-union of two binary uint8 masks."""
    inter = (mask_a & mask_b).sum()
    union = (mask_a | mask_b).sum()
    return float(inter) / (float(union) + 1e-8)


def _forward_softmax(model_service, input_tensor: torch.Tensor, pred_class: int) -> float:
    """
    Run a no-grad forward pass and return softmax confidence for pred_class.
    Uses model_service.forward() directly (does not touch hook gradients).
    """
    with torch.no_grad():
        output, _ = model_service.forward(input_tensor)
        return torch.softmax(output, dim=1)[0, pred_class].item()


def _mask_tensor(
    input_tensor: torch.Tensor,
    heatmap: np.ndarray,
    threshold_percentile: float,
    invert: bool = False,
) -> torch.Tensor:
    """
    Build a [1,1,H,W] binary mask from heatmap and apply it to input_tensor.
    invert=False  → keep NON-salient pixels (zero salient ones out).
    invert=True   → keep salient pixels only (zero non-salient ones out).
    """
    mask_np = _binarize(heatmap, threshold_percentile).astype(np.float32)  # [H,W] in {0,1}
    mask_t  = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(input_tensor.device)
    if invert:
        return input_tensor * mask_t           # keep salient
    else:
        return input_tensor * (1.0 - mask_t)   # zero salient


# ─── Group A: Panboonyuen 2026 ────────────────────────────────────────────────

def perturbation_faithfulness(
    model_service,
    input_tensor: torch.Tensor,
    heatmap: np.ndarray,
    pred_class: int,
    threshold_percentile: float = 80.0,
) -> float:
    """
    Panboonyuen Eq. 5 — perturbation-based faithfulness.
        Faith(x) = softmax(f(x))[y] - softmax(f(x ⊙ (1 - mask)))[y]

    Binarizes heatmap at threshold_percentile, zeros those pixels in the input,
    re-runs a forward pass, and returns the confidence drop.
    Higher values indicate the heatmap highlights causally important pixels.
    """
    orig_conf  = _forward_softmax(model_service, input_tensor, pred_class)
    perturbed  = _mask_tensor(input_tensor, heatmap, threshold_percentile, invert=False)
    pert_conf  = _forward_softmax(model_service, perturbed, pred_class)
    return float(orig_conf - pert_conf)


def localization_accuracy(
    heatmap: np.ndarray,
    gt_mask: np.ndarray | None,
    threshold_percentile: float = 80.0,
) -> float:
    """
    Panboonyuen Eq. 4 — localization accuracy.
        LocAcc = |CAM_binary ∩ GT_mask| / |GT_mask|

    Returns np.nan when gt_mask is None (IQ-OTH/NCCD has no pixel annotations).
    """
    if gt_mask is None:
        return np.nan
    cam_binary = _binarize(heatmap, threshold_percentile)
    gt_binary  = (gt_mask > 0).astype(np.uint8)
    gt_total   = int(gt_binary.sum())
    if gt_total == 0:
        return np.nan
    return float((cam_binary & gt_binary).sum()) / float(gt_total)


def explanation_consistency(
    heatmaps: list[np.ndarray],
    reference_heatmap: np.ndarray | None = None,
    threshold_percentile: float = 80.0,
) -> float:
    """
    Panboonyuen Eq. 6 — explanation consistency (IoU-based).
        Consist = (1/R) * sum_r IoU(CAM_r_binary, CAM_ref_binary)

    Binarizes each heatmap and the reference at threshold_percentile.
    Reference defaults to the mean of all heatmaps if not provided.
    Returns mean IoU in [0, 1].
    """
    if len(heatmaps) == 0:
        return np.nan
    if reference_heatmap is None:
        reference_heatmap = np.mean(np.stack(heatmaps, axis=0), axis=0)
    ref_bin = _binarize(reference_heatmap, threshold_percentile)
    ious = [_iou(_binarize(hm, threshold_percentile), ref_bin) for hm in heatmaps]
    return float(np.mean(ious))


# ─── Group B: Akgündoğdu 2025 ────────────────────────────────────────────────

def xai_fidelity(
    model_service,
    input_tensor: torch.Tensor,
    heatmap: np.ndarray,
    pred_class: int,
    threshold_percentile: float = 80.0,
) -> float:
    """
    Akgündoğdu Eq. 11-15 — fidelity.
        Fid+(ψ) = I{y == f(x)} - I{y == f(x ⊙ (1-mask))}
        Fid-(ψ) = I{y == f(x)} - I{y == f(x ⊙ mask)}
        Return Fid+(ψ) - Fid-(ψ)

    Positive return value means salient regions are causally important.
    All three forward passes use torch.no_grad().
    """
    with torch.no_grad():
        _, pred_orig       = model_service.forward(input_tensor)
        masked_out         = _mask_tensor(input_tensor, heatmap, threshold_percentile, invert=False)
        _, pred_masked_out = model_service.forward(masked_out)
        masked_only        = _mask_tensor(input_tensor, heatmap, threshold_percentile, invert=True)
        _, pred_masked_only = model_service.forward(masked_only)

    fid_plus  = float(pred_orig == pred_class) - float(pred_masked_out  == pred_class)
    fid_minus = float(pred_orig == pred_class) - float(pred_masked_only == pred_class)
    return float(fid_plus - fid_minus)


def xai_stability(
    model_service,
    input_tensor: torch.Tensor,
    heatmap_fn,
    noise_std: float = 0.05,
    n_trials: int = 5,
) -> float:
    """
    Akgündoğdu Eq. 16-17 — stability.
        Stability(I) = mean_trials ρ(E(I), E(I + ε))

    Adds Gaussian noise ε ~ N(0, noise_std²) to the input tensor,
    recomputes the explanation via heatmap_fn, and measures Pearson r.
    Returns np.nan when heatmap_fn is None (cannot recompute on perturbed input).
    """
    if heatmap_fn is None:
        return np.nan

    orig_heatmap = heatmap_fn(input_tensor)
    orig_flat    = orig_heatmap.ravel()

    correlations = []
    for _ in range(n_trials):
        noise        = torch.randn_like(input_tensor) * noise_std
        noisy_input  = input_tensor + noise
        noisy_heatmap = heatmap_fn(noisy_input)
        try:
            r, _ = pearsonr(orig_flat, noisy_heatmap.ravel())
            if not np.isnan(r):
                correlations.append(float(r))
        except Exception:
            pass

    return float(np.mean(correlations)) if correlations else np.nan


def xai_consistency_pearson(
    heatmaps: list[np.ndarray],
) -> float:
    """
    Akgündoğdu Eq. 18 — consistency (Pearson variant).
        Consistency(I) = (2 / n(n-1)) * sum_{i<j} ρ(E_i(I), E_j(I))

    Computes average pairwise Pearson correlation across n repeated explanation runs.
    Returns 1.0 when only one heatmap is provided (trivially consistent).
    """
    n = len(heatmaps)
    if n < 2:
        return 1.0

    correlations = []
    for i in range(n):
        for j in range(i + 1, n):
            try:
                r, _ = pearsonr(heatmaps[i].ravel(), heatmaps[j].ravel())
                if not np.isnan(r):
                    correlations.append(float(r))
            except Exception:
                pass

    return float(np.mean(correlations)) if correlations else np.nan


# ─── Group C: RobustCAM augmentation stability ────────────────────────────────
# Already implemented in robust_cam.global_stability_metrics().
# Returns {"mean_variance", "mean_iou_topk", "mean_spearman"}.
# Only valid for Robust-CAM (requires multiple augmented heatmaps).
# Returns np.nan for single-view methods (Grad-CAM baseline, LIME, SHAP).


# ─── Unified wrapper ──────────────────────────────────────────────────────────

def compute_all_metrics(
    model_service,
    input_tensor: torch.Tensor,
    heatmap: np.ndarray,
    pred_class: int,
    heatmap_fn=None,
    aug_heatmaps: list | None = None,
    fused_heatmap: np.ndarray | None = None,
    gt_mask: np.ndarray | None = None,
    threshold_percentile: float = 80.0,
    noise_std: float = 0.05,
    n_stability_trials: int = 5,
    n_consistency_runs: int = 3,
) -> dict:
    """
    Runs all 9 metrics and returns a flat dict with keys:

    Group A (Panboonyuen):
        "faith"           — perturbation faithfulness
        "loc_acc"         — localization accuracy (NaN if gt_mask=None)
        "consist_iou"     — explanation consistency (IoU-based)

    Group B (Akgündoğdu):
        "fidelity"        — XAI fidelity (Fid+ - Fid-)
        "stability"       — XAI stability (NaN if heatmap_fn=None)
        "consist_pearson" — XAI consistency (Pearson-based)

    Group C (RobustCAM):
        "mean_variance"   — per-pixel variance (NaN if aug_heatmaps=None)
        "mean_iou_topk"   — top-k IoU vs fused map (NaN if aug_heatmaps=None)
        "mean_spearman"   — Spearman ρ vs fused map (NaN if aug_heatmaps=None)

    Any metric that raises an exception returns np.nan and logs a [Warning].
    Dict is suitable for direct mlflow.log_metrics() and CSV export.
    """
    results: dict[str, float] = {}

    # ── Group A ───────────────────────────────────────────────────────────────

    try:
        results["faith"] = perturbation_faithfulness(
            model_service, input_tensor, heatmap, pred_class, threshold_percentile
        )
    except Exception as e:
        print(f"[Warning] faith failed: {e}")
        results["faith"] = np.nan

    try:
        results["loc_acc"] = localization_accuracy(heatmap, gt_mask, threshold_percentile)
    except Exception as e:
        print(f"[Warning] loc_acc failed: {e}")
        results["loc_acc"] = np.nan

    # consist_iou: requires multiple distinct views to be meaningful.
    # Use aug_heatmaps for Robust-CAM; noisy heatmap_fn repeats for others.
    # Return NaN when only a single view is available — IoU(x,x)=1.0 is trivial.
    try:
        if aug_heatmaps is not None and len(aug_heatmaps) > 1:
            consist_maps = aug_heatmaps
            results["consist_iou"] = explanation_consistency(
                consist_maps, heatmap, threshold_percentile
            )
        elif heatmap_fn is not None:
            # Perturb input with noise each run so maps differ; same noise_std as stability.
            consist_maps = []
            for _ in range(n_consistency_runs):
                noise = torch.randn_like(input_tensor) * noise_std
                consist_maps.append(heatmap_fn(input_tensor + noise))
            results["consist_iou"] = explanation_consistency(
                consist_maps, heatmap, threshold_percentile
            )
        else:
            results["consist_iou"] = np.nan
    except Exception as e:
        print(f"[Warning] consist_iou failed: {e}")
        results["consist_iou"] = np.nan

    # ── Group B ───────────────────────────────────────────────────────────────

    try:
        results["fidelity"] = xai_fidelity(
            model_service, input_tensor, heatmap, pred_class, threshold_percentile
        )
    except Exception as e:
        print(f"[Warning] fidelity failed: {e}")
        results["fidelity"] = np.nan

    try:
        results["stability"] = xai_stability(
            model_service, input_tensor, heatmap_fn, noise_std, n_stability_trials
        )
    except Exception as e:
        print(f"[Warning] stability failed: {e}")
        results["stability"] = np.nan

    # consist_pearson: same source logic as consist_iou — NaN with only one view.
    try:
        if aug_heatmaps is not None and len(aug_heatmaps) > 1:
            pearson_maps = aug_heatmaps
            results["consist_pearson"] = xai_consistency_pearson(pearson_maps)
        elif heatmap_fn is not None:
            pearson_maps = []
            for _ in range(n_consistency_runs):
                noise = torch.randn_like(input_tensor) * noise_std
                pearson_maps.append(heatmap_fn(input_tensor + noise))
            results["consist_pearson"] = xai_consistency_pearson(pearson_maps)
        else:
            results["consist_pearson"] = np.nan
    except Exception as e:
        print(f"[Warning] consist_pearson failed: {e}")
        results["consist_pearson"] = np.nan

    # ── Group C ───────────────────────────────────────────────────────────────

    if aug_heatmaps is not None and len(aug_heatmaps) > 0 and fused_heatmap is not None:
        try:
            gc = global_stability_metrics(aug_heatmaps, fused_heatmap)
            results["mean_variance"] = float(gc["mean_variance"])
            results["mean_iou_topk"] = float(gc["mean_iou_topk"])
            results["mean_spearman"] = float(gc["mean_spearman"])
        except Exception as e:
            print(f"[Warning] Group C metrics failed: {e}")
            results["mean_variance"] = np.nan
            results["mean_iou_topk"] = np.nan
            results["mean_spearman"] = np.nan
    else:
        results["mean_variance"] = np.nan
        results["mean_iou_topk"] = np.nan
        results["mean_spearman"] = np.nan

    return results
