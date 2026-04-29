#src/robust_cam.py
# Robust Grad-CAM analysis across augmented images with inverse warping and fusion.

# Import necessary libraries
import numpy as np # for numerical operations
import cv2 # for image processing
from typing import List, Dict, Tuple # for type hints

def warp_heatmap_back(heatmap: np.ndarray, meta: dict, target_shape: Tuple[int,int]):
    """
    Inverse-warp heatmap produced on augmented image back to original image coords.
    heatmap: numpy float HxW in [0,1] (heatmap size equals augmented image resolution)
    meta: dict returned from augmentation service (type, angle, mode)
    target_shape: (H, W) of original image
    """
    h_t, w_t = target_shape
    # Ensure heatmap is same size as target via resize first
    heat_resized = cv2.resize((heatmap*255).astype(np.uint8), (w_t, h_t))
    heat_resized = heat_resized.astype(np.float32) / 255.0

    ttype = meta.get("type", "none")
    if ttype == "none":
        return heat_resized
    if ttype == "flip" and meta.get("mode") == "horizontal":
        # inverse of horizontal flip is another horizontal flip
        return cv2.flip(heat_resized, 1)
    if ttype == "rotation":
        angle = meta.get("angle", 0)
        # inverse rotation by -angle around image center
        center = (w_t/2.0, h_t/2.0)
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)
        warped = cv2.warpAffine((heat_resized*255).astype(np.uint8), M, (w_t, h_t))
        return warped.astype(np.float32) / 255.0
    # default: no-op
    return heat_resized

def fuse_mean(heatmaps: List[np.ndarray]):
    arr = np.stack(heatmaps, axis=0)
    return np.mean(arr, axis=0)

def fuse_median(heatmaps: List[np.ndarray]):
    arr = np.stack(heatmaps, axis=0)
    return np.median(arr, axis=0)

def fuse_weighted(heatmaps: List[np.ndarray], weights: List[float]) -> np.ndarray:
    """
    Weighted average fusion. weights must sum to 1.0 (enforced by normalization).
    Returns float32 array same shape as each heatmap.
    """
    if len(weights) != len(heatmaps):
        raise ValueError(f"len(weights)={len(weights)} != len(heatmaps)={len(heatmaps)}")
    w = np.array(weights, dtype=np.float32)
    w = w / (w.sum() + 1e-8)
    arr = np.stack(heatmaps, axis=0)  # [N, H, W]
    return np.sum(arr * w[:, None, None], axis=0)

def compute_uncertainty(heatmaps: List[np.ndarray]):
    arr = np.stack(heatmaps, axis=0)
    return np.std(arr, axis=0)  # per-pixel std

def global_stability_metrics(heatmaps: List[np.ndarray], fused: np.ndarray, topk_percent=0.1):
    """
    Compute a few simple global metrics:
    - mean per-pixel variance
    - mean IoU between pairwise thresholded masks (top-k by fused map)
    - mean Spearman rank correlation between flattened heatmaps
    """
    import scipy.stats as stats
    arr = np.stack(heatmaps, axis=0)
    mean_var = float(np.mean(np.var(arr, axis=0)))

    # threshold masks by top-k of fused heatmap
    k = int(topk_percent * fused.size)
    if k <= 0:
        k = 1
    flat_idx = np.argsort(fused.ravel())[::-1][:k]
    mask_fused = np.zeros_like(fused, dtype=np.uint8)
    mask_fused.ravel()[flat_idx] = 1

    # pairwise IoU
    N = arr.shape[0]
    ious = []
    for i in range(N):
        hm = arr[i]
        idx = np.argsort(hm.ravel())[::-1][:k]
        mask_i = np.zeros_like(hm, dtype=np.uint8); mask_i.ravel()[idx] = 1
        inter = (mask_i & mask_fused).sum()
        union = (mask_i | mask_fused).sum()
        iou = inter / (union + 1e-8)
        ious.append(iou)
    mean_iou = float(np.mean(ious))

    # Spearman rank corr averaged across heatmaps vs fused
    fvec = fused.ravel()
    rho_list = []
    for i in range(N):
        hv = arr[i].ravel()
        rho, _ = stats.spearmanr(hv, fvec)
        rho_list.append(rho if not np.isnan(rho) else 0.0)
    mean_rho = float(np.mean(rho_list))

    return {"mean_variance": mean_var, "mean_iou_topk": mean_iou, "mean_spearman": mean_rho}