# src/xai_fusion.py
# Pixel-wise voting mask and colormap for cross-method XAI consensus.
# Implements Akgündoğdu & Çelikbaş 2025, Eq. 7-9.

import numpy as np


def binarize_mask(heatmap: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Binarizes a float32 [H, W] heatmap at the given threshold.
    Pixels >= threshold → 1, others → 0.
    Returns uint8 [H, W] in {0, 1}.
    """
    return (heatmap >= threshold).astype(np.uint8)


def compute_voting_mask(
    gradcam_heatmap: np.ndarray,
    lime_heatmap: np.ndarray,
    shap_heatmap: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Binarizes each heatmap at threshold, then sums pixel-wise.
    Returns int32 [H, W] in {0, 1, 2, 3}. 3 = all three methods agree.
    (Akgündoğdu Eq. 7-9)
    """
    m_gradcam = binarize_mask(gradcam_heatmap, threshold).astype(np.int32)
    m_lime    = binarize_mask(lime_heatmap, threshold).astype(np.int32)
    m_shap    = binarize_mask(shap_heatmap, threshold).astype(np.int32)
    return m_gradcam + m_lime + m_shap


def voting_mask_to_colormap(voting_mask: np.ndarray) -> np.ndarray:
    """
    Maps vote counts to colors:
        3 → red    (255,   0,   0)
        2 → purple (128,   0, 128)
        1 → pink   (255, 182, 193)
        0 → blue   ( 30, 144, 255)
    Returns uint8 [H, W, 3].
    """
    h, w = voting_mask.shape
    colormap = np.zeros((h, w, 3), dtype=np.uint8)

    colormap[voting_mask == 0] = (30,  144, 255)   # blue
    colormap[voting_mask == 1] = (255, 182, 193)   # pink
    colormap[voting_mask == 2] = (128,   0, 128)   # purple
    colormap[voting_mask == 3] = (255,   0,   0)   # red

    return colormap


def compute_high_confidence_mask(voting_mask: np.ndarray, min_votes: int = 2) -> np.ndarray:
    """
    Returns a binary mask where vote_count >= min_votes.
    Returns uint8 [H, W] in {0, 1}.
    """
    return (voting_mask >= min_votes).astype(np.uint8)
