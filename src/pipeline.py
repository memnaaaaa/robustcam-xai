# src/pipeline.py
r"""
RobustCAM-ready pipeline with presentation-grade artifact logging.

Saves artifacts into structured folders:
 - original/
 - augmentations/<aug_name>/
 - robust/
 - uncertainty/

Also creates robust_consistency_grid.png (3 x N: augmented images | aligned CAMs | robust overlays).
"""

# Import necessary libraries
import argparse # for argument parsing
import os # for file operations
import pprint # for pretty-printing
import tempfile # for temporary directories
import shutil # for file operations
from typing import List, Dict, Tuple # for type hints

import numpy as np # for numerical operations
import cv2 # for image processing
from PIL import Image # for image handling

import mlflow # for MLflow logging

# Local services (assumes these files are in src/)
from data_service import DataService
from model_service import ModelService
from gradcam_service import GradCAMService
from augmentation_service import AugmentationService
from mlflow_service import MLflowService

# Robust fusion helpers
from robust_cam import (
    warp_heatmap_back,
    fuse_mean,
    fuse_median,
    compute_uncertainty,
    global_stability_metrics,
)


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def _save_numpy_image_rgb(img_np: np.ndarray, path: str):
    """
    Save an RGB numpy array (H,W,3), values in [0,255] or [0,1].
    """
    arr = img_np.copy()
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    # OpenCV expects BGR
    cv2.imwrite(path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


def _save_gray_map(map_np: np.ndarray, path: str):
    """
    Save a single-channel float map in [0,1] or uint8.
    """
    arr = map_np.copy()
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(path, arr)


def _log_artifact_via_mlflow(local_path: str, artifact_subpath: str, mlf: MLflowService):
    """
    Try to use mlflow_service methods first, else fallback to mlflow.log_artifact.
    We assume mlf.start_run() has been called already.
    """
    # If mlflow_service exposes a general artifact logger, try it (not guaranteed).
    try:
        # Some implementations might expose log_artifact wrapper
        if hasattr(mlf, "log_artifact"):
            mlf.log_artifact(local_path, artifact_subpath)
            return
    except Exception:
        pass

    # fallback to mlflow
    try:
        mlflow.log_artifact(local_path, artifact_path=artifact_subpath)
    except Exception as ex:
        print(f"⚠ Failed to log artifact {local_path} to {artifact_subpath}: {ex}")


def create_consistency_grid(
    aug_images: List[np.ndarray],
    aligned_overlays: List[np.ndarray],
    robust_overlays: List[np.ndarray],
    aug_names: List[str],
) -> np.ndarray:
    """
    Build a 4 x N grid:
    row0: text labels (augmentation names)
    row1: augmented images (resized to original size)
    row2: aligned overlays (H,W,3)
    row3: robust overlays (H,W,3)
    All images resized to same H,W.
    """
    assert len(aug_images) == len(aligned_overlays) == len(robust_overlays)

    if len(aug_images) == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    # Use aligned overlays as reference size
    H, W = aligned_overlays[0].shape[:2]

    def to_uint8_rgb(img):
        a = img.copy()
        if a.dtype != np.uint8:
            a = np.clip(a * 255.0, 0, 255).astype(np.uint8)
        if a.ndim == 2:
            a = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
        return a

    def resize_to_original(img):
        arr = to_uint8_rgb(img)
        return cv2.resize(arr, (W, H), interpolation=cv2.INTER_AREA)

    # Build rows 1–3
    row1 = np.hstack([resize_to_original(img) for img in aug_images])
    row2 = np.hstack([resize_to_original(img) for img in aligned_overlays])
    row3 = np.hstack([resize_to_original(img) for img in robust_overlays])

    # ---- Build Row 0: label bar ----
    # Height of label row: 40px
    label_h = 40
    label_row = np.ones((label_h, row1.shape[1], 3), dtype=np.uint8) * 255  # white bar

    # Put labels centered above each column
    for idx, name in enumerate(aug_names):
        # x-start for this column
        x0 = idx * W
        # Position text roughly centered
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text_size = cv2.getTextSize(name, font, font_scale, thickness)[0]
        text_x = x0 + (W - text_size[0]) // 2
        text_y = (label_h + text_size[1]) // 2  # vertically centered
        cv2.putText(
            label_row,
            name,
            (text_x, text_y),
            font,
            font_scale,
            (0, 0, 0),   # black text
            thickness,
            cv2.LINE_AA,
        )

    # Final grid: labeled + 3 rows of images
    grid = np.vstack([label_row, row1, row2, row3])
    return grid


def run_pipeline(
    image_path: str,
    layers: List[int],
    use_postgres: bool = False,
    do_augmentations: bool = True,
    run_name: str | None = None,
    num_aug: int = 6,
    fusion_method: str = "mean",
):
    # --- instantiate services
    ds = DataService()
    ms = ModelService()
    gradcam = GradCAMService()
    aug = AugmentationService()
    mlf = MLflowService()

    # create a temp folder to stage artifacts
    base_temp_dir = _ensure_dir(os.path.join(tempfile.gettempdir(), "vizcnn_artifacts"))
    run_temp_dir = _ensure_dir(os.path.join(base_temp_dir, next(tempfile._get_candidate_names())))

    # --- load & preprocess image
    print(f"\n[1] Loading image: {image_path}")
    input_tensor, orig_pil = ds.get_image_tensor(image_path, augment=False)
    orig_np = np.array(orig_pil)  # H x W x 3 (RGB)
    H_img, W_img = orig_np.shape[:2]

    # --- register hooks once for desired layers
    print(f"\n[2] Registering hooks on layers: {layers}")
    ms.register_hooks(layers)

    # --- run forward+backward on base image and compute Grad-CAMs
    print("\n[3] Running model and computing stagewise Grad-CAM for base image...")
    class_idx, activations, gradients = ms.run(input_tensor)

    # generate stagewise heatmaps + overlays using existing GradCAMService
    print("\n[4] Generating stagewise outputs (heatmaps + overlays)")
    stage_heatmaps, stage_overlays = gradcam.generate_stagewise_outputs(orig_np, activations, gradients)
    # and raw heatmaps for numeric computations (if available)
    try:
        stage_raw, _ = gradcam.generate_stagewise_raw(orig_np, activations, gradients)
    except Exception:
        stage_raw = {}  # fallback if not implemented

    # --- Start MLflow run and log base results
    run_params = {
        "image_path": os.path.abspath(image_path),
        "layers": ", ".join(map(str, layers)),
        "explained_class_index": int(class_idx),
        "do_augmentations": bool(do_augmentations),
        "num_aug": int(num_aug),
        "fusion_method": fusion_method,
    }
    mlf.start_run(run_name=run_name, params=run_params)

    # root artifact directories in MLflow for this run
    ART_ORIG = "original"
    ART_AUG = "augmentations"
    ART_ROBUST = "robust"
    ART_UNCERT = "uncertainty"

    # --- log original artifacts
    print("\n[5] Logging original image + original GradCAM overlays")
    # save original image
    orig_path = os.path.join(run_temp_dir, "original_image.png")
    Image.fromarray(orig_np).save(orig_path)
    _log_artifact_via_mlflow(orig_path, ART_ORIG, mlf)

    # save stagewise gradcam overlays (visual)
    for layer_name, overlay_img in stage_overlays.items():
        # overlay_img expected to be RGB numpy array
        fn = f"gradcam_{layer_name}.png"
        p = os.path.join(run_temp_dir, fn)
        _save_numpy_image_rgb(overlay_img, p)
        _log_artifact_via_mlflow(p, ART_ORIG, mlf)

    # --- Augmentation loop and collection
    fused_maps: Dict[str, np.ndarray] = {}
    fused_overlays: Dict[str, np.ndarray] = {}
    uncertainty_maps: Dict[str, np.ndarray] = {}
    uncertainty_overlays: Dict[str, np.ndarray] = {}
    stability_metrics: Dict[str, Dict] = {}

    per_layer_aug_heatmaps: Dict[str, List[np.ndarray]] = {}
    # for consistency grid building: store lists per augmentation (we'll choose one layer for grid or do for each)
    aug_images_for_grid: List[np.ndarray] = []
    aligned_overlays_for_grid: List[np.ndarray] = []
    robust_overlays_for_grid: List[np.ndarray] = []

    if do_augmentations:
        print("\n[6] Applying augmentations and computing per-augmentation heatmaps")
        augmented_items = aug.apply(orig_pil)  # returns dict: name -> (PIL.Image, meta)

        # possibly sample a subset if num_aug < total augmentations
        aug_items_list = list(augmented_items.items())
        if num_aug is not None and num_aug > 0 and num_aug < len(aug_items_list):
            aug_items_list = aug_items_list[:num_aug]

        # iterate augmentations
        for aug_name, (aug_img, meta) in aug_items_list:
            print(f"  -> Augmentation: {aug_name}")
            aug_dir = _ensure_dir(os.path.join(run_temp_dir, "augmentations", aug_name))

            # save augmented image
            aug_np = np.array(aug_img)
            aug_img_path = os.path.join(aug_dir, "aug_image.png")
            Image.fromarray(aug_np).save(aug_img_path)
            _log_artifact_via_mlflow(aug_img_path, f"{ART_AUG}/{aug_name}", mlf)

            # preprocess and forward/backward
            aug_tensor = ds.preprocess(aug_img)  # [1,3,H,W]
            class_idx_aug, activations_aug, gradients_aug = ms.run(aug_tensor)

            # raw heatmaps at activation resolution
            raw_heatmaps, aug_visual_overlays_dummy = gradcam.generate_stagewise_raw(np.array(aug_img), activations_aug, gradients_aug)

            # per-layer processing for this augmentation
            per_aug_aligned_overlays: Dict[str, str] = {}  # layer -> local file path for aligned overlay on original
            per_aug_gradcam_on_aug_paths: Dict[str, str] = {}
            per_aug_raw_heatmap_paths: Dict[str, str] = {}
            per_aug_aligned_heatmap_paths: Dict[str, str] = {}
            per_aug_robust_on_aug_paths: Dict[str, str] = {}
            per_aug_uncert_on_aug_paths: Dict[str, str] = {}

            for layer_name, raw_hm in raw_heatmaps.items():
                # Save raw heatmap (resized to readable size if you like) - keep as an 8-bit grayscale for artifact
                raw_hm_resized = cv2.resize((raw_hm * 255).astype(np.uint8), (W_img, H_img))
                raw_path = os.path.join(aug_dir, f"raw_heatmap_{layer_name}.png")
                _save_gray_map(raw_hm_resized, raw_path)
                per_aug_raw_heatmap_paths[layer_name] = raw_path
                _log_artifact_via_mlflow(raw_path, f"{ART_AUG}/{aug_name}", mlf)

                # overlay raw heatmap on augmented image (shows GradCAM on augmented image)
                # resize raw heatmap to aug image size for overlay
                aug_h, aug_w = aug_np.shape[:2]
                raw_for_aug = cv2.resize((raw_hm * 255).astype(np.uint8), (aug_w, aug_h)).astype(np.float32) / 255.0
                gradcam_on_aug = gradcam._overlay(aug_np, raw_for_aug)
                grad_on_aug_path = os.path.join(aug_dir, f"gradcam_on_aug_{layer_name}.png")
                _save_numpy_image_rgb(gradcam_on_aug, grad_on_aug_path)
                per_aug_gradcam_on_aug_paths[layer_name] = grad_on_aug_path
                _log_artifact_via_mlflow(grad_on_aug_path, f"{ART_AUG}/{aug_name}", mlf)

                # inverse-warp into original coordinate frame and ensure float [0,1], original image size
                aligned = warp_heatmap_back(raw_hm, meta, target_shape=(H_img, W_img))
                aligned = np.clip(aligned.astype(np.float32), 0.0, 1.0)
                per_layer_aug_heatmaps.setdefault(layer_name, []).append(aligned)

                # save aligned heatmap (grayscale)
                aligned_path = os.path.join(aug_dir, f"aligned_heatmap_{layer_name}_to_original.png")
                _save_gray_map(aligned, aligned_path)
                per_aug_aligned_heatmap_paths[layer_name] = aligned_path
                _log_artifact_via_mlflow(aligned_path, f"{ART_AUG}/{aug_name}", mlf)

                # overlay aligned heatmap on ORIGINAL image (this is the canonical frame to compare augmentations)
                aligned_overlay = gradcam._overlay(orig_np, aligned)
                aligned_overlay_path = os.path.join(aug_dir, f"aligned_overlay_{layer_name}_on_original.png")
                _save_numpy_image_rgb(aligned_overlay, aligned_overlay_path)
                per_aug_aligned_overlays[layer_name] = aligned_overlay_path
                _log_artifact_via_mlflow(aligned_overlay_path, f"{ART_AUG}/{aug_name}", mlf)

            # store some elements for grid (choose the first layer available to represent consistency; here we pick the first layer)
            # If there are multiple layers, you might want a grid per-layer; here we pick the first layer in the dict
            first_layer = next(iter(raw_heatmaps.keys())) if len(raw_heatmaps) > 0 else None
            if first_layer is not None:
                aug_images_for_grid.append(aug_np)
                # aligned overlay image (load from file)
                aligned_overlay_img = np.array(Image.open(per_aug_aligned_overlays[first_layer]).convert("RGB"))
                aligned_overlays_for_grid.append(aligned_overlay_img)
                # robust overlay placeholder (we'll compute fused maps later; for now append a placeholder or blank)
                # append the original overlay for now; we overwrite with fused overlay later
                robust_overlays_for_grid.append(np.zeros_like(aligned_overlay_img))

            # Optionally log per-augmentation summary via mlflow_service if exists
            try:
                if hasattr(mlf, "log_augmented_results"):
                    # the API in your repo expected (aug_name, heatmaps, overlays) - but here we'll give file paths as dicts
                    # safe fallback: do nothing if signature mismatches
                    mlf.log_augmented_results(aug_name, per_aug_raw_heatmap_paths, per_aug_aligned_overlays)
            except Exception:
                pass

        # --- after processing all augmentations, fuse per-layer
        print("\n[7] Fusing heatmaps per-layer and computing uncertainty + stability")
        for layer_name, hm_list in per_layer_aug_heatmaps.items():
            if len(hm_list) == 0:
                continue

            if fusion_method.lower() in ("median", "med"):
                fused = fuse_median(hm_list)
            else:
                fused = fuse_mean(hm_list)
            fused = np.clip(fused.astype(np.float32), 0.0, 1.0)

            # compute uncertainty map
            uncert = compute_uncertainty(hm_list).astype(np.float32)
            if uncert.max() > 0:
                uncert_norm = uncert / (uncert.max() + 1e-8)
            else:
                uncert_norm = uncert

            # Save fused maps (as grayscale heatmap uint8)
            fused_map_uint8 = (fused * 255.0).astype(np.uint8)
            fused_map_path = os.path.join(run_temp_dir, f"robust_heatmap_{layer_name}.png")
            _save_gray_map(fused_map_uint8, fused_map_path)
            _log_artifact_via_mlflow(fused_map_path, ART_ROBUST, mlf)
            fused_maps[layer_name] = fused_map_uint8

            # Save fused overlay (on original)
            fused_overlay = gradcam._overlay(orig_np, fused)
            fused_overlay_path = os.path.join(run_temp_dir, f"robust_overlay_{layer_name}.png")
            _save_numpy_image_rgb(fused_overlay, fused_overlay_path)
            _log_artifact_via_mlflow(fused_overlay_path, ART_ROBUST, mlf)
            fused_overlays[layer_name] = fused_overlay

            # Save uncertainty heatmap and overlay
            uncert_map_uint8 = (uncert_norm * 255.0).astype(np.uint8)
            uncert_map_path = os.path.join(run_temp_dir, f"uncertainty_heatmap_{layer_name}.png")
            _save_gray_map(uncert_map_uint8, uncert_map_path)
            _log_artifact_via_mlflow(uncert_map_path, ART_UNCERT, mlf)
            uncertainty_maps[layer_name] = uncert_map_uint8

            uncert_overlay = gradcam._overlay(orig_np, uncert_norm)
            uncert_overlay_path = os.path.join(run_temp_dir, f"uncertainty_overlay_{layer_name}.png")
            _save_numpy_image_rgb(uncert_overlay, uncert_overlay_path)
            _log_artifact_via_mlflow(uncert_overlay_path, ART_UNCERT, mlf)
            uncertainty_overlays[layer_name] = uncert_overlay

            # compute stability metrics
            metrics = global_stability_metrics(hm_list, fused, topk_percent=0.1)
            stability_metrics[layer_name] = metrics
            # log scalar metrics
            for k, v in metrics.items():
                try:
                    mlf.log_scalar(f"{layer_name}_{k}", float(v))
                except Exception:
                    try:
                        mlf.log_metric(f"{layer_name}_{k}", float(v))
                    except Exception:
                        pass

            # Now create robust overlays applied to each augmentation (for presentation)
            # For each augmentation, create robust overlay on the augmented image and log it under augmentations/<aug_name> as robustcam_on_aug.png
            # We need to iterate the augmentations again in same order
            for aug_name, (aug_img, meta) in aug_items_list:
                aug_np = np.array(aug_img)
                aug_h, aug_w = aug_np.shape[:2]
                # resize fused (which is in original coords) to augmentation image size
                fused_for_aug = cv2.resize((fused * 255).astype(np.uint8), (aug_w, aug_h)).astype(np.float32) / 255.0
                robust_on_aug = gradcam._overlay(aug_np, fused_for_aug)
                robust_on_aug_path = os.path.join(run_temp_dir, "augmentations", aug_name, f"robustcam_on_aug_{layer_name}.png")
                _save_numpy_image_rgb(robust_on_aug, robust_on_aug_path)
                _log_artifact_via_mlflow(robust_on_aug_path, f"{ART_AUG}/{aug_name}", mlf)

                # Also save robust overlay applied to ORIGINAL image saved under robust/
                robust_on_orig_path = os.path.join(run_temp_dir, f"robust_on_aug_{aug_name}_{layer_name}_on_original.png")
                _save_numpy_image_rgb(fused_overlay, robust_on_orig_path)
                _log_artifact_via_mlflow(robust_on_orig_path, ART_ROBUST, mlf)

                # Save uncertainty overlay per augmentation (normalize uncertainty to aug size)
                uncert_for_aug = cv2.resize((uncert_norm * 255).astype(np.uint8), (aug_w, aug_h)).astype(np.float32) / 255.0
                uncert_on_aug = gradcam._overlay(aug_np, uncert_for_aug)
                uncert_on_aug_path = os.path.join(run_temp_dir, "augmentations", aug_name, f"uncertainty_on_aug_{layer_name}.png")
                _save_numpy_image_rgb(uncert_on_aug, uncert_on_aug_path)
                _log_artifact_via_mlflow(uncert_on_aug_path, f"{ART_AUG}/{aug_name}", mlf)

            # For the consistency grid: fill robust_overlays_for_grid corresponding entries for first_layer
            # update robust_overlays_for_grid items (we set to fused overlay on original)
            if len(robust_overlays_for_grid) == len(aligned_overlays_for_grid) == len(aug_images_for_grid):
                # replace the placeholder for this layer: build fused overlay (on original) and set for all positions
                fused_overlay_for_grid = fused_overlay
                robust_overlays_for_grid = [fused_overlay_for_grid.copy() for _ in robust_overlays_for_grid]

        # create and log consistency grid (pick the first chosen layer)
        if len(aug_images_for_grid) > 0:
            grid_img = create_consistency_grid(
                aug_images_for_grid,
                aligned_overlays_for_grid,
                robust_overlays_for_grid,
                [name for (name, _) in aug_items_list],   # list of augmentation names
                )
            grid_path = os.path.join(run_temp_dir, "robust_consistency_grid.png")
            _save_numpy_image_rgb(grid_img, grid_path)
            _log_artifact_via_mlflow(grid_path, ART_ROBUST, mlf)

    else:
        print("\n[6] Skipping augmentations → RobustCAM fusion disabled.")

    # --- finish & cleanup
    try:
        # log final basic scalars
        try:
            mlf.log_scalar("explained_class_index", int(class_idx))
            mlf.log_scalar("num_stage_maps", len(stage_heatmaps))
        except Exception:
            try:
                mlf.log_metric("explained_class_index", int(class_idx))
                mlf.log_metric("num_stage_maps", len(stage_heatmaps))
            except Exception:
                pass
    except Exception:
        pass

    mlf.end_run()
    print("\n[8] Pipeline finished. Artifacts stored in MLflow under structured folders (original/, augmentations/, robust/, uncertainty/).")

    # Optional: keep temp directory for debugging, otherwise remove it:
    # shutil.rmtree(run_temp_dir)
    print(f"Temp artifacts were placed at: {run_temp_dir} (you may delete it when done).")


def parse_args():
    p = argparse.ArgumentParser(description="Run RobustCAM pipeline and log presentation-quality artifacts to MLflow")
    p.add_argument("--image", "-i", type=str, required=True, help="Path to input image")
    p.add_argument("--layers", "-l", type=int, nargs="+", default=[14, 20, 30], help="Layer indices to hook")
    p.add_argument("--no-augment", dest="augment", action="store_false", help="Disable augmentations")
    p.add_argument("--use-postgres", action="store_true", help="Use PostgreSQL MLflow backend (requires configured mlflow_service)")
    p.add_argument("--run-name", type=str, default=None, help="Optional MLflow run name")
    p.add_argument("--num-aug", type=int, default=6, help="Number of augmentations to sample/use (default 6)")
    p.add_argument("--fusion-method", type=str, default="mean", choices=["mean", "median"], help="Fusion method for RobustCAM")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Pipeline arguments:")
    pprint.pprint(vars(args))

    run_pipeline(
        image_path=args.image,
        layers=args.layers,
        use_postgres=args.use_postgres,
        do_augmentations=args.augment,
        run_name=args.run_name,
        num_aug=args.num_aug,
        fusion_method=args.fusion_method,
    )
