# src/mlflow_service.py
# MLflow logging service with PostgreSQL backend + local fallback + augmentation support

# Import necessary libraries
import os # for file operations
from dotenv import load_dotenv
import mlflow # for MLflow tracking
import cv2 # for image saving
import numpy as np # for numerical operations
from datetime import datetime # for timestamping
from mlflow.exceptions import MlflowException # for exception handling

class MLflowService:
    """
    Handles experiment tracking and artifact logging for Grad-CAM runs.
    - Connects to PostgreSQL backend with automatic local fallback.
    - Logs all stagewise and augmented Grad-CAM visualizations.
    """

    def __init__(
        self,
        experiment_name="GradCAM_Experiments",
        tracking_uri_postgres=None,
        fallback_local_uri="./mlruns"
    ):
        """
        Initialize MLflow tracking. Tries PostgreSQL first; falls back to local store if it fails.
        """
        self.experiment_name = experiment_name
        # Load environment variables from .env (if present)
        load_dotenv()

        # Accept tracking URI via parameter or environment variable.
        # Env var: MLFLOW_TRACKING_URI_POSTGRES
        self.tracking_uri = tracking_uri_postgres or os.getenv("MLFLOW_TRACKING_URI_POSTGRES")
        self.fallback_uri = fallback_local_uri
        self.run = None
        # If a tracking URI is provided, try to use it. Otherwise use local fallback.
        if self.tracking_uri:
            try:
                mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(self.experiment_name)
                self.backend = "PostgreSQL"
                print("✅ Connected to MLflow backend")
            except Exception as e:
                print(f"[Warning] Setting tracking URI failed: {e}")
                print("→ Falling back to local mlruns directory.")
                mlflow.set_tracking_uri(self.fallback_uri)
                mlflow.set_experiment(self.experiment_name + "_local")
                self.backend = "local"
        else:
            # No remote tracking configured; default to local runs directory
            print("ℹ️ No Postgres MLflow URI provided via parameter or MLFLOW_TRACKING_URI_POSTGRES; using local mlruns.")
            mlflow.set_tracking_uri(self.fallback_uri)
            mlflow.set_experiment(self.experiment_name + "_local")
            self.backend = "local"

    # -------------------------------------------------------------------------
    # Run management
    # -------------------------------------------------------------------------

    def start_run(self, run_name=None, params=None):
        """
        Start a new MLflow run and log initial parameters.
        """
        run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            self.run = mlflow.start_run(run_name=run_name)
        except MlflowException:
            mlflow.set_tracking_uri(self.fallback_uri)
            self.run = mlflow.start_run(run_name=run_name)

        if params:
            mlflow.log_params(params)
        print(f"\n🧭 Started MLflow run: {run_name} (backend: {self.backend})")

    def end_run(self):
        """
        End the current MLflow run.
        """
        mlflow.end_run()
        print("🏁 MLflow run ended successfully.")

    # -------------------------------------------------------------------------
    # Basic logging helpers
    # -------------------------------------------------------------------------

    def log_scalar(self, key, value):
        """Log a scalar metric."""
        mlflow.log_metric(key, value)

    def log_metrics_dict(self, metrics: dict, prefix: str = "") -> None:
        """
        Log a flat dict of metrics to MLflow, skipping NaN values.
        Keys are optionally prefixed with <prefix>_.
        """
        import math
        clean = {}
        for k, v in metrics.items():
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if math.isnan(fv):
                continue
            key = f"{prefix}_{k}" if prefix else k
            key = key.replace(" ", "_").replace("(", "").replace(")", "")
            clean[key] = fv
        if clean:
            mlflow.log_metrics(clean)

    def _save_temp_image(self, image: np.ndarray, path: str):
        """
        Internal helper for saving numpy images as temporary PNGs.
        """
        if image.ndim == 3 and image.shape[2] == 4:
            img_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        elif image.ndim == 3 and image.shape[2] == 3:
            img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = image
        cv2.imwrite(path, img_bgr)

    # -------------------------------------------------------------------------
    # Grad-CAM logging
    # -------------------------------------------------------------------------

    def log_stagewise_heatmaps(self, heatmaps: dict):
        """
        Log per-layer Grad-CAM heatmaps as PNG images.
        """
        os.makedirs("temp_heatmaps", exist_ok=True)
        for layer_name, heatmap in heatmaps.items():
            temp_path = os.path.join("temp_heatmaps", f"heatmap_{layer_name}.png")
            self._save_temp_image(heatmap, temp_path)
            mlflow.log_artifact(temp_path, artifact_path="heatmaps")
            os.remove(temp_path)

    def log_augmented_overlays(self, overlays: dict):
        """
        Log Grad-CAM overlays (original image + heatmap).
        """
        os.makedirs("temp_overlays", exist_ok=True)
        for layer_name, overlay in overlays.items():
            temp_path = os.path.join("temp_overlays", f"overlay_{layer_name}.png")
            self._save_temp_image(overlay, temp_path)
            mlflow.log_artifact(temp_path, artifact_path="overlays")
            os.remove(temp_path)

    # -------------------------------------------------------------------------
    # Augmentation-specific logging
    # -------------------------------------------------------------------------

    def log_augmented_results(self, augment_name: str, heatmaps: dict, overlays: dict):
        """
        Log Grad-CAM results from augmented images under a dedicated subfolder.

        :param augment_name: e.g., 'rotation_15', 'flip_horizontal', 'color_jitter'
        :param heatmaps: dict[layer_name] = heatmap np.ndarray
        :param overlays: dict[layer_name] = overlay np.ndarray
        """
        base_dir = f"augmentations/{augment_name}"
        os.makedirs("temp_aug", exist_ok=True)

        for layer_name, heatmap in heatmaps.items():
            temp_path = os.path.join("temp_aug", f"{augment_name}_heatmap_{layer_name}.png")
            self._save_temp_image(heatmap, temp_path)
            mlflow.log_artifact(temp_path, artifact_path=f"{base_dir}/heatmaps")
            os.remove(temp_path)

        for layer_name, overlay in overlays.items():
            temp_path = os.path.join("temp_aug", f"{augment_name}_overlay_{layer_name}.png")
            self._save_temp_image(overlay, temp_path)
            mlflow.log_artifact(temp_path, artifact_path=f"{base_dir}/overlays")
            os.remove(temp_path)

        print(f"📦 Logged augmentation '{augment_name}' results to MLflow.")

    # -------------------------------------------------------------------------
    # Robust Grad-CAM logging
    # -------------------------------------------------------------------------

    def log_fused_results(self, fused_heatmaps: dict, uncertainty_maps: dict, metrics: dict):
        """
        fused_heatmaps: dict[layer] = uint8 heatmap
        uncertainty_maps: dict[layer] = uint8 map
        metrics: dict[layer] = dict of scalars
        """
        self.log_stagewise_heatmaps(fused_heatmaps)
        # save uncertainties
        os.makedirs("temp_uncert", exist_ok=True)
        for layer, um in uncertainty_maps.items():
            temp_path = os.path.join("temp_uncert", f"uncert_{layer}.png")
            self._save_temp_image(um, temp_path)
            mlflow.log_artifact(temp_path, artifact_path=f"uncertainty/{layer}")
            os.remove(temp_path)
        # log metrics
        for layer, m in metrics.items():
            for k, v in m.items():
                mlflow.log_metric(f"{layer}_{k}", float(v))
        print("📦 Logged fused & uncertainty results.")

    def log_voting_mask_artifacts(
        self,
        voting_mask: np.ndarray,
        colormap: np.ndarray,
        image_stem: str,
    ):
        """
        Logs the voting mask (grayscale, scaled 0-85 per vote count) and its RGB
        colormap under MLflow artifact path images/<image_stem>/.
        voting_mask: int32 [H, W] in {0,1,2,3}
        colormap:    uint8  [H, W, 3] RGB
        image_stem:  base filename without extension (used as subfolder)
        """
        os.makedirs("temp_voting", exist_ok=True)
        artifact_path = f"images/{image_stem}"

        gray_path = os.path.join("temp_voting", f"{image_stem}_voting_gray.png")
        gray = (voting_mask.astype(np.float32) / 3.0 * 255).astype(np.uint8)
        cv2.imwrite(gray_path, gray)
        mlflow.log_artifact(gray_path, artifact_path=artifact_path)
        os.remove(gray_path)

        color_path = os.path.join("temp_voting", f"{image_stem}_voting_color.png")
        cv2.imwrite(color_path, cv2.cvtColor(colormap, cv2.COLOR_RGB2BGR))
        mlflow.log_artifact(color_path, artifact_path=artifact_path)
        os.remove(color_path)

        print(f"📦 Logged voting mask artifacts for '{image_stem}'.")