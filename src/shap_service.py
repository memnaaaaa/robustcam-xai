# src/shap_service.py
# SHAP GradientExplainer wrapper for per-pixel attribution maps.

import random

import numpy as np
import torch
import shap


class SHAPService:
    """
    Wraps shap.GradientExplainer for per-pixel attribution maps
    in float32 [H, W] format matching GradCAMService outputs.

    Pass background_tensor and model_service at construction time so the
    GradientExplainer is built once and reused across explain() calls.
    Alternatively, call explain() with explicit background_tensor to override.
    """

    def __init__(self, model_service=None, background_tensor: torch.Tensor | None = None):
        self._explainer = None
        if model_service is not None and background_tensor is not None:
            device = model_service.device
            model_service.model.eval()
            self._explainer = shap.GradientExplainer(
                model_service.model, background_tensor.to(device)
            )

    def explain(
        self,
        model_service,
        input_tensor: torch.Tensor,
        background_tensor: torch.Tensor | None = None,
        target_class: int = 0,
    ) -> np.ndarray:
        """
        Returns float32 [224, 224] in [0, 1] (absolute SHAP values, normalized).

        Uses the pre-built explainer from __init__ if available.
        Falls back to building one from background_tensor if provided.
        Steps:
        1. GradientExplainer(model_service.model, background_tensor)
        2. shap_values = explainer.shap_values(input_tensor)
           -> list of 3 arrays, each [1, 3, 224, 224], or ndarray [..., num_classes]
        3. shap_values[target_class][0] -> [3, 224, 224]
        4. np.abs(...).sum(axis=0) -> [224, 224]
        5. Normalize to [0, 1].
        """
        try:
            device = model_service.device
            input_tensor = input_tensor.to(device)

            model_service.model.eval()

            if self._explainer is not None:
                explainer = self._explainer
            elif background_tensor is not None:
                explainer = shap.GradientExplainer(
                    model_service.model, background_tensor.to(device)
                )
            else:
                raise ValueError("SHAPService requires either a pre-built explainer "
                                 "(pass model_service+background_tensor to __init__) "
                                 "or background_tensor passed to explain().")
            shap_values = explainer.shap_values(input_tensor)

            # Handle both list form and ndarray form (newer shap versions)
            if isinstance(shap_values, np.ndarray):
                # shape: [1, 3, 224, 224, num_classes] or [num_classes, 1, 3, 224, 224]
                if shap_values.ndim == 5 and shap_values.shape[-1] > shap_values.shape[0]:
                    # [1, 3, 224, 224, num_classes]
                    channel_map = shap_values[0, :, :, :, target_class]  # [3, 224, 224]
                else:
                    # [num_classes, 1, 3, 224, 224]
                    channel_map = shap_values[target_class][0]
            else:
                # list of arrays
                channel_map = shap_values[target_class]
                if isinstance(channel_map, np.ndarray) and channel_map.ndim == 4:
                    channel_map = channel_map[0]  # [3, 224, 224]

            channel_map = np.abs(channel_map).sum(axis=0)  # [224, 224]
            channel_map = channel_map.astype(np.float32)

            min_val = channel_map.min()
            max_val = channel_map.max()
            channel_map = (channel_map - min_val) / (max_val - min_val + 1e-8)

            return channel_map

        except Exception as e:
            print(f"[Warning] SHAPService.explain failed: {e}")
            return np.zeros((224, 224), dtype=np.float32)

    @staticmethod
    def build_background_tensor(
        data_service,
        image_paths: list[str],
        n_background: int = 10,
        seed: int = 42,
    ) -> torch.Tensor:
        """
        Randomly samples n_background paths from image_paths (training images only).
        Preprocesses each via data_service.get_image_tensor().
        Stacks into [n_background, 3, 224, 224].
        Returns torch.Tensor.
        """
        rng = random.Random(seed)
        sampled = rng.sample(image_paths, min(n_background, len(image_paths)))

        tensors = []
        for path in sampled:
            try:
                tensor, _ = data_service.get_image_tensor(path)
                tensors.append(tensor.squeeze(0))  # [3, 224, 224]
            except Exception as e:
                print(f"[Warning] Could not load background image '{path}': {e}")

        if not tensors:
            raise RuntimeError("No background images could be loaded for SHAP.")

        return torch.stack(tensors, dim=0)  # [n_background, 3, 224, 224]
