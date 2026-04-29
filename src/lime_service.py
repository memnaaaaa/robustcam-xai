# src/lime_service.py
# LIME explanation wrapper producing float32 [H, W] saliency maps
# compatible with GradCAMService outputs.

import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from data_service import IMAGENET_MEAN, IMAGENET_STD


class LIMEService:
    """
    Wraps lime.lime_image.LimeImageExplainer to produce normalized saliency
    maps in float32 [H, W] format matching GradCAMService outputs.

    Warning: num_samples=1000 takes ~30-60s per image on CPU.
    Use num_samples=200 for quick tests.
    """

    def __init__(self, num_samples: int = 1000, random_state: int = 42):
        from lime import lime_image
        self.num_samples = num_samples
        self.random_state = random_state
        self.explainer = lime_image.LimeImageExplainer(random_state=random_state)

    def explain(
        self,
        pil_image: Image.Image,
        predict_fn,
        target_class: int,
        image_size: int = 224,
        top_labels: int = 3,
    ) -> np.ndarray:
        """
        Generates a LIME saliency map for target_class.

        Args:
            pil_image: PIL Image to explain.
            predict_fn: callable taking np.ndarray [N, H, W, 3] uint8 and
                returning np.ndarray [N, 3] float32 softmax probabilities.
            target_class: class index (0=Normal, 1=Benign, 2=Malignant).
            image_size: resize image to this size before explanation.
            top_labels: number of top labels to pass to LIME.

        Returns:
            float32 [H, W] heatmap in [0, 1].
        """
        # 1. Prepare uint8 [H, W, 3] numpy array
        img_resized = pil_image.convert("RGB").resize((image_size, image_size))
        img_array = np.array(img_resized, dtype=np.uint8)

        # 2. Run LIME explanation
        explanation = self.explainer.explain_instance(
            img_array,
            predict_fn,
            top_labels=top_labels,
            num_samples=self.num_samples,
            random_seed=self.random_state,
        )

        # 3. Extract segment weights for target class
        # Fall back to top predicted label if target_class not available
        if target_class in explanation.local_exp:
            seg_weights = explanation.local_exp[target_class]
        else:
            fallback = explanation.top_labels[0]
            print(f"[Warning] LIME: target_class={target_class} not in local_exp; "
                  f"falling back to top label {fallback}.")
            seg_weights = explanation.local_exp[fallback]

        # 4. Build [H, W] importance map from segment weights
        segments = explanation.segments  # [H, W] int array
        importance_map = np.zeros(segments.shape, dtype=np.float32)
        for seg_id, weight in seg_weights:
            importance_map[segments == seg_id] = weight

        # 5. Take absolute value and normalize to [0, 1]
        importance_map = np.abs(importance_map)
        min_val = importance_map.min()
        max_val = importance_map.max()
        importance_map = (importance_map - min_val) / (max_val - min_val + 1e-8)

        return importance_map.astype(np.float32)

    def build_predict_fn(self, model_service, data_service):
        """
        Returns a LIME-compatible predict_fn.

        The returned function accepts np.ndarray [N, H, W, 3] uint8 images
        and returns np.ndarray [N, 3] float32 softmax probabilities.

        Uses model_service.model directly with torch.no_grad() — does NOT
        call model_service.run() (avoids contaminating hooks).
        """
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        device = model_service.device
        model = model_service.model

        def predict_fn(images: np.ndarray) -> np.ndarray:
            """
            Args:
                images: np.ndarray [N, H, W, 3] uint8
            Returns:
                np.ndarray [N, 3] float32 softmax probabilities
            """
            tensors = []
            for img in images:
                pil = Image.fromarray(img.astype(np.uint8), mode="RGB")
                t = preprocess(pil)
                tensors.append(t)

            batch = torch.stack(tensors, dim=0).to(device)  # [N, 3, 224, 224]

            with torch.no_grad():
                logits = model(batch)  # [N, num_classes]
                probs = torch.softmax(logits, dim=1)  # [N, 3]

            return probs.cpu().numpy().astype(np.float32)

        return predict_fn
