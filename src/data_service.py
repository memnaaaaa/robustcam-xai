# src/data_service.py
# Handles image loading, preprocessing, and augmentations.

import torch
from torchvision import transforms
from PIL import Image
import random

# Canonical ImageNet normalization constants — import from here to avoid duplication.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# DataService class definition
class DataService:
    def __init__(self, image_size: int = 224):
        """
        Handles loading, preprocessing, and augmentations for input images.
        :param image_size: target size for model input (default 224 for VGG16)
        """
        self.image_size = image_size

        # Standard VGG16 normalization (ImageNet mean/std)
        self.preprocess_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        # Basic augmentation pipeline
        self.augmentation_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])

    def load_image(self, image_path: str):
        """
        Loads an image from disk and converts it to RGB.
        :param image_path: path to the image file
        :return: PIL.Image
        """
        image = Image.open(image_path).convert("RGB")
        return image

    def preprocess(self, image: Image.Image):
        """
        Applies preprocessing (resize, normalize) for VGG16.
        :param image: PIL.Image
        :return: torch.Tensor of shape [1, 3, H, W]
        """
        tensor = self.preprocess_transform(image).unsqueeze(0)
        return tensor

    def augment_image(self, image: Image.Image):
        """
        Applies random augmentation to the image.
        :param image: PIL.Image
        :return: Augmented PIL.Image
        """
        augmented = self.augmentation_transform(image)
        return augmented

    def get_image_tensor(self, image_path: str, augment: bool = False):
        """
        Loads, optionally augments, and preprocesses an image.
        :param image_path: path to image file
        :param augment: whether to apply data augmentation
        :return: (preprocessed_tensor, original_or_augmented_PIL_image)
        """
        image = self.load_image(image_path)

        if augment:
            # Random chance to apply augmentation
            if random.random() > 0.5:
                image = self.augment_image(image)

        tensor = self.preprocess(image)
        return tensor, image
