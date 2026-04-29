# src/vgg_structure.py
# Lists the layer structure of VGG16 for reference when setting hooks.

# Import necessary libraries
from torchvision import models # for VGG16 model

# Print VGG16 layer structure with indices
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
for idx, layer in enumerate(model.features):
    print(f"{idx}: {layer.__class__.__name__}")
