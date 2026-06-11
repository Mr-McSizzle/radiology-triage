import torch
import torchxrayvision as xrv
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# -------------------------
# Load model
# -------------------------
model = xrv.models.DenseNet(weights="densenet121-res224-all")
model.eval()

# -------------------------
# Load image
# -------------------------
img_path = "data/00006585_010.png"

img = Image.open(img_path).convert("L")
img = np.array(img)

# Save original for display
original = img.copy()

# Normalize
img = xrv.datasets.normalize(img, 255)

# Add channel
img = img[None, :, :]

# Resize
transform = xrv.datasets.XRayResizer(224)
img = transform(img)

# Tensor
input_tensor = torch.from_numpy(img).unsqueeze(0)

# -------------------------
# Get predictions
# -------------------------
with torch.no_grad():
    preds = model(input_tensor)

preds = preds[0].numpy()

# Find top prediction
top_idx = np.argmax(preds)

print("Top Finding:", model.pathologies[top_idx])
print("Score:", preds[top_idx])

# -------------------------
# GradCAM
# -------------------------
target_layers = [model.features.denseblock4]

cam = GradCAM(
    model=model,
    target_layers=target_layers
)

targets = [ClassifierOutputTarget(top_idx)]

grayscale_cam = cam(
    input_tensor=input_tensor,
    targets=targets
)

grayscale_cam = grayscale_cam[0]

# -------------------------
# Prepare image
# -------------------------
rgb_img = cv2.resize(original, (224,224))
rgb_img = rgb_img.astype(np.float32) / 255.0

rgb_img = np.stack([rgb_img]*3, axis=-1)

visualization = show_cam_on_image(
    rgb_img,
    grayscale_cam,
    use_rgb=True
)

# -------------------------
# Save
# -------------------------
plt.figure(figsize=(8,8))
plt.imshow(visualization)
plt.axis("off")

plt.savefig("gradcam_output.png")
plt.show()

print("\nSaved: gradcam_output.png")