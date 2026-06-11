import torch
import torchxrayvision as xrv
from PIL import Image
import numpy as np
import pandas as pd
# Load model
print("Loading model...")
model = xrv.models.DenseNet(weights="densenet121-res224-all")
model.eval()

# Load image
img_path = "data/00006585_010.png"

img = Image.open(img_path).convert("L")
img = np.array(img)

# Normalize
img = xrv.datasets.normalize(img, 255)

# Add channel dimension
img = img[None, :, :]

# Resize to 224x224
transform = xrv.datasets.XRayResizer(224)
img = transform(img)

# Convert to tensor
img = torch.from_numpy(img).unsqueeze(0)

# Run inference
with torch.no_grad():
    preds = model(img)

preds = preds[0].numpy()
# Pair disease names with scores
results = list(zip(model.pathologies, preds))
results.sort(key=lambda x: x[1], reverse=True)

print("\nTop Findings:\n")

for pathology, score in results[:5]:
    print(f"{pathology}: {score:.3f}")

top_idx = np.argmax(preds)
# Urgency logic
top_score = results[0][1]

if top_score > 0.5:
    urgency = "HIGH"
elif top_score > 0.2:
    urgency = "MEDIUM"
else:
    urgency = "LOW"

print(f"\nUrgency Level: {urgency}")

# Simple report
top_finding = results[0][0]

report = f"""
===========================
PRELIMINARY TRIAGE REPORT
===========================

Top Finding: {top_finding}

Urgency Level: {urgency}

This scan has been automatically flagged by the AI triage system for radiologist review.

Note: This is NOT a diagnosis and should be reviewed by a qualified medical professional.
"""

print(report)