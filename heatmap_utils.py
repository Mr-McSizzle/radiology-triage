import torch
import torchxrayvision as xrv
import numpy as np
import cv2

from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

model = xrv.models.DenseNet(
    weights="densenet121-res224-all"
)

model.eval()


def generate_heatmap(uploaded_file):

    img = Image.open(uploaded_file).convert("L")
    img = np.array(img)

    original = img.copy()

    img = xrv.datasets.normalize(img, 255)

    img = img[None, :, :]

    transform = xrv.datasets.XRayResizer(224)

    img = transform(img)

    input_tensor = torch.from_numpy(img).unsqueeze(0)

    with torch.no_grad():
        preds = model(input_tensor)

    preds = preds[0].numpy()

    top_idx = np.argmax(preds)

    target_layers = [
        model.features.denseblock4
    ]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    targets = [
        ClassifierOutputTarget(top_idx)
    ]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    rgb_img = cv2.resize(
        original,
        (224,224)
    )

    rgb_img = rgb_img.astype(
        np.float32
    ) / 255.0

    rgb_img = np.stack(
        [rgb_img]*3,
        axis=-1
    )

    heatmap = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    return heatmap