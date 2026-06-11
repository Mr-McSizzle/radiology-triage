import os

import numpy as np
import torch
import torchxrayvision as xrv
from PIL import Image

from labels import LABELS

DETECTION_THRESHOLD = 0.40
WEAK_THRESHOLD = 0.30
TOP_K_PREDICTIONS = 5
NO_FINDINGS_LABEL = "No Finding"
CHECKPOINT_PATH = os.path.join("models", "best_model.pth")


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.version, "hip", None) is not None:
        return torch.device("cuda")
    return torch.device("cpu")


def _load_model(checkpoint_path: str = CHECKPOINT_PATH) -> torch.nn.Module:
    device = _get_device()
    model = xrv.models.DenseNet(weights=None)
    if hasattr(model, "op_threshs"):
        model.op_threshs = None
    if hasattr(model, "apply_sigmoid"):
        model.apply_sigmoid = False
    if hasattr(model, "classifier"):
        in_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_features, len(LABELS))
    elif hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, len(LABELS))
    else:
        raise RuntimeError("Unable to replace DenseNet output layer")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


_MODEL = None


def _prepare_image(uploaded_file):
    img = Image.open(uploaded_file).convert("L")
    img = np.array(img, dtype=np.float32)
    img = xrv.datasets.normalize(img, 255)
    img = img[None, :, :]
    transform = xrv.datasets.XRayResizer(224)
    img = transform(img)
    return torch.from_numpy(img).unsqueeze(0)


def _select_findings(results, threshold=DETECTION_THRESHOLD, weak_threshold=WEAK_THRESHOLD):
    strong_findings = [
        (label, score)
        for label, score in results
        if score >= threshold
    ]
    weak_findings = [
        (label, score)
        for label, score in results
        if weak_threshold <= score < threshold
    ]

    if strong_findings:
        return strong_findings, weak_findings, False

    if len(weak_findings) >= 2:
        return [], weak_findings, True

    return [], weak_findings, True


def _get_model() -> torch.nn.Module:
    global _MODEL
    if _MODEL is None:
        _MODEL = _load_model()
    return _MODEL


def _get_top_predictions(uploaded_file, top_k=TOP_K_PREDICTIONS):
    model = _get_model()
    device = next(model.parameters()).device
    input_tensor = _prepare_image(uploaded_file).to(device)

    with torch.no_grad():
        preds = model(input_tensor)

    preds = torch.sigmoid(preds)

    preds = preds[0].cpu().numpy()
    results = sorted(
        zip(LABELS, preds),
        key=lambda x: x[1],
        reverse=True,
    )
    return results[:top_k]


def predict_xray(uploaded_file, threshold=DETECTION_THRESHOLD, weak_threshold=WEAK_THRESHOLD, top_k=TOP_K_PREDICTIONS):
    results = _get_top_predictions(uploaded_file, top_k)
    selected_findings, weak_findings, no_findings = _select_findings(
        results,
        threshold=threshold,
        weak_threshold=weak_threshold,
    )

    display_label = NO_FINDINGS_LABEL if no_findings else (selected_findings[0][0] if selected_findings else results[0][0])
    display_score = selected_findings[0][1] if selected_findings else results[0][1]

    return {
        "predictions": results,
        "selected_findings": [label for label, _ in selected_findings],
        "selected_scores": selected_findings,
        "display_label": display_label,
        "display_score": display_score,
        "top_score": results[0][1] if results else 0.0,
        "no_findings": no_findings,
        "threshold": threshold,
        "weak_findings": weak_findings,
    }
