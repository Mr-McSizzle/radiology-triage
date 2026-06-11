import torch
import torchxrayvision as xrv
import numpy as np
from PIL import Image

# Configurable thresholds
DETECTION_THRESHOLD = 0.40
WEAK_THRESHOLD = 0.30
TOP_K_PREDICTIONS = 5
NO_FINDINGS_LABEL = "No Significant Findings"

# Load model once
model = xrv.models.DenseNet(
    weights="densenet121-res224-all"
)
model.eval()


def _prepare_image(uploaded_file):
    img = Image.open(uploaded_file).convert("L")
    img = np.array(img)
    img = xrv.datasets.normalize(img, 255)
    img = img[None, :, :]
    transform = xrv.datasets.XRayResizer(224)
    img = transform(img)
    return torch.from_numpy(img).unsqueeze(0)


def _get_top_predictions(uploaded_file, top_k=TOP_K_PREDICTIONS):
    input_tensor = _prepare_image(uploaded_file)
    with torch.no_grad():
        preds = model(input_tensor)

    preds = preds[0].numpy()
    results = sorted(
        zip(model.pathologies, preds),
        key=lambda x: x[1],
        reverse=True
    )
    return results[:top_k]


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

    # Healthy-scan heuristic: multiple weak scores without a confident finding
    if len(weak_findings) >= 2:
        return [], weak_findings, True

    return [], weak_findings, True


def predict_xray(uploaded_file,
                  threshold=DETECTION_THRESHOLD,
                  weak_threshold=WEAK_THRESHOLD,
                  top_k=TOP_K_PREDICTIONS):
    """Run model inference and apply clinical confidence filtering."""
    results = _get_top_predictions(uploaded_file, top_k)
    selected_findings, weak_findings, no_significant = _select_findings(
        results,
        threshold=threshold,
        weak_threshold=weak_threshold
    )

    display_label = NO_FINDINGS_LABEL if no_significant else selected_findings[0][0]
    display_score = selected_findings[0][1] if selected_findings else results[0][1]

    return {
        "predictions": results,
        "selected_findings": [label for label, score in selected_findings],
        "selected_scores": selected_findings,
        "display_label": display_label,
        "display_score": display_score,
        "top_score": results[0][1] if results else 0.0,
        "no_findings": no_significant,
        "threshold": threshold,
        "weak_findings": weak_findings
    }