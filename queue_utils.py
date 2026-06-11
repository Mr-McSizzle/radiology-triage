import pandas as pd

from model_utils import NO_FINDINGS_LABEL
try:
    from clinical_reasoning import calculate_risk
except Exception:
    calculate_risk = None


HIGH_SEVERITY = {
    "Pneumothorax",
    "Mass",
    "Lung Opacity",
    "Lung Lesion"
}

MEDIUM_SEVERITY = {
    "Effusion",
    "Infiltration",
    "Consolidation",
    "Atelectasis"
}

LOW_SEVERITY = {
    "Fibrosis",
    "Pleural Thickening",
    "Emphysema"
}


def get_urgency(detected_findings, top_score=0.0, age=None, gender=None, view_position=None):
    """Return an urgency label. Prefer rule-based clinical reasoning when available.

    Returns the emoji-prefixed urgency string (e.g., '🔴 HIGH').
    """
    if not detected_findings or detected_findings == [NO_FINDINGS_LABEL]:
        return "🟢 LOW"

    # Prefer the clinical reasoning engine when available
    if calculate_risk is not None:
        try:
            risk_score, urgency = calculate_risk(
                detected_findings,
                [],
                age=age,
                gender=gender,
                view_position=view_position,
            )
            if urgency == "HIGH":
                return "🔴 HIGH"
            if urgency == "MEDIUM":
                return "🟡 MEDIUM"
            return "🟢 LOW"
        except Exception:
            # fallback to legacy mapping below
            pass

    detected_set = set(detected_findings)

    if detected_set & HIGH_SEVERITY:
        return "🔴 HIGH"

    if detected_set & MEDIUM_SEVERITY:
        return "🟡 MEDIUM"

    if detected_set & LOW_SEVERITY:
        return "🟢 LOW"

    # Fall back to confidence if no explicit severity label set matches
    if top_score >= 0.55:
        return "🟡 MEDIUM"

    return "🟢 LOW"


def create_queue_dataframe(queue_data):

    df = pd.DataFrame(queue_data)

    priority_order = {
        "🔴 HIGH": 0,
        "🟡 MEDIUM": 1,
        "🟢 LOW": 2
    }

    df["priority"] = df["Urgency"].map(priority_order)

    df = df.sort_values("priority")

    return df.drop(columns=["priority"])