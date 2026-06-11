from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class PatientRecord:
    patient_id: str
    filename: str
    age: int = None
    gender: str = None
    view_position: str = None
    findings: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    risk_score: int = None
    urgency: str = None
    reason: str = None


# Severity groups for imaging findings
HIGH_SEVERITY = {"pneumothorax", "mass", "lung opacity", "lung lesion", "nodule"}
MEDIUM_SEVERITY = {"effusion", "infiltration", "consolidation", "atelectasis"}
LOW_SEVERITY = {"fibrosis", "pleural thickening", "emphysema"}


def _normalize_text(s: str) -> str:
    return (s or "").strip().lower()


def calculate_risk(
    findings: List[str],
    scores: List[float],
    age: int = None,
    gender: str = None,
    view_position: str = None,
) -> Tuple[int, str, str]:
    """
    Calculate a rule-based risk score and urgency using NIH chest X-ray metadata.

    Inputs:
    - findings: list of finding names (strings)
    - scores: list of confidences parallel to findings
    - age: patient age (int)
    - gender: patient gender (string)
    - view_position: study view position (e.g. PA, AP)

    Outputs:
    - risk_score (int)
    - urgency (LOW/MEDIUM/HIGH)
    - reason (string)
    """
    score = 0
    reasons = []

    # Age rules
    if age is not None:
        try:
            age_val = int(age)
        except Exception:
            age_val = None
        if age_val is not None:
            if age_val > 80:
                score += 3
                reasons.append("Age > 80 (+3)")
            elif age_val > 65:
                score += 2
                reasons.append("Age > 65 (+2)")

    # View position rules
    if view_position is not None:
        position = _normalize_text(view_position)
        if position == "ap":
            score += 1
            reasons.append("AP view position (+1)")
        elif position == "pa":
            reasons.append("PA view position (standard) (+0)")

    # Findings rules
    normalized_findings = [(_normalize_text(f), s) for f, s in zip(findings or [], scores or [])]
    for fn, conf in normalized_findings:
        added = 0
        if any(h in fn for h in HIGH_SEVERITY):
            added = 3
        elif any(m in fn for m in MEDIUM_SEVERITY):
            added = 2
        elif any(l in fn for l in LOW_SEVERITY):
            added = 1
        elif fn:
            try:
                added = 1 if conf >= 0.5 else 0
            except Exception:
                added = 0
        if added:
            score += added
            reasons.append(f"Finding: {fn} (+{added})")

    # Simple post-processing: clamp to non-negative int
    risk_score = int(max(0, round(score)))

    # Map risk to urgency
    if risk_score <= 2:
        urgency = "LOW"
    elif 3 <= risk_score <= 5:
        urgency = "MEDIUM"
    else:
        urgency = "HIGH"

    # Build explanation
    if reasons:
        reason_text = "; ".join(reasons)
    else:
        reason_text = "No high-risk features detected from provided inputs."

    if gender:
        reason_text = f"Gender: {gender}. " + reason_text

    return risk_score, urgency, reason_text
