from typing import Dict, List

LABELS: List[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
    "No Finding",
]

LABEL_SYNONYMS: Dict[str, str] = {
    "Pleural Thickening": "Pleural_Thickening",
    "No Findings": "No Finding",
}

label_to_index: Dict[str, int] = {label: index for index, label in enumerate(LABELS)}
index_to_label: Dict[int, str] = {index: label for label, index in label_to_index.items()}


def normalize_label(raw_label: str) -> str:
    if raw_label is None:
        return "No Finding"

    label = str(raw_label).strip()
    if not label:
        return "No Finding"

    if label in LABEL_SYNONYMS:
        return LABEL_SYNONYMS[label]

    return label.replace(" ", "_")


def normalize_findings(finding_labels: str) -> List[str]:
    if finding_labels is None:
        return ["No Finding"]

    raw_labels = [label.strip() for label in str(finding_labels).split("|") if label.strip()]
    normalized = []
    for raw_label in raw_labels:
        canonical = normalize_label(raw_label)
        if canonical in LABELS:
            normalized.append(canonical)

    if not normalized:
        return ["No Finding"]

    seen = []
    for label in normalized:
        if label not in seen:
            seen.append(label)
    return seen
