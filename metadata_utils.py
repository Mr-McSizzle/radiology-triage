import os
from functools import lru_cache
from typing import Optional, Dict

import pandas as pd


CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "Data_Entry_2017.csv")


def _load_metadata_df() -> pd.DataFrame:
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Metadata CSV not found at {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    # Ensure consistent string values for key lookup
    if "Image Index" in df.columns:
        df["Image Index"] = df["Image Index"].astype(str).str.strip()
    return df


@lru_cache(maxsize=1024)
def get_patient_metadata(image_name: str) -> Optional[Dict[str, object]]:
    """Return NIH ChestXray14 metadata for the given image filename."""
    if not image_name:
        return None

    image_name = os.path.basename(str(image_name)).strip()
    df = _load_metadata_df()

    matches = df.loc[df["Image Index"] == image_name]
    if matches.empty:
        return None

    row = matches.iloc[0]
    patient_id = row.get("Patient ID")
    age = row.get("Patient Age")
    gender = row.get("Patient Gender")
    view_position = row.get("View Position")

    try:
        age_val = int(age) if pd.notna(age) else None
    except Exception:
        age_val = None

    return {
        "patient_id": str(patient_id).strip() if pd.notna(patient_id) else None,
        "age": age_val,
        "gender": str(gender).strip() if pd.notna(gender) else None,
        "view_position": str(view_position).strip() if pd.notna(view_position) else None,
    }
