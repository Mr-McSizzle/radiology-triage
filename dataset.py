import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchxrayvision as xrv

from labels import LABELS, label_to_index, normalize_findings


def build_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for label in LABELS:
        df[label] = df["Finding Labels"].apply(
            lambda raw: 1 if label in normalize_findings(raw) else 0
        )
    return df


def create_splits(df: pd.DataFrame, seed: int = 42) -> List[pd.DataFrame]:
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_total = len(shuffled)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)

    train = shuffled.iloc[:n_train].reset_index(drop=True)
    val = shuffled.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test = shuffled.iloc[n_train + n_val:].reset_index(drop=True)

    return train, val, test


def prepare_dataset_splits(
    csv_path: str = "data/Data_Entry_2017.csv",
    image_dir: str = "data/images",
    output_dir: str = ".",
    seed: int = 42,
) -> None:
    df = pd.read_csv(csv_path)
    df = df[["Image Index", "Finding Labels"]].copy()
    df["Finding Labels"] = df["Finding Labels"].fillna("No Finding")
    df["Finding Labels"] = df["Finding Labels"].replace({"No Findings": "No Finding"})

    available_images = set(
        f for f in os.listdir(image_dir) if f.lower().endswith(".png")
    )
    df = df[df["Image Index"].isin(available_images)].copy()
    df = df.reset_index(drop=True)

    df = build_label_columns(df)

    train_df, val_df, test_df = create_splits(df, seed=seed)

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Saved split files: {train_path}, {val_path}, {test_path}")


class NIHChestDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        transform: Optional[object] = None,
    ):
        self.image_dir = image_dir
        self.df = pd.read_csv(csv_path)
        self.df["Finding Labels"] = self.df["Finding Labels"].fillna("No Finding")
        if not set(LABELS).issubset(self.df.columns):
            self.df = build_label_columns(self.df)

        self.image_names = self.df["Image Index"].tolist()
        self.labels = self.df[LABELS].astype(np.float32).to_numpy()
        self.transform = transform if transform is not None else xrv.datasets.XRayResizer(224)

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int):
        image_name = self.image_names[index]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("L")
        image = np.array(image, dtype=np.float32)
        image = xrv.datasets.normalize(image, 255)
        image = image[None, :, :]
        image = self.transform(image)
        image_tensor = torch.from_numpy(image).float()
        label_tensor = torch.from_numpy(self.labels[index]).float()
        return image_tensor, label_tensor


if __name__ == "__main__":
    prepare_dataset_splits()
