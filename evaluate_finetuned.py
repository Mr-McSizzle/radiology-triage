import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchxrayvision as xrv
from dataset import NIHChestDataset
from labels import LABELS


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.version, "hip", None) is not None:
        return torch.device("cuda")
    return torch.device("cpu")


def load_finetuned_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = xrv.models.DenseNet(weights=None)
    if hasattr(model, "op_threshs"):
        model.op_threshs = None
    if hasattr(model, "apply_sigmoid"):
        model.apply_sigmoid = False
    if hasattr(model, "classifier"):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, len(LABELS))
    elif hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, len(LABELS))
    else:
        raise RuntimeError("Unable to replace DenseNet output layer")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def compute_metrics(predictions: np.ndarray, targets: np.ndarray, threshold: float = 0.5):
    preds = (predictions >= threshold).astype(int)
    targets = targets.astype(int)

    tp = ((preds == 1) & (targets == 1)).sum(axis=0)
    fp = ((preds == 1) & (targets == 0)).sum(axis=0)
    fn = ((preds == 0) & (targets == 1)).sum(axis=0)
    tn = ((preds == 0) & (targets == 0)).sum(axis=0)

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp, dtype=float), where=(precision + recall) > 0)

    micro_tp = tp.sum()
    micro_fp = fp.sum()
    micro_fn = fn.sum()

    micro_precision = micro_tp / (micro_tp + micro_fp) if micro_tp + micro_fp > 0 else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if micro_tp + micro_fn > 0 else 0.0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)) if micro_precision + micro_recall > 0 else 0.0

    return {
        "per_label": {
            label: {
                "tp": int(tp[idx]),
                "fp": int(fp[idx]),
                "fn": int(fn[idx]),
                "tn": int(tn[idx]),
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
            }
            for idx, label in enumerate(LABELS)
        },
        "micro": {
            "precision": float(micro_precision),
            "recall": float(micro_recall),
            "f1": float(micro_f1),
        },
    }


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            all_predictions.append(probabilities)
            all_targets.append(targets.numpy())

    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)
    return all_predictions, all_targets


def main(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"Selected device: {device}")

    model = load_finetuned_model(args.checkpoint, device)
    test_dataset = NIHChestDataset(args.test_csv, args.image_dir)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    predictions, targets = evaluate(model, test_loader, device)
    metrics = compute_metrics(predictions, targets, threshold=args.threshold)

    report_path = args.output_report
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("FINETUNED MODEL EVALUATION\n")
        report_file.write("=======================\n")
        report_file.write(f"Test samples: {len(test_dataset)}\n")
        report_file.write(f"Threshold: {args.threshold:.2f}\n")
        report_file.write("\nMICRO AVERAGE\n")
        report_file.write(f"Precision: {metrics['micro']['precision']:.4f}\n")
        report_file.write(f"Recall: {metrics['micro']['recall']:.4f}\n")
        report_file.write(f"F1 Score: {metrics['micro']['f1']:.4f}\n")
        report_file.write("\nPER-LABEL METRICS\n")
        for label, stats in metrics["per_label"].items():
            report_file.write(
                f"{label}: TP={stats['tp']} FP={stats['fp']} FN={stats['fn']} TN={stats['tn']} "
                f"Precision={stats['precision']:.4f} Recall={stats['recall']:.4f} F1={stats['f1']:.4f}\n"
            )

    print(f"Saved evaluation report to {report_path}")
    print("MICRO METRICS")
    print(f"Precision: {metrics['micro']['precision']:.4f}")
    print(f"Recall: {metrics['micro']['recall']:.4f}")
    print(f"F1 Score: {metrics['micro']['f1']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned NIH ChestXray14 model.")
    parser.add_argument("--checkpoint", default="models/best_model.pth", help="Path to the best model checkpoint.")
    parser.add_argument("--test-csv", default="test.csv", help="Test split CSV file.")
    parser.add_argument("--image-dir", default="data/images", help="Path to NIH image folder.")
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary classification.")
    parser.add_argument("--output-report", default="finetuned_evaluation_report.txt", help="Path for evaluation summary.")
    args = parser.parse_args()
    main(args)
