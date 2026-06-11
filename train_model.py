import argparse
import csv
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import torchxrayvision as xrv
from dataset import NIHChestDataset, build_label_columns
from labels import LABELS


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.version, "hip", None) is not None:
        return torch.device("cuda")
    return torch.device("cpu")


def get_model(num_labels: int) -> torch.nn.Module:
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    if hasattr(model, "classifier"):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_labels)
    elif hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_labels)
    else:
        raise RuntimeError("Unable to replace DenseNet output layer")

    # Disable TorchXRayVision output post-processing so the model returns raw logits.
    # This avoids the built-in op_norm path that expects the original 18-label output size.
    if hasattr(model, "op_threshs"):
        model.op_threshs = None
    if hasattr(model, "apply_sigmoid"):
        model.apply_sigmoid = False
    return model


def set_parameter_requires_grad(model: torch.nn.Module, freeze: bool) -> None:
    for param in model.parameters():
        param.requires_grad = not freeze


def freeze_backbone(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif hasattr(model, "fc"):
        for param in model.fc.parameters():
            param.requires_grad = True


def unfreeze_final_block(model: torch.nn.Module) -> None:
    if hasattr(model, "features"):
        features = model.features
        if hasattr(features, "denseblock4"):
            for param in features.denseblock4.parameters():
                param.requires_grad = True
        if hasattr(features, "norm5"):
            for param in features.norm5.parameters():
                param.requires_grad = True


def compute_pos_weight(df: pd.DataFrame) -> torch.Tensor:
    label_counts = df[LABELS].sum(axis=0).astype(np.float32)
    total = len(df)
    neg = total - label_counts
    pos_weight = neg / (label_counts + 1e-6)
    pos_weight = pos_weight.replace([np.inf, -np.inf], 1.0)
    pos_weight = pos_weight.fillna(1.0)
    return torch.tensor(pos_weight.values, dtype=torch.float32)


def metrics_from_outputs(outputs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    sigmoid = torch.sigmoid(outputs)
    predictions = (sigmoid >= threshold).float()
    targets = targets.float()

    true_positive = (predictions * targets).sum().item()
    false_positive = (predictions * (1.0 - targets)).sum().item()
    false_negative = ((1.0 - predictions) * targets).sum().item()

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
    return precision, recall, f1


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple:
    model.train()
    epoch_loss = 0.0
    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    num_batches = 0

    first_batch_printed = False
    first_forward_printed = False
    first_loss_printed = False
    first_backward_printed = False

    for images, targets in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        targets = targets.to(device)
        if not first_batch_printed:
            print("First batch loaded")
            first_batch_printed = True

        optimizer.zero_grad()
        outputs = model(images)
        if not first_forward_printed:
            print("First forward pass completed")
            first_forward_printed = True

        loss = criterion(outputs, targets)
        if not first_loss_printed:
            print("First loss computed")
            first_loss_printed = True

        loss.backward()
        if not first_backward_printed:
            print("First backward pass completed")
            first_backward_printed = True

        optimizer.step()

        precision, recall, f1 = metrics_from_outputs(outputs, targets)
        epoch_loss += loss.item()
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1
        num_batches += 1

    if num_batches == 0:
        return 0.0, 0.0, 0.0, 0.0

    return (
        epoch_loss / num_batches,
        precision_sum / num_batches,
        recall_sum / num_batches,
        f1_sum / num_batches,
    )


def evaluate_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple:
    model.eval()
    epoch_loss = 0.0
    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Val", leave=False):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)
            precision, recall, f1 = metrics_from_outputs(outputs, targets)

            epoch_loss += loss.item()
            precision_sum += precision
            recall_sum += recall
            f1_sum += f1
            num_batches += 1

    if num_batches == 0:
        return 0.0, 0.0, 0.0, 0.0

    return (
        epoch_loss / num_batches,
        precision_sum / num_batches,
        recall_sum / num_batches,
        f1_sum / num_batches,
    )


def save_checkpoint(checkpoint_path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, val_loss: float) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
    }
    torch.save(checkpoint, checkpoint_path)


def plot_metrics(history: list, output_dir: str) -> None:
    epochs = [row[0] for row in history]
    train_losses = [row[1] for row in history]
    val_losses = [row[2] for row in history]
    f1_scores = [row[5] for row in history]

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, f1_scores, label="F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "f1_curve.png"), dpi=200)
    plt.close()


def write_training_log(history: list, output_path: str) -> None:
    with open(output_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["epoch", "stage", "train_loss", "val_loss", "precision", "recall", "f1"])
        for row in history:
            writer.writerow(row)


def prepare_dataloader(csv_file: str, image_dir: str, batch_size: int, num_workers: int, device: torch.device, debug: bool = False, max_samples: int = 100) -> DataLoader:
    dataset = NIHChestDataset(csv_file, image_dir)
    if debug:
        max_n = min(max_samples, len(dataset))
        dataset = Subset(dataset, list(range(max_n)))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=("train" in csv_file.lower()) if not debug else True,
        num_workers=num_workers,
        pin_memory=(device.type != "cpu"),
    )


def main(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"Selected device: {device}")

    os.makedirs(args.model_dir, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    pos_weight = compute_pos_weight(train_df).to(device)

    train_loader = prepare_dataloader(args.train_csv, args.image_dir, args.batch_size, args.num_workers, device, debug=getattr(args, 'debug', False), max_samples=getattr(args, 'debug_max_samples', 100))
    val_loader = prepare_dataloader(args.val_csv, args.image_dir, args.batch_size, args.num_workers, device, debug=getattr(args, 'debug', False), max_samples=getattr(args, 'debug_val_samples', 20))

    model = get_model(len(LABELS)).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    history = []
    global_epoch = 0

    if getattr(args, 'debug', False):
        stage_settings = [{"name": "Debug", "epochs": 1, "lr": args.stage1_lr, "unfreeze_backbone": False}]
    else:
        stage_settings = [
            {"name": "Stage 1", "epochs": args.stage1_epochs, "lr": args.stage1_lr, "unfreeze_backbone": False},
            {"name": "Stage 2", "epochs": args.stage2_epochs, "lr": args.stage2_lr, "unfreeze_backbone": True},
        ]

        if args.full_finetune:
            stage_settings.append({"name": "Stage 3", "epochs": args.stage3_epochs, "lr": args.stage3_lr, "unfreeze_backbone": True, "full_finetune": True})

    for stage_index, settings in enumerate(stage_settings, start=1):
        print(f"\n=== {settings['name']} ===")
        if settings.get("full_finetune"):
            for param in model.parameters():
                param.requires_grad = True
        elif settings["unfreeze_backbone"]:
            freeze_backbone(model)
            unfreeze_final_block(model)
        else:
            freeze_backbone(model)

        optimizer = optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=settings["lr"],
            weight_decay=args.weight_decay,
        )

        for epoch in range(1, settings["epochs"] + 1):
            global_epoch += 1
            print(f"Epoch {global_epoch} / stage {stage_index} ({settings['name']})")

            train_loss, train_precision, train_recall, train_f1 = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_precision, val_recall, val_f1 = evaluate_epoch(
                model, val_loader, criterion, device
            )

            history.append([
                global_epoch,
                settings["name"],
                train_loss,
                val_loss,
                val_precision,
                val_recall,
                val_f1,
            ])

            print(
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Prec: {val_precision:.4f} | Rec: {val_recall:.4f} | F1: {val_f1:.4f}"
            )

            latest_path = os.path.join(args.model_dir, "latest_model.pth")
            save_checkpoint(latest_path, model, optimizer, global_epoch, val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = global_epoch
                best_path = os.path.join(args.model_dir, "best_model.pth")
                save_checkpoint(best_path, model, optimizer, global_epoch, val_loss)
                print(f"Saved new best model to {best_path}")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter} / {args.patience}")

            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    write_training_log(history, os.path.join(args.model_dir, "training_log.csv"))
    plot_metrics(history, args.model_dir)

    print(f"Training complete. Best epoch: {best_epoch} with val loss {best_val_loss:.4f}")
    print(f"Saved best model and latest model in {args.model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a fine-tuned DenseNet121 model for NIH ChestXray14.")
    parser.add_argument("--image-dir", default="data/images", help="Path to NIH image folder.")
    parser.add_argument("--train-csv", default="train.csv", help="Training split CSV file.")
    parser.add_argument("--val-csv", default="val.csv", help="Validation split CSV file.")
    parser.add_argument("--model-dir", default="models", help="Directory to save model checkpoints.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--stage1-epochs", type=int, default=3, help="Epochs for frozen backbone stage.")
    parser.add_argument("--stage1-lr", type=float, default=1e-3, help="Learning rate for stage 1.")
    parser.add_argument("--stage2-epochs", type=int, default=7, help="Epochs for final block tuning stage.")
    parser.add_argument("--stage2-lr", type=float, default=1e-4, help="Learning rate for stage 2.")
    parser.add_argument("--full-finetune", action="store_true", help="Enable stage 3 full network fine-tuning.")
    parser.add_argument("--stage3-epochs", type=int, default=3, help="Epochs for full fine-tuning stage.")
    parser.add_argument("--stage3-lr", type=float, default=1e-5, help="Learning rate for stage 3.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW.")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience on validation loss.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode: small dataset, 1 epoch, extra prints.")
    parser.add_argument("--debug-max-samples", type=int, default=100, help="Max training samples when debug is enabled.")
    parser.add_argument("--debug-val-samples", type=int, default=20, help="Max validation samples when debug is enabled.")
    args = parser.parse_args()
    main(args)
