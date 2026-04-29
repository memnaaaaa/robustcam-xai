"""
Phase 0 — Fine-tune ResNet50 on IQ-OTH/NCCD lung cancer CT dataset.

Usage:
    python src/train.py --data-root data --epochs 25 --batch-size 32
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

import mlflow
from data_service import IMAGENET_MEAN, IMAGENET_STD
from iq_othncc_dataset import IQOTHNCCDDataset
from mlflow_service import MLflowService

# ── constants ─────────────────────────────────────────────────────────────────

CLASS_NAMES = {0: "Normal", 1: "Benign", 2: "Malignant"}

SUPPORTED_ARCHS = ["resnet50", "resnet101", "densenet161", "efficientnet_b0", "vit_b_16"]

# ── dataset ───────────────────────────────────────────────────────────────────

class IQOTHNCCDTorchDataset(Dataset):
    """
    PyTorch Dataset wrapping the IQ-OTH/NCCD directory structure.
    Applies ImageNet-standard preprocessing (resize 224, normalize).
    Supports train/val/test splits via a pre-shuffled index list.
    """

    def __init__(self, data_root: str, samples: list[tuple[str, int]], augment: bool = False):
        """
        samples: list of (image_path, label) — produced by split_dataset()
        augment: if True, applies random horizontal flip + rotation during training
        """
        self.samples = samples
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        tensor = self.transform(img)
        return tensor, label


def split_dataset(
    data_root: str,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list, list, list]:
    """
    Delegates to IQOTHNCCDDataset for deterministic splitting.
    Returns (train_samples, val_samples, test_samples) as (path, label) tuples.
    """
    ds = IQOTHNCCDDataset(data_root, split="all",
                          train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    train_samples = [(p, l) for p, l, _ in ds.get_split_samples("train")]
    val_samples   = [(p, l) for p, l, _ in ds.get_split_samples("val")]
    test_samples  = [(p, l) for p, l, _ in ds.get_split_samples("test")]
    print(f"Dataset split — train: {len(train_samples)}, val: {len(val_samples)}, test: {len(test_samples)}")
    return train_samples, val_samples, test_samples


# ── model ─────────────────────────────────────────────────────────────────────

def build_model(
    arch: str,
    num_classes: int = 3,
    freeze_backbone: bool = True,
) -> tuple[nn.Module, list, list]:
    """
    Loads ImageNet-pretrained weights for the given architecture, replaces the
    classification head to match num_classes, and optionally freezes backbone params.

    Returns:
        (model, head_params, backbone_params)
        head_params     — parameter list for the new classification head (always trained)
        backbone_params — parameter list for everything else (frozen when freeze_backbone=True)
    """
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(f"arch '{arch}' not supported. Choose from {SUPPORTED_ARCHS}")

    if arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        head_params     = list(model.fc.parameters())
        backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]

    elif arch == "resnet101":
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        head_params     = list(model.fc.parameters())
        backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]

    elif arch == "densenet161":
        model = models.densenet161(weights=models.DenseNet161_Weights.IMAGENET1K_V1)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        head_params     = list(model.classifier.parameters())
        backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]

    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_feat = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_feat, num_classes)
        head_params     = list(model.classifier.parameters())
        backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]

    elif arch == "vit_b_16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        head_params     = list(model.heads.parameters())
        backbone_params = [p for n, p in model.named_parameters() if "heads" not in n]

    if freeze_backbone:
        for p in backbone_params:
            p.requires_grad = False

    return model, head_params, backbone_params


def build_resnet50(num_classes: int = 3, freeze_backbone: bool = True) -> nn.Module:
    """Legacy wrapper — use build_model('resnet50', ...) for new code."""
    model, _, _ = build_model("resnet50", num_classes=num_classes, freeze_backbone=freeze_backbone)
    return model


# ── training utilities ────────────────────────────────────────────────────────

def compute_class_weights(samples: list[tuple[str, int]], num_classes: int = 3) -> torch.Tensor:
    """
    Computes inverse-frequency class weights to handle dataset imbalance.
    Returns a float tensor of shape [num_classes] for nn.CrossEntropyLoss(weight=...).
    """
    counts = [0] * num_classes
    for _, label in samples:
        counts[label] += 1
    total = sum(counts)
    weights = [total / (num_classes * c) if c > 0 else 1.0 for c in counts]
    print(f"Class counts: {counts}, weights: {[f'{w:.3f}' for w in weights]}")
    return torch.tensor(weights, dtype=torch.float32)


def save_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float],
    val_accs: list[float],
    save_path: str = "results/figures/training/training_curves_resnet50.png",
):
    """
    Saves a 2-panel figure (loss | accuracy) across epochs to save_path.
    Called at the end of train() regardless of whether training converged.
    Also logs the figure to MLflow as an artifact.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, train_losses, "b-o", label="Train Loss", markersize=4)
    ax1.plot(epochs, val_losses,   "r-o", label="Val Loss",   markersize=4)
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, "b-o", label="Train Acc", markersize=4)
    ax2.plot(epochs, val_accs,   "r-o", label="Val Acc",   markersize=4)
    ax2.set_title("Training and Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves saved to {save_path}")

    # log to MLflow if a run is active
    try:
        import mlflow
        if mlflow.active_run() is not None:
            mlflow.log_artifact(save_path, artifact_path="figures/training")
    except Exception as e:
        print(f"[Warning] Could not log training curves to MLflow: {e}")


# ── main training loop ────────────────────────────────────────────────────────

def train(
    data_root: str = "data",
    arch: str = "resnet50",
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
    freeze_backbone: bool = True,
    checkpoint_dir: str = "checkpoints",
    results_dir: str = "results",
    seed: int = 42,
    experiment_name: str = None,
):
    """
    Full training loop for any supported architecture on IQ-OTH/NCCD.
    Saves best checkpoint to checkpoints/<arch>_iqothnc.pth.
    Saves training curves to results/figures/training/training_curves_<arch>.png.

    The saved checkpoint dict contains:
        {
            "epoch": int,
            "model_state_dict": ...,
            "val_acc": float,
            "val_loss": float,
            "class_names": {0: "Normal", 1: "Benign", 2: "Malignant"},
            "arch": str,
            "num_classes": 3,
        }
    """
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(f"arch '{arch}' not supported. Choose from {SUPPORTED_ARCHS}")

    if experiment_name is None:
        experiment_name = f"{arch}_IQ_OTH_NCCD_Finetune"

    # reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}  |  arch: {arch}")

    # directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    curves_dir = os.path.join(results_dir, "figures", "training")
    os.makedirs(curves_dir, exist_ok=True)

    # data
    train_samples, val_samples, _ = split_dataset(data_root, seed=seed)

    train_dataset = IQOTHNCCDTorchDataset(data_root, train_samples, augment=True)
    val_dataset   = IQOTHNCCDTorchDataset(data_root, val_samples,   augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)

    # model — build_model returns (model, head_params, backbone_params)
    model, head_params, backbone_params = build_model(
        arch, num_classes=3, freeze_backbone=freeze_backbone
    )
    model = model.to(device)

    # optimizer — head-only when frozen; differential LR when fully unfrozen
    if freeze_backbone:
        optimizer = torch.optim.Adam(head_params, lr=lr)
    else:
        optimizer = torch.optim.Adam([
            {"params": backbone_params, "lr": lr * 0.1},
            {"params": head_params,     "lr": lr},
        ])

    # weighted loss
    class_weights = compute_class_weights(train_samples).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # MLflow setup — delegates load_dotenv() + URI resolution + local fallback to MLflowService
    try:
        mlf = MLflowService(experiment_name=experiment_name)
        mlf.start_run(
            run_name=f"{arch}_run",
            params={
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "freeze_backbone": freeze_backbone,
                "seed": seed,
                "arch": arch,
                "num_classes": 3,
            },
        )
        use_mlflow = True
    except Exception as e:
        print(f"[Warning] MLflow not available: {e}")
        use_mlflow = False

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    best_val_acc  = 0.0
    best_ckpt_path = os.path.join(checkpoint_dir, f"{arch}_iqothnc.pth")

    try:
        for epoch in range(1, epochs + 1):
            # ── train ──
            model.train()
            running_loss = 0.0
            correct = 0
            total   = 0

            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += inputs.size(0)

            train_loss = running_loss / total
            train_acc  = correct / total

            # ── validate ──
            model.eval()
            val_running_loss = 0.0
            val_correct = 0
            val_total   = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item() * inputs.size(0)
                    preds = outputs.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total   += inputs.size(0)

            val_loss = val_running_loss / val_total
            val_acc  = val_correct / val_total

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            print(
                f"Epoch [{epoch:3d}/{epochs}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            if use_mlflow:
                try:
                    mlflow.log_metrics({
                        "train_loss": train_loss,
                        "train_acc":  train_acc,
                        "val_loss":   val_loss,
                        "val_acc":    val_acc,
                    }, step=epoch)
                except Exception as e:
                    print(f"[Warning] MLflow metrics log failed: {e}")

            # save best checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint = {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc":          val_acc,
                    "val_loss":         val_loss,
                    "class_names":      CLASS_NAMES,
                    "arch":             arch,
                    "num_classes":      3,
                }
                torch.save(checkpoint, best_ckpt_path)
                print(f"  -> Saved best checkpoint (val_acc={val_acc:.4f}) to {best_ckpt_path}")

    finally:
        # always save curves
        curves_path = os.path.join(curves_dir, f"training_curves_{arch}.png")
        save_training_curves(train_losses, val_losses, train_accs, val_accs, curves_path)

        if use_mlflow:
            try:
                mlflow.log_metrics({"best_val_acc": best_val_acc})
                mlf.end_run()
            except Exception as e:
                print(f"[Warning] MLflow end_run failed: {e}")

    print(f"\nTraining complete. Best val_acc={best_val_acc:.4f}")
    print(f"Checkpoint: {best_ckpt_path}")
    return best_ckpt_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Fine-tune a CNN on IQ-OTH/NCCD lung CT dataset")
    p.add_argument("--arch",           default="resnet50", choices=SUPPORTED_ARCHS)
    p.add_argument("--data-root",      default="data")
    p.add_argument("--epochs",         default=25, type=int)
    p.add_argument("--batch-size",     default=32, type=int)
    p.add_argument("--lr",             default=1e-3, type=float)
    p.add_argument("--no-freeze",      action="store_true")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--results-dir",    default="results")
    p.add_argument("--seed",           default=42, type=int)
    args = p.parse_args()

    train(
        arch=args.arch,
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        freeze_backbone=not args.no_freeze,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        seed=args.seed,
    )
