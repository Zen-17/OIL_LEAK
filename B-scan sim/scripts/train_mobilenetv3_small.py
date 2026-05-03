#!/usr/bin/env python
"""Train MobileNetV3-Small on B-scan images (oil vs no_oil)."""

from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _sanitize_args_for_ckpt(args: argparse.Namespace) -> Dict[str, object]:
    safe: Dict[str, object] = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            safe[k] = str(v)
        else:
            safe[k] = v
    return safe


def _load_checkpoint(path: Path, device: torch.device) -> Dict[str, object]:
    # PyTorch 2.6 changed torch.load default to weights_only=True.
    # Our own checkpoints include training metadata, so we explicitly opt out.
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        # Fallback for older torch versions that do not expose weights_only.
        return torch.load(path, map_location=device)


class SubsetWithTransform(Dataset):
    """A simple subset wrapper that allows per-split transforms."""

    def __init__(self, base: ImageFolder, indices: Sequence[int], transform) -> None:
        self.base = base
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        base_idx = self.indices[idx]
        path, target = self.base.samples[base_idx]
        image = self.base.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def stratified_split(
    targets: Sequence[int],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    class_to_indices: Dict[int, List[int]] = {}
    for i, label in enumerate(targets):
        class_to_indices.setdefault(int(label), []).append(i)

    rng = random.Random(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    for indices in class_to_indices.values():
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        if n_test < 1:
            n_test = 1
            if n_val > 1:
                n_val -= 1
            elif n_train > 1:
                n_train -= 1

        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train : n_train + n_val])
        test_indices.extend(indices[n_train + n_val : n_train + n_val + n_test])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)
    return train_indices, val_indices, test_indices


def build_transforms(image_size: int, hflip_prob: float, rotate_deg: float):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=hflip_prob),
            transforms.RandomAffine(
                degrees=(-rotate_deg, rotate_deg),
                translate=(0.06, 0.03),
                scale=(0.95, 1.05),
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    eval_tf = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )
    return train_tf, eval_tf


def build_model(num_classes: int, pretrained: bool) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def binary_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    positive_label: int,
) -> Dict[str, float]:
    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)

    tp = int(((y_true_np == positive_label) & (y_pred_np == positive_label)).sum())
    tn = int(((y_true_np != positive_label) & (y_pred_np != positive_label)).sum())
    fp = int(((y_true_np != positive_label) & (y_pred_np == positive_label)).sum())
    fn = int(((y_true_np == positive_label) & (y_pred_np != positive_label)).sum())

    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    positive_label: int,
) -> Dict[str, float]:
    model.eval()
    total = 0
    total_loss = 0.0
    y_true: List[int] = []
    y_pred: List[int] = []

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp and device.type == "cuda"
        else nullcontext()
    )

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with amp_ctx:
            logits = model(images)
            loss = criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)

        bsz = labels.size(0)
        total += bsz
        total_loss += float(loss.item()) * bsz
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    m = binary_metrics(y_true, y_pred, positive_label=positive_label)
    m["loss"] = total_loss / max(1, total)
    return m


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MobileNetV3-Small on B-scan image folders"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Image root (must contain subfolders like oil/ and no_oil/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "mobilenetv3_small",
        help="Directory to save checkpoints and metrics",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--hflip-prob",
        type=float,
        default=0.5,
        help="Horizontal mirror probability for training augmentation (default: 0.5)",
    )
    parser.add_argument(
        "--rotate-deg",
        type=float,
        default=7.0,
        help="Max absolute rotation degree for RandomAffine (default: 7.0)",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument(
        "--monitor",
        type=str,
        default="val_acc",
        choices=["val_acc", "val_f1", "val_loss"],
        help="Metric used to select best checkpoint and early stopping (default: val_acc)",
    )
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train from scratch (default uses ImageNet pretrained weights)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data_root.exists():
        raise FileNotFoundError(f"--data-root does not exist: {args.data_root}")
    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be in (0,1)")
    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be in (0,1)")
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("--train-ratio + --val-ratio must be < 1")
    if not (0.0 <= args.hflip_prob <= 1.0):
        raise ValueError("--hflip-prob must be in [0,1]")
    if args.rotate_deg < 0.0:
        raise ValueError("--rotate-deg must be >= 0")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}")
    print(f"[info] torch: {torch.__version__}, cuda_available={torch.cuda.is_available()}")

    out_dir = args.output_dir.resolve()
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    base = ImageFolder(root=str(args.data_root))
    class_to_idx = base.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    if len(class_to_idx) != 2:
        raise ValueError(
            f"Expected exactly 2 classes, got {len(class_to_idx)}: {class_to_idx}"
        )

    positive_label = class_to_idx.get("oil", max(class_to_idx.values()))

    train_idx, val_idx, test_idx = stratified_split(
        base.targets,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    def _split_count(indices: Sequence[int]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for i in indices:
            name = idx_to_class[int(base.targets[i])]
            counts[name] = counts.get(name, 0) + 1
        return counts

    train_count = _split_count(train_idx)
    val_count = _split_count(val_idx)
    test_count = _split_count(test_idx)
    print(f"[info] split_counts train={train_count} val={val_count} test={test_count}")

    train_tf, eval_tf = build_transforms(
        image_size=args.image_size,
        hflip_prob=args.hflip_prob,
        rotate_deg=args.rotate_deg,
    )
    train_ds = SubsetWithTransform(base, train_idx, train_tf)
    val_ds = SubsetWithTransform(base, val_idx, eval_tf)
    test_ds = SubsetWithTransform(base, test_idx, eval_tf)

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )

    train_targets = [base.targets[i] for i in train_idx]
    counts = torch.bincount(torch.tensor(train_targets), minlength=2).float()
    class_weights = counts.sum() / (counts.clamp_min(1.0) * 2.0)

    model = build_model(num_classes=2, pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = args.amp and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    args_for_ckpt = _sanitize_args_for_ckpt(args)

    config = {
        **vars(args),
        "device": str(device),
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "num_train": len(train_ds),
        "num_val": len(val_ds),
        "num_test": len(test_ds),
        "class_weights": class_weights.tolist(),
        "split_counts": {
            "train": train_count,
            "val": val_count,
            "test": test_count,
        },
    }
    config["data_root"] = str(args.data_root.resolve())
    config["output_dir"] = str(out_dir)
    with (out_dir / "config.json").open("w", encoding="ascii") as f:
        json.dump(config, f, indent=2, ensure_ascii=True)

    monitor_mode = "min" if args.monitor == "val_loss" else "max"
    best_score = float("inf") if monitor_mode == "min" else -1.0
    best_epoch = -1
    stale = 0

    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if args.amp and device.type == "cuda"
            else nullcontext()
        )

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with amp_ctx:
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = torch.argmax(logits, dim=1)
            batch = labels.size(0)
            running_total += batch
            running_loss += float(loss.item()) * batch
            running_correct += int((preds == labels).sum().item())

        scheduler.step()

        train_loss = running_loss / max(1, running_total)
        train_acc = running_correct / max(1, running_total)
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=args.amp,
            positive_label=positive_label,
        )

        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        last_ckpt = ckpt_dir / "last.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_monitor_score": best_score,
                "monitor": args.monitor,
                "class_to_idx": class_to_idx,
                "args": args_for_ckpt,
            },
            last_ckpt,
        )

        current_score = float(row[args.monitor])
        is_better = (
            current_score < best_score
            if monitor_mode == "min"
            else current_score > best_score
        )

        if is_better:
            best_score = current_score
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_monitor_score": best_score,
                    "monitor": args.monitor,
                    "class_to_idx": class_to_idx,
                    "args": args_for_ckpt,
                },
                ckpt_dir / "best.pt",
            )
        else:
            stale += 1
            if stale >= args.patience:
                print(f"[info] early stopping at epoch {epoch}, best epoch={best_epoch}")
                break

    history_path = out_dir / "history.json"
    with history_path.open("w", encoding="ascii") as f:
        json.dump(history, f, indent=2, ensure_ascii=True)

    best_path = ckpt_dir / "best.pt"
    if not best_path.exists():
        best_path = ckpt_dir / "last.pt"
        if not best_path.exists():
            raise FileNotFoundError("Cannot find best.pt or last.pt in checkpoints")
    best_ckpt = _load_checkpoint(best_path, device=device)
    model.load_state_dict(best_ckpt["model_state"])
    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        use_amp=args.amp,
        positive_label=positive_label,
    )

    print("")
    print("=== Test Metrics (Best Checkpoint) ===")
    print(f"best_epoch: {best_ckpt['epoch']}")
    print(
        f"monitor: {best_ckpt.get('monitor', args.monitor)} "
        f"best_score={best_ckpt.get('best_monitor_score', float('nan')):.4f}"
    )
    print(f"test_loss: {test_metrics['loss']:.4f}")
    print(f"test_acc: {test_metrics['acc']:.4f}")
    print(f"test_precision: {test_metrics['precision']:.4f}")
    print(f"test_recall: {test_metrics['recall']:.4f}")
    print(f"test_f1: {test_metrics['f1']:.4f}")
    print(
        "confusion: "
        f"TP={int(test_metrics['tp'])} FP={int(test_metrics['fp'])} "
        f"TN={int(test_metrics['tn'])} FN={int(test_metrics['fn'])}"
    )

    with (out_dir / "test_metrics.json").open("w", encoding="ascii") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=True)

    print(f"[done] outputs: {out_dir}")


if __name__ == "__main__":
    main()
