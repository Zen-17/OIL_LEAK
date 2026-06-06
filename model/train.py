"""
Training script for EfficientNet-Lite0 dual-head GPR B-scan classifier.

Two-phase strategy (per README §4.2):
  Phase 1  freeze first 3/4 backbone layers, train heads + tail  (--epochs-frozen)
  Phase 2  unfreeze all, full fine-tune at smaller LR             (--epochs-full)

Usage:
  conda activate torch_gpu
  cd D:\\GPRMax
  python model/train.py
  python model/train.py --data gprmax_sealed_bag_V8/labels.csv --batch 32
  python model/train.py --epochs-frozen 10 --epochs-full 20
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

# Allow  `python model/train.py`  from project root
sys.path.insert(0, str(Path(__file__).parent))
from efficientnet_dual import EfficientNetDual, dual_loss, CLASS_NAMES

ROOT      = Path(__file__).parent.parent
MODEL_DIR = Path(__file__).parent
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# clutter field → Head-B class index  (must match CLASS_NAMES order)
CLUTTER_TO_TYPE: dict[str, int] = {
    "":           0,   # oil  (leak scenario)
    "bare":       4,   # background
    "rock_sm":    3,   # rock
    "rock_md":    3,
    "pipe":       2,   # non-metal PVC pipe
    "metal_sm":   1,   # metal pipe (small)
    "metal_md":   1,   # metal pipe (medium)
}


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_rows(csv_path: Path) -> list[tuple[str, int, int]]:
    """Return list of (abs_image_path, label_a, label_b)."""
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            img_path = ROOT / r["filename"]
            if not img_path.exists():
                continue
            label_a = int(r["label"])
            label_b = CLUTTER_TO_TYPE.get(r.get("clutter", ""), 4)
            rows.append((str(img_path), label_a, label_b))
    return rows


class GaussianNoise:
    def __init__(self, std: float = 0.02):
        self.std = std

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return torch.clamp(t + torch.randn_like(t) * self.std, 0.0, 1.0)


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_transform(augment: bool) -> transforms.Compose:
    if augment:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
            transforms.ToTensor(),
            GaussianNoise(std=0.02),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


class BscanDataset(Dataset):
    def __init__(self, rows: list, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        path, la, lb = self.rows[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(la, dtype=torch.float32), torch.tensor(lb, dtype=torch.long)


def split_dataset(rows: list, val_ratio: float = 0.15, seed: int = 42):
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * val_ratio))
    return rows[n_val:], rows[:n_val]


def make_sampler(rows: list) -> WeightedRandomSampler:
    """Oversample minority class so each epoch sees balanced label_a counts."""
    labels  = [r[1] for r in rows]
    counts  = np.bincount(labels, minlength=2).astype(float)
    weights = [1.0 / counts[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ── Training helpers ──────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scaler, weight_b=None):
    model.train()
    tot_loss = tot_correct = n = 0
    for imgs, la, lb in loader:
        imgs, la, lb = imgs.to(DEVICE), la.to(DEVICE), lb.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            prob_a, logits_b = model(imgs)
            loss = dual_loss(prob_a, logits_b, la, lb, weight_b=weight_b)
        # detach before backward so statistics don't hold the computation graph
        batch_correct = ((prob_a.detach() > 0.5).float() == la).sum().item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        tot_loss    += loss.item() * len(imgs)
        tot_correct += batch_correct
        n += len(imgs)
    return tot_loss / n, tot_correct / n


@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    tot_loss = tot_ca = tot_cb = n = 0
    for imgs, la, lb in loader:
        imgs, la, lb = imgs.to(DEVICE), la.to(DEVICE), lb.to(DEVICE)
        prob_a, logits_b = model(imgs)
        loss     = dual_loss(prob_a, logits_b, la, lb)
        tot_loss += loss.item() * len(imgs)
        tot_ca   += ((prob_a > 0.5).float() == la).sum().item()
        tot_cb   += (logits_b.argmax(1) == lb).sum().item()
        n += len(imgs)
    return tot_loss / n, tot_ca / n, tot_cb / n


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "gprmax_sealed_bag_V8" / "labels.csv"),
                        help="path to labels.csv")
    parser.add_argument("--epochs-frozen", type=int, default=10,
                        help="epochs with backbone frozen (Phase 1)")
    parser.add_argument("--epochs-full",   type=int, default=20,
                        help="epochs with full fine-tune (Phase 2)")
    parser.add_argument("--batch",    type=int,   default=32)
    parser.add_argument("--lr",       type=float, default=1e-3,
                        help="Phase-1 learning rate")
    parser.add_argument("--lr-full",  type=float, default=2e-4,
                        help="Phase-2 learning rate")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--workers",  type=int,   default=4)
    parser.add_argument("--no-pretrain", action="store_true",
                        help="skip ImageNet pretrained weights (for offline machines)")
    args = parser.parse_args()

    print(f"Device : {DEVICE}")
    print(f"Classes: {CLASS_NAMES}")

    rows = load_rows(Path(args.data))
    if not rows:
        print(f"ERROR: no images found via {args.data}")
        print("       Run the simulation pipeline first to generate bscan_preview/ images.")
        sys.exit(1)

    print(f"Dataset: {len(rows)} samples  (from {args.data})")
    la_counts = np.bincount([r[1] for r in rows], minlength=2)
    lb_counts = np.bincount([r[2] for r in rows], minlength=5)
    print(f"  Head-A  noleak={la_counts[0]}  leak={la_counts[1]}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  Head-B  [{i}] {name:<16} {lb_counts[i]}")

    train_rows, val_rows = split_dataset(rows, val_ratio=args.val_ratio)
    print(f"\nTrain: {len(train_rows)}   Val: {len(val_rows)}")

    # Head-B class weight: inverse frequency, normalised so mean weight = 1
    lb_train = np.bincount([r[2] for r in train_rows], minlength=5).astype(float)
    lb_train  = np.maximum(lb_train, 1)          # avoid div-by-zero for missing classes
    weight_b  = torch.tensor(
        lb_train.sum() / (5.0 * lb_train), dtype=torch.float32
    ).to(DEVICE)
    print(f"Head-B class weights: { {n: f'{weight_b[i].item():.2f}' for i, n in enumerate(CLASS_NAMES)} }")

    train_ds = BscanDataset(train_rows, build_transform(augment=True))
    val_ds   = BscanDataset(val_rows,   build_transform(augment=False))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch,
        sampler=make_sampler(train_rows),
        num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    model  = EfficientNetDual(pretrained=not args.no_pretrain).to(DEVICE)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    # ── Phase 1: frozen backbone ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 1  frozen backbone  {args.epochs_frozen} epochs  lr={args.lr}")
    model.freeze_backbone(ratio=0.75)
    opt1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=args.epochs_frozen)

    best_acc = 0.0
    for ep in range(1, args.epochs_frozen + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, opt1, scaler, weight_b)
        va_loss, va_acc_a, va_acc_b = eval_epoch(model, val_loader)
        sched1.step()
        flag = ""
        if va_acc_a > best_acc:
            best_acc = va_acc_a
            torch.save(model.state_dict(), MODEL_DIR / "best_phase1.pth")
            flag = "  ✓ saved"
        print(f"  ep{ep:02d}  tr={tr_loss:.4f}/{tr_acc:.3f}  "
              f"val={va_loss:.4f}/A={va_acc_a:.3f}/B={va_acc_b:.3f}{flag}")

    # ── Phase 2: full fine-tune ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 2  full fine-tune   {args.epochs_full} epochs  lr={args.lr_full}")
    # start Phase 2 from the best Phase 1 checkpoint, not the last epoch
    model.load_state_dict(torch.load(MODEL_DIR / "best_phase1.pth", map_location=DEVICE))
    model.unfreeze_all()
    opt2 = torch.optim.AdamW(model.parameters(), lr=args.lr_full, weight_decay=1e-4)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=args.epochs_full)

    best_acc = 0.0
    save_path = MODEL_DIR / "efficientnet_dual.pth"
    for ep in range(1, args.epochs_full + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, opt2, scaler, weight_b)
        va_loss, va_acc_a, va_acc_b = eval_epoch(model, val_loader)
        sched2.step()
        flag = ""
        if va_acc_a > best_acc:
            best_acc = va_acc_a
            torch.save(model.state_dict(), save_path)
            flag = "  ✓ saved"
        print(f"  ep{ep:02d}  tr={tr_loss:.4f}/{tr_acc:.3f}  "
              f"val={va_loss:.4f}/A={va_acc_a:.3f}/B={va_acc_b:.3f}{flag}")

    print(f"\nBest val Head-A accuracy: {best_acc:.3f}")
    print(f"Final model: {save_path}")


if __name__ == "__main__":
    main()
