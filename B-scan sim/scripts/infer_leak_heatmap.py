#!/usr/bin/env python
"""Infer x-axis oil-leak probability heatmap from one B-scan image."""

from __future__ import annotations

import argparse
import csv
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small


def _load_checkpoint(path: Path, device: torch.device) -> Dict[str, object]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _build_model(num_classes: int) -> nn.Module:
    model = mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def _build_eval_transform(image_size: int):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )


def _prob_to_rgb(p: float) -> Tuple[int, int, int]:
    p = float(np.clip(p, 0.0, 1.0))
    r = int(round(255.0 * p))
    g = int(round(180.0 * (1.0 - abs(2.0 * p - 1.0))))
    b = int(round(255.0 * (1.0 - p)))
    return r, g, b


def _moving_average(arr: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return arr
    k = max(1, int(k))
    if k % 2 == 0:
        k += 1
    pad = k // 2
    arr_pad = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(arr_pad, kernel, mode="valid")


@torch.no_grad()
def _infer_probs(
    model: nn.Module,
    image: Image.Image,
    image_size: int,
    oil_index: int,
    window_width_px: int,
    stride_px: int,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    w, h = image.size
    tf = _build_eval_transform(image_size=image_size)

    x_starts = list(range(0, max(1, w - window_width_px + 1), stride_px))
    if not x_starts or x_starts[-1] != max(0, w - window_width_px):
        x_starts.append(max(0, w - window_width_px))

    centers: List[float] = []
    crops: List[torch.Tensor] = []
    for x0 in x_starts:
        x1 = min(w, x0 + window_width_px)
        x0 = max(0, x1 - window_width_px)
        crop = image.crop((x0, 0, x1, h))
        crops.append(tf(crop))
        centers.append(float(x0 + x1) / 2.0)

    all_probs: List[float] = []
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if (use_amp and device.type == "cuda")
        else nullcontext()
    )
    model.eval()
    for i in range(0, len(crops), batch_size):
        batch = torch.stack(crops[i : i + batch_size], dim=0).to(device)
        with amp_ctx:
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)[:, oil_index]
        all_probs.extend(probs.detach().cpu().tolist())

    x_centers = np.asarray(centers, dtype=np.float32)
    p_centers = np.asarray(all_probs, dtype=np.float32)
    return x_centers, p_centers


def _save_outputs(
    image: Image.Image,
    x_probs: np.ndarray,
    out_prefix: Path,
    overlay_alpha: float,
    bar_height: int,
) -> Dict[str, object]:
    w, h = image.size
    base = np.asarray(image.convert("RGB"), dtype=np.float32)

    colors = np.asarray([_prob_to_rgb(float(p)) for p in x_probs], dtype=np.float32)
    heat_cols = np.repeat(colors[None, :, :], h, axis=0)
    alpha_col = (overlay_alpha * (0.20 + 0.80 * x_probs)).reshape(1, w, 1)
    overlay = base * (1.0 - alpha_col) + heat_cols * alpha_col
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    heatbar = np.repeat(colors[None, :, :], bar_height, axis=0).astype(np.uint8)

    overlay_img = Image.fromarray(overlay, mode="RGB")
    heatbar_img = Image.fromarray(heatbar, mode="RGB")

    canvas = Image.new("RGB", (w, h + bar_height + 4), color=(0, 0, 0))
    canvas.paste(heatbar_img, (0, 0))
    canvas.paste(overlay_img, (0, bar_height + 4))

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    overlay_path = out_prefix.with_name(out_prefix.name + "_overlay.png")
    heatbar_path = out_prefix.with_name(out_prefix.name + "_heatbar.png")
    canvas_path = out_prefix.with_name(out_prefix.name + "_combined.png")
    csv_path = out_prefix.with_name(out_prefix.name + "_xprob.csv")
    summary_path = out_prefix.with_name(out_prefix.name + "_summary.json")

    overlay_img.save(overlay_path)
    heatbar_img.save(heatbar_path)
    canvas.save(canvas_path)

    with csv_path.open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f)
        writer.writerow(["x_px", "x_norm", "p_oil"])
        for x in range(w):
            writer.writerow([x, x / max(1, (w - 1)), float(x_probs[x])])

    peaks = np.argsort(-x_probs)[: min(5, w)]
    summary = {
        "width": int(w),
        "height": int(h),
        "max_p_oil": float(np.max(x_probs)),
        "mean_p_oil": float(np.mean(x_probs)),
        "top_peaks": [
            {"x_px": int(x), "x_norm": float(x / max(1, (w - 1))), "p_oil": float(x_probs[x])}
            for x in peaks
        ],
        "outputs": {
            "overlay": str(overlay_path),
            "heatbar": str(heatbar_path),
            "combined": str(canvas_path),
            "csv": str(csv_path),
        },
    }
    with summary_path.open("w", encoding="ascii") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    summary["outputs"]["summary"] = str(summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate x-axis oil-leak probability heatmap for one B-scan image"
    )
    parser.add_argument("--image", type=Path, required=True, help="Input B-scan PNG")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Model checkpoint path (e.g., runs/.../checkpoints/best.pt)",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Output prefix path (default: <image_stem>_leakprob in image directory)",
    )
    parser.add_argument(
        "--window-width-px",
        type=int,
        default=0,
        help="Sliding window width in pixels (0 = auto by ratio)",
    )
    parser.add_argument(
        "--window-width-ratio",
        type=float,
        default=0.22,
        help="Window width ratio of image width when --window-width-px=0 (default: 0.22)",
    )
    parser.add_argument("--stride-px", type=int, default=4, help="Sliding stride (default: 4)")
    parser.add_argument("--smooth-k", type=int, default=9, help="Smoothing kernel size (default: 9)")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size (default: 64)")
    parser.add_argument(
        "--image-size",
        type=int,
        default=0,
        help="Model input image size (0 = read from checkpoint args)",
    )
    parser.add_argument(
        "--oil-class-name",
        type=str,
        default="oil",
        help="Class name used as positive label (default: oil)",
    )
    parser.add_argument("--overlay-alpha", type=float, default=0.45, help="Overlay alpha (default: 0.45)")
    parser.add_argument("--bar-height", type=int, default=24, help="Top heatbar height (default: 24)")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device (default: auto)",
    )
    parser.add_argument("--amp", action="store_true", help="Use autocast on CUDA")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"--image does not exist: {args.image}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"--checkpoint does not exist: {args.checkpoint}")
    if args.window_width_ratio <= 0.0 or args.window_width_ratio > 1.0:
        raise ValueError("--window-width-ratio must be in (0,1]")
    if args.stride_px < 1:
        raise ValueError("--stride-px must be >= 1")
    if args.smooth_k < 1:
        raise ValueError("--smooth-k must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if not (0.0 <= args.overlay_alpha <= 1.0):
        raise ValueError("--overlay-alpha must be in [0,1]")
    if args.bar_height < 1:
        raise ValueError("--bar-height must be >= 1")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        device = torch.device(args.device)

    ckpt = _load_checkpoint(args.checkpoint, device=device)
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint format unsupported: expected dict-like checkpoint")

    class_to_idx = ckpt.get("class_to_idx", {"no_oil": 0, "oil": 1})
    if args.oil_class_name in class_to_idx:
        oil_index = int(class_to_idx[args.oil_class_name])
    else:
        oil_index = 1

    model_args = ckpt.get("args", {}) if isinstance(ckpt.get("args", {}), dict) else {}
    image_size = int(args.image_size) if args.image_size > 0 else int(model_args.get("image_size", 224))
    num_classes = len(class_to_idx) if isinstance(class_to_idx, dict) else 2

    model = _build_model(num_classes=num_classes).to(device)
    state = ckpt.get("model_state", None)
    if state is None:
        raise KeyError("Cannot find 'model_state' in checkpoint")
    model.load_state_dict(state)

    image = Image.open(args.image).convert("L")
    w, _ = image.size
    if args.window_width_px > 0:
        window_width_px = int(args.window_width_px)
    else:
        window_width_px = max(8, int(round(w * args.window_width_ratio)))
    window_width_px = min(window_width_px, w)

    x_centers, p_centers = _infer_probs(
        model=model,
        image=image,
        image_size=image_size,
        oil_index=oil_index,
        window_width_px=window_width_px,
        stride_px=args.stride_px,
        batch_size=args.batch_size,
        device=device,
        use_amp=args.amp,
    )

    x_full = np.arange(w, dtype=np.float32)
    p_full = np.interp(x_full, x_centers, p_centers).astype(np.float32)
    p_full = np.clip(_moving_average(p_full, args.smooth_k), 0.0, 1.0)

    if args.output_prefix is None:
        out_prefix = args.image.with_name(args.image.stem + "_leakprob")
    else:
        out_prefix = args.output_prefix

    summary = _save_outputs(
        image=image,
        x_probs=p_full,
        out_prefix=out_prefix,
        overlay_alpha=args.overlay_alpha,
        bar_height=args.bar_height,
    )
    print(f"[info] device={device}, image_size={image_size}, oil_index={oil_index}")
    print(
        f"[info] window_width_px={window_width_px}, stride_px={args.stride_px}, "
        f"smooth_k={args.smooth_k}"
    )
    print(f"[info] max_p_oil={summary['max_p_oil']:.4f}, mean_p_oil={summary['mean_p_oil']:.4f}")
    print("[done] outputs:")
    for key, value in summary["outputs"].items():
        print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()

