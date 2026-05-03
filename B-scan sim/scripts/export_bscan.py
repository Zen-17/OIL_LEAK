#!/usr/bin/env python
"""Convert gprMax .out files to grayscale B-scan images."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
from PIL import Image


def _find_dataset(h5: h5py.File, rx_name: str, component: str) -> np.ndarray:
    if "rxs" not in h5:
        raise KeyError("Cannot find 'rxs' group in output file.")

    rxs = h5["rxs"]
    if rx_name not in rxs:
        available = ", ".join(sorted(rxs.keys()))
        raise KeyError(f"Receiver '{rx_name}' not found. Available: {available}")

    rx_group = rxs[rx_name]
    if component not in rx_group:
        available = ", ".join(sorted(rx_group.keys()))
        raise KeyError(
            f"Component '{component}' not found in {rx_name}. Available: {available}"
        )

    data = np.asarray(rx_group[component][:], dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D data for B-scan, got shape {data.shape}")

    # We want shape: (time_samples, traces).
    if data.shape[0] < data.shape[1]:
        data = data.T

    return data


def _dewow(data: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return data

    kernel = np.ones(window, dtype=np.float32) / float(window)
    out = np.empty_like(data)
    for i in range(data.shape[1]):
        trend = np.convolve(data[:, i], kernel, mode="same")
        out[:, i] = data[:, i] - trend
    return out


def _background_remove(data: np.ndarray) -> np.ndarray:
    mean_trace = np.mean(data, axis=1, keepdims=True)
    return data - mean_trace


def _zero_time_correct(
    data: np.ndarray,
    search_ratio: float,
    smooth_window: int,
) -> np.ndarray:
    n_samples, n_traces = data.shape
    search_n = int(round(n_samples * search_ratio))
    search_n = max(8, min(n_samples, search_n))

    picks = np.argmax(np.abs(data[:search_n, :]), axis=0).astype(np.int32)

    if smooth_window > 1:
        if smooth_window % 2 == 0:
            smooth_window += 1
        pad = smooth_window // 2
        padded = np.pad(picks, (pad, pad), mode="edge")
        smooth = np.empty_like(picks)
        for i in range(n_traces):
            smooth[i] = int(np.median(padded[i : i + smooth_window]))
        picks = smooth

    out = np.zeros_like(data)
    for i in range(n_traces):
        p = int(picks[i])
        if p >= n_samples - 1:
            continue
        out[: n_samples - p, i] = data[p:, i]
    return out


def _time_gain(data: np.ndarray, gain_alpha: float) -> np.ndarray:
    if gain_alpha <= 0:
        return data

    n_samples = data.shape[0]
    t = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)
    gain = np.exp(gain_alpha * t)
    return data * gain[:, None]


def _mute_top(data: np.ndarray, mute_top_ratio: float, mute_fade_ratio: float) -> np.ndarray:
    if mute_top_ratio <= 0:
        return data

    n_samples = data.shape[0]
    mute_n = int(round(n_samples * mute_top_ratio))
    if mute_n <= 0:
        return data

    out = data.copy()
    mute_n = min(mute_n, n_samples)
    out[:mute_n, :] = 0.0

    fade_n = int(round(n_samples * max(0.0, mute_fade_ratio)))
    if fade_n > 0 and mute_n < n_samples:
        end = min(n_samples, mute_n + fade_n)
        ramp = np.linspace(0.0, 1.0, end - mute_n, endpoint=False, dtype=np.float32)
        out[mute_n:end, :] *= ramp[:, None]

    return out


def _normalize_to_uint8(data: np.ndarray, clip_percentile: float) -> np.ndarray:
    clip_value = np.percentile(np.abs(data), clip_percentile)
    if clip_value <= 1e-12:
        clip_value = 1e-12

    data = np.clip(data / clip_value, -1.0, 1.0)
    img = ((data + 1.0) * 127.5).astype(np.uint8)
    return img


def export_bscan(
    in_out_file: Path,
    out_png: Path,
    rx_name: str,
    component: str,
    dewow_window: int,
    zero_time_correct: bool,
    zero_time_search_ratio: float,
    zero_time_smooth: int,
    remove_background: bool,
    mute_top_ratio: float,
    mute_fade_ratio: float,
    gain_alpha: float,
    clip_percentile: float,
    resize: Tuple[int, int] | None,
    out_npy: Path | None,
) -> None:
    with h5py.File(in_out_file, "r") as h5:
        data = _find_dataset(h5, rx_name=rx_name, component=component)

    if dewow_window > 1:
        data = _dewow(data, window=dewow_window)

    if zero_time_correct:
        data = _zero_time_correct(
            data,
            search_ratio=zero_time_search_ratio,
            smooth_window=zero_time_smooth,
        )

    if remove_background:
        data = _background_remove(data)

    data = _mute_top(
        data,
        mute_top_ratio=mute_top_ratio,
        mute_fade_ratio=mute_fade_ratio,
    )

    data = _time_gain(data, gain_alpha=gain_alpha)

    if out_npy is not None:
        out_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_npy, data)

    image_u8 = _normalize_to_uint8(data, clip_percentile=clip_percentile)

    img = Image.fromarray(image_u8, mode="L")
    if resize is not None:
        img = img.resize(resize, Image.Resampling.LANCZOS)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert gprMax .out file into grayscale B-scan image"
    )
    parser.add_argument("--input", required=True, type=Path, help="Input .out file")
    parser.add_argument("--output", required=True, type=Path, help="Output grayscale PNG")
    parser.add_argument("--rx", default="rx1", help="Receiver name in HDF5 (default: rx1)")
    parser.add_argument(
        "--component",
        default="Ez",
        help="Field component in HDF5, e.g. Ex/Ey/Ez/Hx/Hy/Hz (default: Ez)",
    )
    parser.add_argument(
        "--dewow-window",
        type=int,
        default=9,
        help="Moving-average window for dewow (default: 9; set <=1 to disable)",
    )
    parser.add_argument(
        "--zero-time-correct",
        action="store_true",
        help="Apply per-trace zero-time correction using direct-wave pick",
    )
    parser.add_argument(
        "--zero-time-search-ratio",
        type=float,
        default=0.15,
        help="Search ratio in [0,1] for direct-wave pick (default: 0.15)",
    )
    parser.add_argument(
        "--zero-time-smooth",
        type=int,
        default=9,
        help="Median smoothing window for pick sequence across traces (default: 9)",
    )
    parser.add_argument(
        "--no-background-remove",
        action="store_true",
        help="Disable per-time-sample mean subtraction",
    )
    parser.add_argument(
        "--mute-top-ratio",
        type=float,
        default=0.0,
        help="Mute earliest time samples by ratio [0,1) to suppress ground/direct wave (default: 0)",
    )
    parser.add_argument(
        "--mute-fade-ratio",
        type=float,
        default=0.03,
        help="Fade-in ratio [0,1) after mute to avoid hard boundary (default: 0.03)",
    )
    parser.add_argument(
        "--gain-alpha",
        type=float,
        default=1.8,
        help="Exponential time gain alpha (default: 1.8; set 0 to disable)",
    )
    parser.add_argument(
        "--clip-percentile",
        type=float,
        default=99.5,
        help="Robust clip percentile on absolute amplitude (default: 99.5)",
    )
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        help="Optional output image size, e.g. --resize 256 256",
    )
    parser.add_argument(
        "--save-npy",
        type=Path,
        default=None,
        help="Optional path to save pre-normalized B-scan as .npy",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not (0.0 < args.zero_time_search_ratio <= 1.0):
        raise ValueError("--zero-time-search-ratio must be in (0,1]")
    if args.zero_time_smooth < 1:
        raise ValueError("--zero-time-smooth must be >= 1")

    resize = tuple(args.resize) if args.resize is not None else None
    export_bscan(
        in_out_file=args.input,
        out_png=args.output,
        rx_name=args.rx,
        component=args.component,
        dewow_window=args.dewow_window,
        zero_time_correct=args.zero_time_correct,
        zero_time_search_ratio=args.zero_time_search_ratio,
        zero_time_smooth=args.zero_time_smooth,
        remove_background=not args.no_background_remove,
        mute_top_ratio=args.mute_top_ratio,
        mute_fade_ratio=args.mute_fade_ratio,
        gain_alpha=args.gain_alpha,
        clip_percentile=args.clip_percentile,
        resize=resize,
        out_npy=args.save_npy,
    )


if __name__ == "__main__":
    main()
