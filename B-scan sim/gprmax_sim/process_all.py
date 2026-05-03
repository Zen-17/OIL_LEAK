#!/usr/bin/env python
"""Post-process baseline and oil-leak gprMax outputs for B-scan and spectrum analysis."""

from __future__ import annotations

import glob
import re
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


N_TRACES_EXPECTED = 21
SCAN_X0_M = 0.05
SCAN_STEP_M = 0.02

FREQ_LOW_HZ = 720e6
FREQ_HIGH_HZ = 2500e6
HF_THRESHOLD_HZ = 1600e6

SCENES = {
    "baseline": {"out_dir": "output_baseline", "bscan_png": "bscan_baseline.png", "color": "tab:blue"},
    "oil_leak": {"out_dir": "output_oil", "bscan_png": "bscan_oil.png", "color": "tab:red"},
}


def _trace_sort_key(path: Path) -> Tuple[int, int, str]:
    stem = path.stem
    m = re.search(r"(\d+)$", stem)
    if m:
        return (0, int(m.group(1)), stem)
    return (1, 10**9, stem)


def _list_trace_files(scene_dir: Path) -> List[Path]:
    files = [Path(p) for p in glob.glob(str(scene_dir / "*.out"))]
    files = [p for p in files if "merged" not in p.stem.lower()]
    files.sort(key=_trace_sort_key)

    if len(files) < N_TRACES_EXPECTED:
        raise RuntimeError(
            f"{scene_dir} has only {len(files)} .out files, expected at least {N_TRACES_EXPECTED}."
        )
    if len(files) > N_TRACES_EXPECTED:
        print(
            f"[warn] {scene_dir} has {len(files)} .out files; using first {N_TRACES_EXPECTED} by index order."
        )
        files = files[:N_TRACES_EXPECTED]
    return files


def _read_trace(out_file: Path, component: str = "Ez") -> Tuple[np.ndarray, float]:
    with h5py.File(out_file, "r") as h5:
        if "rxs" not in h5:
            raise KeyError(f"{out_file} has no 'rxs' group.")

        rxs = h5["rxs"]
        rx_names = sorted(rxs.keys())
        if not rx_names:
            raise KeyError(f"{out_file} has empty 'rxs' group.")
        rx = rxs[rx_names[0]]

        if component in rx:
            ds = rx[component]
        else:
            keys = sorted(rx.keys())
            if not keys:
                raise KeyError(f"{out_file} receiver group has no field components.")
            ds = rx[keys[0]]
            print(f"[warn] {out_file.name}: component '{component}' not found, using '{keys[0]}'.")

        trace = np.asarray(ds[:], dtype=np.float64).reshape(-1)
        dt = h5.attrs.get("dt", None)
        if dt is None:
            raise KeyError(f"{out_file} missing root attr 'dt'.")

    return trace, float(dt)


def _build_bscan(out_files: List[Path], component: str = "Ez") -> Tuple[np.ndarray, float]:
    traces: List[np.ndarray] = []
    dt_ref: float | None = None
    min_len: int | None = None

    for f in out_files:
        tr, dt = _read_trace(f, component=component)
        if dt_ref is None:
            dt_ref = dt
        elif not np.isclose(dt_ref, dt, rtol=1e-6, atol=0.0):
            raise RuntimeError(f"Inconsistent dt in {f}: {dt} vs {dt_ref}")

        traces.append(tr)
        min_len = len(tr) if min_len is None else min(min_len, len(tr))

    assert dt_ref is not None and min_len is not None
    trimmed = [tr[:min_len] for tr in traces]
    bscan = np.column_stack(trimmed)  # (time, traces)
    return bscan, dt_ref


def _background_remove(bscan: np.ndarray) -> np.ndarray:
    return bscan - np.mean(bscan, axis=1, keepdims=True)


def _save_bscan_gray(bscan: np.ndarray, out_png: Path, clip_percentile: float = 99.5) -> None:
    v = np.percentile(np.abs(bscan), clip_percentile)
    if v <= 1e-12:
        v = 1e-12
    norm = np.clip(bscan / v, -1.0, 1.0)
    img_u8 = ((norm + 1.0) * 127.5).astype(np.uint8)
    Image.fromarray(img_u8, mode="L").save(out_png)


def _spectrum_features(
    bscan: np.ndarray,
    dt: float,
    f_low: float = FREQ_LOW_HZ,
    f_high: float = FREQ_HIGH_HZ,
    hf_threshold: float = HF_THRESHOLD_HZ,
) -> Dict[str, np.ndarray]:
    n = bscan.shape[0]
    freqs = np.fft.rfftfreq(n, d=dt)
    spec = np.fft.rfft(bscan, axis=0)
    amp = np.abs(spec)
    power = amp**2

    band = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(band):
        raise RuntimeError("No FFT bins inside [720MHz, 2500MHz].")

    f_band = freqs[band]
    amp_band = amp[band, :]
    power_band = power[band, :]

    denom = np.sum(power_band, axis=0)
    denom = np.where(denom <= 1e-20, 1e-20, denom)

    fc = np.sum(f_band[:, None] * power_band, axis=0) / denom

    hf_mask = f_band > hf_threshold
    if np.any(hf_mask):
        hfr = np.sum(power_band[hf_mask, :], axis=0) / denom
    else:
        hfr = np.zeros_like(fc)

    return {"freqs_band": f_band, "amp_band": amp_band, "fc": fc, "hfr": hfr}


def main() -> None:
    root = Path(__file__).resolve().parent

    scene_data: Dict[str, Dict[str, np.ndarray | float | List[Path]]] = {}
    for scene_name, cfg in SCENES.items():
        out_dir = root / cfg["out_dir"]
        if not out_dir.exists():
            raise FileNotFoundError(f"Missing scene output folder: {out_dir}")

        files = _list_trace_files(out_dir)
        bscan_raw, dt = _build_bscan(files, component="Ez")
        bscan_bg = _background_remove(bscan_raw)
        _save_bscan_gray(bscan_bg, root / cfg["bscan_png"])

        feat = _spectrum_features(bscan_bg, dt=dt)
        scene_data[scene_name] = {
            "files": files,
            "dt": dt,
            "bscan_bg": bscan_bg,
            "fc": feat["fc"],
            "hfr": feat["hfr"],
            "freqs_band": feat["freqs_band"],
            "amp_band": feat["amp_band"],
        }

    # shared axes
    x_positions = SCAN_X0_M + SCAN_STEP_M * np.arange(N_TRACES_EXPECTED)
    n_samples = scene_data["baseline"]["bscan_bg"].shape[0]  # type: ignore[index]
    dt = float(scene_data["baseline"]["dt"])  # type: ignore[index]
    t_ns = np.arange(n_samples) * dt * 1e9

    # Figure 1: B-scan compare
    fig1, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, scene_name in zip(axes, ["baseline", "oil_leak"]):
        bscan = scene_data[scene_name]["bscan_bg"]  # type: ignore[index]
        ax.imshow(
            bscan,
            cmap="gray",
            aspect="auto",
            extent=[x_positions[0], x_positions[-1], t_ns[-1], t_ns[0]],
        )
        ax.set_title(scene_name)
        ax.set_xlabel("Scan position (m)")
        ax.set_ylabel("Two-way travel time (ns)")
    fig1.savefig(root / "bscan_compare.png", dpi=150)
    plt.close(fig1)

    # Figure 2: spectral centroid compare
    fig2, ax2 = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for scene_name in ["baseline", "oil_leak"]:
        fc_mhz = scene_data[scene_name]["fc"] / 1e6  # type: ignore[index]
        ax2.plot(x_positions, fc_mhz, marker="o", ms=4, lw=1.6, label=scene_name, color=SCENES[scene_name]["color"])
    ax2.set_xlabel("Scan position (m)")
    ax2.set_ylabel("Spectral centroid (MHz)")
    ax2.set_title("Spectral Centroid Comparison")
    ax2.grid(alpha=0.25)
    ax2.legend()
    fig2.savefig(root / "fc_compare.png", dpi=150)
    plt.close(fig2)

    # Figure 3: high-frequency ratio compare
    fig3, ax3 = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for scene_name in ["baseline", "oil_leak"]:
        hfr = scene_data[scene_name]["hfr"]  # type: ignore[index]
        ax3.plot(x_positions, hfr, marker="o", ms=4, lw=1.6, label=scene_name, color=SCENES[scene_name]["color"])
    ax3.set_xlabel("Scan position (m)")
    ax3.set_ylabel("High-frequency ratio")
    ax3.set_title("HFR Comparison")
    ax3.grid(alpha=0.25)
    ax3.legend()
    fig3.savefig(root / "hfr_compare.png", dpi=150)
    plt.close(fig3)

    # Figure 4: center trace normalized spectrum compare
    center_idx = 10  # 11th trace
    fig4, ax4 = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for scene_name in ["baseline", "oil_leak"]:
        freqs_mhz = scene_data[scene_name]["freqs_band"] / 1e6  # type: ignore[index]
        amp_center = scene_data[scene_name]["amp_band"][:, center_idx]  # type: ignore[index]
        amp_norm = amp_center / max(np.max(amp_center), 1e-12)
        ax4.plot(freqs_mhz, amp_norm, lw=1.8, label=scene_name, color=SCENES[scene_name]["color"])
    ax4.set_xlabel("Frequency (MHz)")
    ax4.set_ylabel("Normalized amplitude")
    ax4.set_title("Center Trace Spectrum (11th Trace)")
    ax4.grid(alpha=0.25)
    ax4.legend()
    fig4.savefig(root / "spectrum_compare.png", dpi=150)
    plt.close(fig4)

    # Statistics printout
    fc_base = scene_data["baseline"]["fc"] / 1e6  # type: ignore[index]
    fc_oil = scene_data["oil_leak"]["fc"] / 1e6  # type: ignore[index]
    hfr_base = scene_data["baseline"]["hfr"]  # type: ignore[index]
    hfr_oil = scene_data["oil_leak"]["hfr"]  # type: ignore[index]

    print("场景：baseline")
    print(f"  频谱重心均值：{np.mean(fc_base):.3f} MHz")
    print(f"  频谱重心标准差：{np.std(fc_base):.3f} MHz")
    print(f"  高频能量比均值：{np.mean(hfr_base):.3f}")
    print(f"  高频能量比标准差：{np.std(hfr_base):.3f}")
    print("")
    print("场景：oil_leak")
    print(f"  频谱重心均值：{np.mean(fc_oil):.3f} MHz")
    print(f"  频谱重心标准差：{np.std(fc_oil):.3f} MHz")
    print(f"  高频能量比均值：{np.mean(hfr_oil):.3f}")
    print(f"  高频能量比标准差：{np.std(hfr_oil):.3f}")
    print("")
    print("两场景对比：")
    print(f"  频谱重心差值（baseline - oil）：{(np.mean(fc_base) - np.mean(fc_oil)):.3f} MHz")
    print(f"  高频能量比差值（baseline - oil）：{(np.mean(hfr_base) - np.mean(hfr_oil)):.3f}")


if __name__ == "__main__":
    main()

