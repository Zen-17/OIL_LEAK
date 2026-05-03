#!/usr/bin/env python
"""Generate gprMax stepped-frequency (720~2500 MHz) oil-leak scene input files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple


def _frange_inclusive(start: float, stop: float, step: float) -> List[float]:
    values: List[float] = []
    x = start
    eps = step * 1e-6
    while x <= stop + eps:
        values.append(round(x, 6))
        x += step
    return values


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _ellipse_slices(
    *,
    x_center: float,
    z_center: float,
    x_radius: float,
    z_radius: float,
    z_min: float,
    z_max: float,
    y_min: float,
    y_max: float,
    dx: float,
    x_lo: float,
    x_hi: float,
    material: str,
) -> List[str]:
    lines: List[str] = []
    x_start = _clip(x_center - x_radius, x_lo, x_hi)
    x_end = _clip(x_center + x_radius, x_lo, x_hi)

    n = max(1, int(math.ceil((x_end - x_start) / dx)))
    for i in range(n):
        x0 = x_start + i * dx
        x1 = min(x_end, x0 + dx)
        x_mid = 0.5 * (x0 + x1)

        term = 1.0 - ((x_mid - x_center) ** 2) / (x_radius**2)
        if term <= 0.0:
            continue

        z_half = z_radius * math.sqrt(term)
        zz0 = max(z_min, z_center - z_half)
        zz1 = min(z_max, z_center + z_half)
        if zz1 - zz0 < dx:
            continue

        lines.append(
            f"#box: {x0:.3f} {y_min:.3f} {zz0:.3f} {x1:.3f} {y_max:.3f} {zz1:.3f} {material}"
        )

    return lines


def build_scene(
    *,
    freq_hz: float,
    traces: int,
    tx_x0: float,
    tx_rx_spacing: float,
    trace_step: float,
    waveform: str,
) -> Tuple[str, Dict[str, float | int | str]]:
    # ---- Requested fixed setup ----
    dx = dy = dz = 0.003
    x_size, y_size, z_size = 0.5, 0.1, 3.5
    time_window = 80e-9
    pml_cells = 10

    # keep scan line just outside top PML; "height=0" is relative to this ground surface
    z_surface = pml_cells * dz
    antenna_height = 0.0
    z_ant = z_surface + antenna_height
    y_ant = 0.5 * y_size

    # material properties
    er_bg, sig_bg = 18.0, 0.02
    er_transition, sig_transition = 8.0, 0.005
    er_core, sig_core = 2.5, 0.0001

    # leak geometry (depth is measured from z_surface)
    x_center = 0.25
    z_center = z_surface + 0.20
    leak_z_min = z_surface + 0.05
    leak_z_max = z_surface + 0.35

    # transition zone ellipse
    tr_xr, tr_zr = 0.21, 0.14
    # core zone ellipse
    core_xr, core_zr = 0.15, 0.08

    # keep anomaly away from y-PML
    y_lo = pml_cells * dy + dy
    y_hi = y_size - pml_cells * dy - dy
    if y_hi <= y_lo:
        raise ValueError("y-domain too small after reserving PML region for anomaly.")

    # scan boundary check
    rx_x0 = tx_x0 + tx_rx_spacing
    tx_last = tx_x0 + (traces - 1) * trace_step
    rx_last = rx_x0 + (traces - 1) * trace_step
    safe_x_min = pml_cells * dx + dx
    safe_x_max = x_size - pml_cells * dx - dx
    if tx_x0 < safe_x_min or tx_last > safe_x_max or rx_x0 < safe_x_min or rx_last > safe_x_max:
        raise ValueError(
            "Tx/Rx scan path exceeds safe domain interior. "
            f"safe_x=[{safe_x_min:.3f},{safe_x_max:.3f}], "
            f"tx_range=[{tx_x0:.3f},{tx_last:.3f}], rx_range=[{rx_x0:.3f},{rx_last:.3f}]"
        )

    # outer then inner, so inner overrides to core material
    transition_lines = _ellipse_slices(
        x_center=x_center,
        z_center=z_center,
        x_radius=tr_xr,
        z_radius=tr_zr,
        z_min=leak_z_min,
        z_max=leak_z_max,
        y_min=y_lo,
        y_max=y_hi,
        dx=dx,
        x_lo=safe_x_min,
        x_hi=safe_x_max,
        material="oil_transition",
    )
    core_lines = _ellipse_slices(
        x_center=x_center,
        z_center=z_center,
        x_radius=core_xr,
        z_radius=core_zr,
        z_min=leak_z_min,
        z_max=leak_z_max,
        y_min=y_lo,
        y_max=y_hi,
        dx=dx,
        x_lo=safe_x_min,
        x_hi=safe_x_max,
        material="oil_core",
    )

    lines = [
        f"#title: SFCW oil leak scene ({freq_hz/1e6:.1f} MHz)",
        f"#domain: {x_size:.3f} {y_size:.3f} {z_size:.3f}",
        f"#dx_dy_dz: {dx:.3f} {dy:.3f} {dz:.3f}",
        f"#time_window: {time_window:.3e}",
        f"#pml_cells: {pml_cells} {pml_cells} {pml_cells} {pml_cells} {pml_cells} {pml_cells}",
        "",
        f"#material: {er_bg:.3f} {sig_bg:.5f} 1.0 0.0 soil_bg",
        f"#material: {er_transition:.3f} {sig_transition:.5f} 1.0 0.0 oil_transition",
        f"#material: {er_core:.3f} {sig_core:.5f} 1.0 0.0 oil_core",
        "",
        "#box: 0.000 0.000 0.000 0.500 0.100 3.500 free_space",
        f"#box: 0.000 0.000 {z_surface:.3f} 0.500 0.100 3.500 soil_bg",
        "",
        "## Oil contamination: transition zone (outer ellipse cylinder)",
        *transition_lines,
        "",
        "## Oil contamination: core zone (inner ellipse cylinder)",
        *core_lines,
        "",
        f"#waveform: {waveform} 1.0 {freq_hz:.6e} txpulse",
        f"#hertzian_dipole: z {tx_x0:.3f} {y_ant:.3f} {z_ant:.3f} txpulse",
        f"#rx: {rx_x0:.3f} {y_ant:.3f} {z_ant:.3f}",
        f"#src_steps: {trace_step:.3f} 0.000 0.000",
        f"#rx_steps: {trace_step:.3f} 0.000 0.000",
        "",
        "## Run example:",
        f"## python -m gprMax <this_file>.in -n {traces}",
    ]

    meta: Dict[str, float | int | str] = {
        "frequency_hz": float(freq_hz),
        "traces": traces,
        "tx_x0": tx_x0,
        "rx_x0": rx_x0,
        "tx_rx_spacing": tx_rx_spacing,
        "trace_step": trace_step,
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "x_size": x_size,
        "y_size": y_size,
        "z_size": z_size,
        "time_window": time_window,
        "pml_cells": pml_cells,
        "z_surface": z_surface,
        "antenna_height": antenna_height,
        "waveform": waveform,
    }
    return "\n".join(lines) + "\n", meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate stepped-frequency gprMax .in files for fixed oil-leak scene"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sim_inputs") / "sfcw_oil_720_2500",
        help="Directory to write generated .in files",
    )
    parser.add_argument("--f-start-mhz", type=float, default=720.0, help="Start frequency in MHz")
    parser.add_argument("--f-stop-mhz", type=float, default=2500.0, help="Stop frequency in MHz")
    parser.add_argument(
        "--f-step-mhz",
        type=float,
        default=20.0,
        help="Frequency step in MHz (default: 20)",
    )
    parser.add_argument(
        "--waveform",
        type=str,
        default="contsine",
        choices=["contsine", "gaussiansine", "ricker"],
        help="Waveform type for each frequency point (default: contsine)",
    )
    parser.add_argument("--traces", type=int, default=96, help="Trace count for each .in (default: 96)")
    parser.add_argument(
        "--tx-rx-spacing",
        type=float,
        default=0.1,
        help="Tx-Rx spacing in meters (default: 0.1)",
    )
    parser.add_argument(
        "--tx-x0",
        type=float,
        default=0.06,
        help="Initial Tx x-position in meters (default: 0.06)",
    )
    parser.add_argument(
        "--trace-step",
        type=float,
        default=0.003,
        help="Scan step per trace in x direction (default: 0.003)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.f_start_mhz <= 0 or args.f_stop_mhz <= 0:
        raise ValueError("Frequencies must be > 0")
    if args.f_stop_mhz < args.f_start_mhz:
        raise ValueError("--f-stop-mhz must be >= --f-start-mhz")
    if args.f_step_mhz <= 0:
        raise ValueError("--f-step-mhz must be > 0")
    if args.traces < 2:
        raise ValueError("--traces must be >= 2")
    if args.tx_rx_spacing <= 0:
        raise ValueError("--tx-rx-spacing must be > 0")
    if args.trace_step <= 0:
        raise ValueError("--trace-step must be > 0")

    freqs_mhz = _frange_inclusive(args.f_start_mhz, args.f_stop_mhz, args.f_step_mhz)
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata: Dict[str, object] = {
        "f_start_mhz": args.f_start_mhz,
        "f_stop_mhz": args.f_stop_mhz,
        "f_step_mhz": args.f_step_mhz,
        "waveform": args.waveform,
        "traces": args.traces,
        "tx_rx_spacing": args.tx_rx_spacing,
        "tx_x0": args.tx_x0,
        "trace_step": args.trace_step,
        "files": [],
    }

    for f_mhz in freqs_mhz:
        f_hz = f_mhz * 1e6
        content, meta = build_scene(
            freq_hz=f_hz,
            traces=args.traces,
            tx_x0=args.tx_x0,
            tx_rx_spacing=args.tx_rx_spacing,
            trace_step=args.trace_step,
            waveform=args.waveform,
        )
        filename = f"sfcw_oil_{int(round(f_mhz)):04d}MHz.in"
        path = out_dir / filename
        path.write_text(content, encoding="ascii")
        metadata["files"].append(
            {
                "file": filename,
                "frequency_mhz": f_mhz,
                "frequency_hz": f_hz,
                **meta,
            }
        )

    meta_path = out_dir / "sfcw_oil_metadata.json"
    with meta_path.open("w", encoding="ascii") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=True)

    print(f"[done] generated {len(freqs_mhz)} files in: {out_dir}")
    print(f"[done] metadata: {meta_path}")
    if freqs_mhz:
        print("[hint] run one frequency example:")
        print(
            f"  python -m gprMax \"{(out_dir / f'sfcw_oil_{int(round(freqs_mhz[0])):04d}MHz.in')}\" -n {args.traces}"
        )


if __name__ == "__main__":
    main()
