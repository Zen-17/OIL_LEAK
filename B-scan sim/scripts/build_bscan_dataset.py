#!/usr/bin/env python
"""Automated pipeline: generate varied gprMax scenes and export many B-scan images."""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from generate_gprmax_in import (
    NEG_SCENARIOS,
    NEG_SCENARIOS_SANDBOX,
    POS_SCENARIOS,
    POS_SCENARIOS_SANDBOX,
    PROFILES,
    build_scene,
)


@dataclass
class SampleTask:
    index: int
    label: str
    scenario: str
    frequency_hz: float
    seed: int
    traces: int
    gain_alpha: float
    clip_percentile: float
    mute_top_ratio: float


def _build_target_env(python_exe: Path) -> Dict[str, str]:
    env = os.environ.copy()
    prefix = python_exe.resolve().parent
    path_parts = [
        str(prefix / "Library" / "bin"),
        str(prefix / "Scripts"),
        str(prefix / "bin"),
        str(prefix),
    ]
    existing = env.get("PATH", "")
    env["PATH"] = os.pathsep.join(path_parts + ([existing] if existing else []))
    return env


def _run_command(cmd: List[str], cwd: Path | None = None, env: Dict[str, str] | None = None) -> None:
    print("[cmd]", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def _check_gpu_toolchain(python_exe: Path, env: Dict[str, str]) -> None:
    nvcc_path = shutil.which("nvcc", path=env.get("PATH"))
    if nvcc_path is None:
        raise RuntimeError(
            "GPU mode requested but 'nvcc' was not found for the target environment. "
            "Install CUDA Toolkit (or cuda-nvcc) and ensure nvcc is discoverable."
        )

    check_nvcc = subprocess.run(
        [str(python_exe), "-c", "import shutil,sys; sys.exit(0 if shutil.which('nvcc') else 1)"],
        env=env,
    )
    if check_nvcc.returncode != 0:
        raise RuntimeError(
            "GPU mode requested but Python subprocess cannot find 'nvcc' in PATH."
        )

    if shutil.which("cl.exe", path=env.get("PATH")) is None:
        raise RuntimeError(
            "GPU mode requested but 'cl.exe' (MSVC compiler) was not found in PATH. "
            "Install Visual Studio Build Tools (Desktop development with C++) "
            "and run this command from a Developer Command Prompt."
        )

    check_pycuda = subprocess.run(
        [str(python_exe), "-c", "import pycuda.driver as drv; print('pycuda_ok')"],
        env=env,
    )
    if check_pycuda.returncode != 0:
        raise RuntimeError(
            "GPU mode requested but pycuda is unavailable in the target environment."
        )


def _find_merged_out(input_file: Path) -> Path:
    candidate = input_file.with_name(input_file.stem + "_merged.out")
    if candidate.exists():
        return candidate

    wildcard = sorted(input_file.parent.glob(f"{input_file.stem}*_merged.out"))
    if wildcard:
        return wildcard[-1]

    raise FileNotFoundError(f"Merged out file not found for {input_file}")


def _parse_frequencies(text: str) -> List[float]:
    values: List[float] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("No valid frequencies provided")
    return values


def _make_task(
    rng: random.Random,
    index: int,
    traces: int,
    oil_ratio: float,
    frequencies_hz: List[float],
    base_seed: int,
    profile: str,
    fixed_mute_top_ratio: float | None,
    fixed_gain_alpha: float | None,
    fixed_clip_percentile: float | None,
) -> SampleTask:
    label = "oil" if rng.random() < oil_ratio else "no_oil"
    if profile == "sandbox_local_easy":
        scenario = rng.choice(
            POS_SCENARIOS_SANDBOX if label == "oil" else NEG_SCENARIOS_SANDBOX
        )
    else:
        scenario = rng.choice(POS_SCENARIOS if label == "oil" else NEG_SCENARIOS)
    frequency_hz = rng.choice(frequencies_hz)

    # Use deterministic per-sample seed so names are reproducible.
    seed = base_seed + index * 7919 + rng.randint(1, 1000)

    if profile == "sandbox_local_easy":
        gain_alpha = (
            fixed_gain_alpha if fixed_gain_alpha is not None else rng.uniform(1.7, 2.5)
        )
        clip_percentile = (
            fixed_clip_percentile
            if fixed_clip_percentile is not None
            else rng.uniform(98.8, 99.5)
        )
        mute_top_ratio = (
            fixed_mute_top_ratio
            if fixed_mute_top_ratio is not None
            else rng.uniform(0.05, 0.10)
        )
    else:
        gain_alpha = (
            fixed_gain_alpha if fixed_gain_alpha is not None else rng.uniform(1.3, 2.3)
        )
        clip_percentile = (
            fixed_clip_percentile
            if fixed_clip_percentile is not None
            else rng.uniform(98.8, 99.8)
        )
        mute_top_ratio = fixed_mute_top_ratio if fixed_mute_top_ratio is not None else 0.0

    return SampleTask(
        index=index,
        label=label,
        scenario=scenario,
        frequency_hz=frequency_hz,
        seed=seed,
        traces=traces,
        gain_alpha=gain_alpha,
        clip_percentile=clip_percentile,
        mute_top_ratio=mute_top_ratio,
    )


def _sample_name(task: SampleTask) -> str:
    freq_mhz = int(round(task.frequency_hz / 1e6))
    return (
        f"{task.index:05d}_{task.label}_{task.scenario}_"
        f"f{freq_mhz}MHz_t{task.traces}_s{task.seed}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate many varied B-scan images with full gprMax automation"
    )
    parser.add_argument(
        "--count",
        type=int,
        required=True,
        help="Number of B-scan images to generate",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting index for sample naming/metadata (default: 0)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dataset_auto"),
        help="Output root directory (default: dataset_auto)",
    )
    parser.add_argument(
        "--traces",
        type=int,
        default=220,
        help="Number of traces per B-scan (default: 220)",
    )
    parser.add_argument(
        "--oil-ratio",
        type=float,
        default=0.5,
        help="Probability of oil-positive samples in [0,1] (default: 0.5)",
    )
    parser.add_argument(
        "--frequencies",
        type=str,
        default="6.0e8,8.0e8,9.0e8,1.0e9",
        help="Comma-separated center frequencies in Hz",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="realistic",
        choices=PROFILES,
        help="Simulation profile (default: realistic)",
    )
    parser.add_argument(
        "--sandbox-tilt-max-deg",
        type=float,
        default=10.0,
        help="Max absolute tilt angle (deg) for sandbox anomalies (default: 10)",
    )
    parser.add_argument(
        "--sandbox-clutter-level",
        type=float,
        default=0.35,
        help="Sandbox pipe/stone clutter intensity in [0,1] (default: 0.35)",
    )
    parser.add_argument(
        "--simple-water",
        action="store_true",
        help="Use medium-low moisture band with reduced variability in sandbox profile",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260331,
        help="Global random seed (default: 20260331)",
    )
    parser.add_argument(
        "--python-exe",
        type=Path,
        default=Path(sys.executable),
        help="Python executable that has gprMax installed",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use gprMax GPU solver (-gpu). Requires pycuda in the target env.",
    )
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=(256, 256),
        help="Output image size (default: 256 256)",
    )
    parser.add_argument(
        "--keep-merged-out",
        action="store_true",
        help="Keep merged .out files under output_root/out",
    )
    parser.add_argument(
        "--keep-in-files",
        action="store_true",
        help="Keep generated .in files under output_root/inputs",
    )
    parser.add_argument(
        "--mute-top-ratio",
        type=float,
        default=None,
        help=(
            "Optional fixed top-mute ratio in [0,1) for exported images. "
            "If omitted, sandbox_local_easy uses randomized 0.05~0.10 and realistic uses 0.0."
        ),
    )
    parser.add_argument(
        "--mute-fade-ratio",
        type=float,
        default=0.03,
        help="Fade-in ratio in [0,1) after top mute (default: 0.03)",
    )
    parser.add_argument(
        "--gain-alpha-fixed",
        type=float,
        default=None,
        help="Optional fixed gain alpha for all samples (default: randomized)",
    )
    parser.add_argument(
        "--clip-percentile-fixed",
        type=float,
        default=None,
        help="Optional fixed clip percentile for all samples (default: randomized)",
    )
    parser.add_argument(
        "--dewow-window",
        type=int,
        default=9,
        help="Dewow moving-average window passed to exporter (default: 9)",
    )
    parser.add_argument(
        "--no-background-remove",
        action="store_true",
        help="Disable background trace-mean subtraction during PNG export",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only generate .in and metadata, skip simulation/export",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.count <= 0:
        raise ValueError("--count must be > 0")
    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0")
    if args.traces < 2:
        raise ValueError("--traces must be >= 2 for B-scan")
    if not (0.0 <= args.oil_ratio <= 1.0):
        raise ValueError("--oil-ratio must be in [0,1]")
    if args.gpu and args.dry_run:
        print("Warning: --gpu has no effect in --dry-run mode.")
    if args.sandbox_tilt_max_deg < 0.0:
        raise ValueError("--sandbox-tilt-max-deg must be >= 0")
    if not (0.0 <= args.sandbox_clutter_level <= 1.0):
        raise ValueError("--sandbox-clutter-level must be in [0,1]")
    if args.mute_top_ratio is not None and not (0.0 <= args.mute_top_ratio < 1.0):
        raise ValueError("--mute-top-ratio must be in [0,1)")
    if not (0.0 <= args.mute_fade_ratio < 1.0):
        raise ValueError("--mute-fade-ratio must be in [0,1)")
    if args.gain_alpha_fixed is not None and args.gain_alpha_fixed < 0.0:
        raise ValueError("--gain-alpha-fixed must be >= 0")
    if args.clip_percentile_fixed is not None and not (0.0 < args.clip_percentile_fixed <= 100.0):
        raise ValueError("--clip-percentile-fixed must be in (0,100]")
    if args.dewow_window < 1:
        raise ValueError("--dewow-window must be >= 1")

    frequencies_hz = _parse_frequencies(args.frequencies)
    target_env = _build_target_env(args.python_exe)

    root = args.output_root.resolve()
    inputs_dir = root / "inputs"
    out_dir = root / "out"
    images_dir = root / "images"
    oil_img_dir = images_dir / "oil"
    no_oil_img_dir = images_dir / "no_oil"
    meta_path = root / "metadata.csv"

    for d in (root, inputs_dir, out_dir, oil_img_dir, no_oil_img_dir):
        d.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    if args.gpu and not args.dry_run:
        _check_gpu_toolchain(args.python_exe, target_env)

    rows: List[Dict[str, object]] = []

    for offset in range(args.count):
        idx = args.start_index + offset
        task = _make_task(
            rng=rng,
            index=idx,
            traces=args.traces,
            oil_ratio=args.oil_ratio,
            frequencies_hz=frequencies_hz,
            base_seed=args.seed,
            profile=args.profile,
            fixed_mute_top_ratio=args.mute_top_ratio,
            fixed_gain_alpha=args.gain_alpha_fixed,
            fixed_clip_percentile=args.clip_percentile_fixed,
        )

        sample_name = _sample_name(task)
        in_path = inputs_dir / f"{sample_name}.in"

        content, sim_meta = build_scene(
            label=task.label,
            freq_hz=task.frequency_hz,
            scenario=task.scenario,
            seed=task.seed,
            profile=args.profile,
            sandbox_tilt_max_deg=args.sandbox_tilt_max_deg,
            sandbox_clutter_level=args.sandbox_clutter_level,
            simplify_water=args.simple_water,
        )
        in_path.write_text(content, encoding="ascii")

        img_dir = oil_img_dir if task.label == "oil" else no_oil_img_dir
        img_path = img_dir / f"{sample_name}.png"

        merged_out_path: Path | None = None

        if not args.dry_run:
            gpr_cmd = [
                str(args.python_exe),
                "-m",
                "gprMax",
                str(in_path),
                "-n",
                str(task.traces),
            ]
            if args.gpu:
                gpr_cmd.append("-gpu")
            _run_command(gpr_cmd, env=target_env)

            merge_cmd = [
                str(args.python_exe),
                "-m",
                "tools.outputfiles_merge",
                str(in_path.with_suffix("")),
                "--remove-files",
            ]
            _run_command(merge_cmd, env=target_env)

            merged_out_path = _find_merged_out(in_path)

            export_cmd = [
                str(args.python_exe),
                str((Path(__file__).parent / "export_bscan.py").resolve()),
                "--input",
                str(merged_out_path),
                "--output",
                str(img_path),
                "--resize",
                str(args.resize[0]),
                str(args.resize[1]),
                "--dewow-window",
                str(args.dewow_window),
                "--gain-alpha",
                f"{task.gain_alpha:.4f}",
                "--clip-percentile",
                f"{task.clip_percentile:.4f}",
                "--mute-top-ratio",
                f"{task.mute_top_ratio:.4f}",
                "--mute-fade-ratio",
                f"{args.mute_fade_ratio:.4f}",
            ]
            if args.no_background_remove:
                export_cmd.append("--no-background-remove")
            _run_command(export_cmd, env=target_env)

            if args.keep_merged_out:
                keep_path = out_dir / merged_out_path.name
                shutil.copy2(merged_out_path, keep_path)
            else:
                merged_out_path.unlink(missing_ok=True)

        if not args.keep_in_files:
            in_path.unlink(missing_ok=True)

        merged_saved = merged_out_path is not None and merged_out_path.exists()

        row: Dict[str, object] = {
            "index": task.index,
            "sample_name": sample_name,
            "label": task.label,
            "scenario": task.scenario,
            "profile": args.profile,
            "frequency_hz": task.frequency_hz,
            "traces": task.traces,
            "seed": task.seed,
            "gain_alpha": round(task.gain_alpha, 4),
            "clip_percentile": round(task.clip_percentile, 4),
            "mute_top_ratio": round(task.mute_top_ratio, 4),
            "dewow_window": args.dewow_window,
            "no_background_remove": int(args.no_background_remove),
            "image_path": str(img_path.relative_to(root)) if img_path.exists() else "",
            "input_path": str(in_path.relative_to(root)) if in_path.exists() else "",
            "merged_out_path": (
                str(merged_out_path.relative_to(root))
                if merged_saved and merged_out_path is not None
                else ""
            ),
            "oil_type": sim_meta.get("oil_type", ""),
            "fractal_dimension": sim_meta.get("fractal_dimension", ""),
            "fractal_bins": sim_meta.get("fractal_bins", ""),
            "water_low": sim_meta.get("water_low", ""),
            "water_high": sim_meta.get("water_high", ""),
            "oil_tilt_deg": sim_meta.get("oil_tilt_deg", ""),
            "oil_mirror_lr": sim_meta.get("oil_mirror_lr", ""),
            "sandbox_pipe_count": sim_meta.get("sandbox_pipe_count", ""),
            "sandbox_stone_count": sim_meta.get("sandbox_stone_count", ""),
            "sandbox_clutter_level": sim_meta.get("sandbox_clutter_level", ""),
            "simplify_water": sim_meta.get("simplify_water", ""),
        }
        rows.append(row)

        print(f"[ok] {offset + 1}/{args.count}: {sample_name}")

    fieldnames = [
        "index",
        "sample_name",
        "label",
        "scenario",
        "profile",
        "frequency_hz",
        "traces",
        "seed",
        "gain_alpha",
        "clip_percentile",
        "mute_top_ratio",
        "dewow_window",
        "no_background_remove",
        "image_path",
        "input_path",
        "merged_out_path",
        "oil_type",
        "fractal_dimension",
        "fractal_bins",
        "water_low",
        "water_high",
        "oil_tilt_deg",
        "oil_mirror_lr",
        "sandbox_pipe_count",
        "sandbox_stone_count",
        "sandbox_clutter_level",
        "simplify_water",
    ]

    with meta_path.open("w", newline="", encoding="ascii") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Metadata: {meta_path}")


if __name__ == "__main__":
    main()
