#!/usr/bin/env python
"""Run gprMax B-scan simulation and export grayscale image in one command."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _build_runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    prefix = Path(sys.executable).resolve().parent
    path_parts = [
        str(prefix / "Library" / "bin"),
        str(prefix / "Scripts"),
        str(prefix / "bin"),
        str(prefix),
    ]
    existing = env.get("PATH", "")
    env["PATH"] = os.pathsep.join(path_parts + ([existing] if existing else []))
    return env


def _run_command(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("[cmd]", " ".join(cmd))
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}: {' '.join(cmd)}")


def _find_merged_output(input_file: Path) -> Path:
    stem = input_file.with_suffix("")
    candidates = [
        stem.with_name(stem.name + "_merged").with_suffix(".out"),
        stem.with_suffix(".out"),
    ]

    for c in candidates:
        if c.exists():
            return c

    wildcard = sorted(input_file.parent.glob(f"{input_file.stem}*.out"))
    if wildcard:
        return wildcard[-1]

    raise FileNotFoundError(
        "Cannot find merged .out file. Checked: "
        + ", ".join(str(c) for c in candidates)
    )


def _check_gpu_toolchain(env: dict[str, str]) -> None:
    if shutil.which("nvcc", path=env.get("PATH")) is None:
        raise RuntimeError(
            "GPU mode requested but 'nvcc' was not found in PATH. "
            "Install CUDA Toolkit and ensure nvcc is available."
        )
    if shutil.which("cl.exe", path=env.get("PATH")) is None:
        raise RuntimeError(
            "GPU mode requested but 'cl.exe' (MSVC compiler) was not found in PATH. "
            "Install Visual Studio Build Tools (Desktop development with C++) "
            "and run from a Developer Command Prompt."
        )

    check = subprocess.run(
        [sys.executable, "-c", "import pycuda.driver as drv; print('pycuda_ok')"],
        capture_output=True,
        text=True,
        env=env,
    )
    if check.returncode != 0:
        raise RuntimeError(
            "GPU mode requested but pycuda is not working in this environment. "
            "Install/repair pycuda in the current conda env."
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run gprMax .in simulation then export grayscale B-scan"
    )
    parser.add_argument("--input", required=True, type=Path, help="Input .in file")
    parser.add_argument(
        "--traces",
        type=int,
        default=220,
        help="Number of traces for B-scan (default: 220)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Pass -gpu to gprMax",
    )
    parser.add_argument(
        "--remove-trace-files",
        action="store_true",
        help="Ask outputfiles_merge to remove per-trace output files",
    )
    parser.add_argument(
        "--output-image",
        required=True,
        type=Path,
        help="Output B-scan grayscale PNG path",
    )
    parser.add_argument(
        "--rx", default="rx1", help="Receiver group name in HDF5 (default: rx1)"
    )
    parser.add_argument(
        "--component",
        default="Ez",
        help="Field component in HDF5 (default: Ez)",
    )
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        help="Optional output image size, e.g. --resize 256 256",
    )
    parser.add_argument(
        "--gain-alpha",
        type=float,
        default=1.8,
        help="Exponential time gain alpha (default: 1.8)",
    )
    parser.add_argument(
        "--clip-percentile",
        type=float,
        default=99.5,
        help="Clip percentile for robust normalization (default: 99.5)",
    )
    parser.add_argument(
        "--zero-time-correct",
        action="store_true",
        help="Apply per-trace zero-time correction during B-scan export",
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.traces < 2:
        raise ValueError("--traces must be >= 2 for B-scan generation.")

    in_file = args.input.resolve()
    if not in_file.exists():
        raise FileNotFoundError(f"Input file not found: {in_file}")

    runtime_env = _build_runtime_env()

    if args.gpu:
        _check_gpu_toolchain(runtime_env)

    gpr_cmd = [sys.executable, "-m", "gprMax", str(in_file), "-n", str(args.traces)]
    if args.gpu:
        gpr_cmd.append("-gpu")
    _run_command(gpr_cmd, env=runtime_env)

    merge_cmd = [sys.executable, "-m", "tools.outputfiles_merge", str(in_file.with_suffix(""))]
    if args.remove_trace_files:
        merge_cmd.append("--remove-files")
    _run_command(merge_cmd, env=runtime_env)

    merged_out = _find_merged_output(in_file)

    export_cmd = [
        sys.executable,
        str((Path(__file__).parent / "export_bscan.py").resolve()),
        "--input",
        str(merged_out),
        "--output",
        str(args.output_image.resolve()),
        "--rx",
        args.rx,
        "--component",
        args.component,
        "--gain-alpha",
        str(args.gain_alpha),
        "--clip-percentile",
        str(args.clip_percentile),
    ]

    if args.zero_time_correct:
        export_cmd.append("--zero-time-correct")
    export_cmd.extend(
        [
            "--zero-time-search-ratio",
            str(args.zero_time_search_ratio),
            "--zero-time-smooth",
            str(args.zero_time_smooth),
        ]
    )

    if args.resize is not None:
        export_cmd.extend(["--resize", str(args.resize[0]), str(args.resize[1])])

    _run_command(export_cmd, env=runtime_env)
    print(f"B-scan image saved to: {args.output_image.resolve()}")


if __name__ == "__main__":
    main()
