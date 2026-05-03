"""
一体化流水线：仿真 → 出灰度 B-scan 图像。
用法:
  单个文件（仿真+出图）:  python run_pipeline.py bag_small_flat_d020_x25.in
  全部文件:              python run_pipeline.py --all
  跳过仿真只出图:         python run_pipeline.py --all --no-sim
"""

import subprocess
import sys
import os
import glob
from pathlib import Path

import plot_bscan   # 复用 plot_bscan.py 中的 process()

SCRIPT_DIR = Path(__file__).parent.resolve()
N_SCANS    = 21


def run_sim(in_file: str) -> bool:
    print(f"\n[仿真] {in_file}")
    ret = subprocess.run(
        [sys.executable, "-m", "gprMax", in_file, "-n", str(N_SCANS)],
        cwd=SCRIPT_DIR
    )
    if ret.returncode != 0:
        print(f"  [错误] 仿真失败: {in_file}")
        return False
    return True


def process(in_file: str, run_simulation: bool = True) -> bool:
    in_path = SCRIPT_DIR / in_file
    if not in_path.exists():
        print(f"[跳过] 文件不存在: {in_file}")
        return False

    if run_simulation:
        if not run_sim(in_file):
            return False

    stem = Path(in_file).stem
    print(f"[出图] stem={stem}")
    return plot_bscan.process(stem, resize=(256, 256))


def main():
    args          = sys.argv[1:]
    run_simulation = "--no-sim" not in args
    args          = [a for a in args if not a.startswith("--")]

    if not args and "--all" not in sys.argv:
        print("用法:")
        print("  python run_pipeline.py bag_small_flat_d020_x25.in")
        print("  python run_pipeline.py --all")
        print("  python run_pipeline.py --all --no-sim")
        sys.exit(0)

    if not args:  # --all
        in_files = sorted(SCRIPT_DIR.glob("bag_*.in"))
        in_files += list(SCRIPT_DIR.glob("baseline.in"))
        in_files = [f.name for f in in_files]
    else:
        in_files = args

    total   = len(in_files)
    success = 0
    for i, f in enumerate(in_files, 1):
        print(f"\n{'='*55}")
        print(f"[{i}/{total}] {f}")
        if process(f, run_simulation=run_simulation):
            success += 1

    print(f"\n{'='*55}")
    print(f"完成: {success}/{total}  →  {plot_bscan.PREVIEW_DIR}")


if __name__ == "__main__":
    main()
