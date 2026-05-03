"""
一键完整流水线：生成 .in → 仿真 → 出 B-scan 图 → 删除 .out → labels.csv

每个场景按顺序独立处理：仿真完成后立即出图，出图成功后删除 .out 文件，
保持目录整洁，不堆积中间文件。

用法:
  python run_all.py                     # 全流程（生成 + 仿真 + 出图）
  python run_all.py --no-gen            # 跳过生成（用已有 .in 文件）
  python run_all.py --no-sim            # 跳过仿真（用已有 .out 文件）
  python run_all.py --no-gen --no-sim   # 只出图 + 生成 labels.csv
  python run_all.py --only leak         # 只处理 leak_ 文件
  python run_all.py --only noleak       # 只处理 noleak_ 文件
  python run_all.py --gpu 0             # 使用 GPU 0 加速仿真（需要 pycuda）
  python run_all.py --keep-out          # 出图后保留 .out 文件（默认删除）

输出:
  bscan_preview/{stem}_bscan.png        256×256 灰度 B-scan 图像
  labels.csv                            每行: filename,label,bag_type,soil,depth_m,cx_m,clutter
"""

import subprocess
import sys
import csv
import re
import time
from pathlib import Path

import plot_bscan

SCRIPT_DIR = Path(__file__).parent.resolve()
N_SCANS    = 21
LABELS_CSV = SCRIPT_DIR / "labels.csv"


# ── Helpers ───────────────────────────────────────────────────────────────────

def run_sim(fname: str, gpu_id: int | None = None) -> bool:
    cmd = [sys.executable, "-m", "gprMax", fname, "-n", str(N_SCANS)]
    if gpu_id is not None:
        cmd += ["-gpu", str(gpu_id)]
    ret = subprocess.run(cmd, cwd=SCRIPT_DIR)
    return ret.returncode == 0


def delete_out_files(stem: str) -> int:
    """删除 stem 对应的全部 .out 文件，返回删除数量。"""
    deleted = 0
    for f in SCRIPT_DIR.glob(f"{stem}*.out"):
        if re.fullmatch(r"\d+", f.stem[len(stem):]):
            f.unlink()
            deleted += 1
    return deleted


def parse_stem(stem: str) -> dict:
    """Extract label metadata from file stem."""
    # noleak_bare_{soil_key}
    m = re.match(r"noleak_bare_(.+)", stem)
    if m:
        return {"label": 0, "bag_type": "", "soil": m.group(1),
                "depth_m": "", "cx_m": "", "clutter": "bare"}
    # noleak_rock{size}_{soil_key}_d{D}_x{X}
    m = re.match(r"noleak_(rock\w+)_(soil_\w+)_d(\d+)_x(\d+)", stem)
    if m:
        return {"label": 0, "bag_type": "", "soil": m.group(2),
                "depth_m": int(m.group(3)) / 100, "cx_m": int(m.group(4)) / 100,
                "clutter": m.group(1)}
    # noleak_pipe_{soil_key}_d{D}_x{X}
    m = re.match(r"noleak_pipe_(soil_\w+)_d(\d+)_x(\d+)", stem)
    if m:
        return {"label": 0, "bag_type": "", "soil": m.group(1),
                "depth_m": int(m.group(2)) / 100, "cx_m": int(m.group(3)) / 100,
                "clutter": "pipe"}
    # leak_{bag_type}_{soil_key}_d{D}_x{X}
    m = re.match(r"leak_(.+?)_(soil_\w+)_d(\d+)_x(\d+)", stem)
    if m:
        return {"label": 1, "bag_type": m.group(1), "soil": m.group(2),
                "depth_m": int(m.group(3)) / 100, "cx_m": int(m.group(4)) / 100,
                "clutter": ""}
    return {"label": -1, "bag_type": "", "soil": "", "depth_m": "", "cx_m": "", "clutter": ""}


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    argv = sys.argv[1:]
    do_gen    = "--no-gen"    not in argv
    do_sim    = "--no-sim"    not in argv
    keep_out  = "--keep-out"  in argv

    gpu_id = None
    if "--gpu" in argv:
        idx = argv.index("--gpu")
        gpu_id = int(argv[idx + 1]) if idx + 1 < len(argv) else 0

    only_filter = None
    if "--only" in argv:
        idx = argv.index("--only")
        only_filter = argv[idx + 1] if idx + 1 < len(argv) else None

    t0 = time.time()

    # ── Step 1: Generate .in files ────────────────────────────────────────────
    print("=" * 65)
    if do_gen:
        print("[1/3] 生成仿真文件 (.in) ...")
        subprocess.run([sys.executable, "generate_cases.py"], cwd=SCRIPT_DIR, check=True)
    else:
        print("[1/3] 跳过生成（--no-gen）")

    # ── Collect .in files ─────────────────────────────────────────────────────
    leak_files   = sorted(SCRIPT_DIR.glob("leak_*.in"))
    noleak_files = sorted(SCRIPT_DIR.glob("noleak_*.in"))

    if only_filter == "leak":
        in_files = leak_files
    elif only_filter == "noleak":
        in_files = noleak_files
    else:
        in_files = noleak_files + leak_files   # noleak first (faster)

    total = len(in_files)
    print(f"\n找到 {len(noleak_files)} 个 noleak + {len(leak_files)} 个 leak = {total} 个文件")

    # ── Step 2+3: Simulate → Plot → (Delete .out) ─────────────────────────────
    print("\n" + "=" * 65)
    gpu_info  = f"  [GPU {gpu_id}]" if gpu_id is not None else "  [CPU]"
    keep_info = "  [保留 .out]" if keep_out else "  [出图后删除 .out]"
    print(f"[2/3] 仿真 + 出图（每个场景 {N_SCANS} 道）{gpu_info}{keep_info}")

    rows     = []
    sim_ok   = sim_skip = sim_fail = 0
    plot_ok  = plot_fail = 0
    out_del  = 0

    for i, fpath in enumerate(in_files, 1):
        fname = fpath.name
        stem  = fpath.stem

        elapsed = time.time() - t0
        eta_str = ""
        if i > 1:
            avg = elapsed / (i - 1)
            remaining = avg * (total - i + 1)
            eta_str = f"  ETA≈{fmt_time(remaining)}"

        print(f"\n{'─' * 65}")
        print(f"[{i}/{total}] {fname}{eta_str}")

        # ── 仿真 ──────────────────────────────────────────────────────────────
        if do_sim:
            if run_sim(fname, gpu_id=gpu_id):
                sim_ok += 1
            else:
                print(f"  [错误] 仿真失败，跳过出图")
                sim_fail += 1
                row = parse_stem(stem)
                row["filename"] = f"bscan_preview/{stem}_bscan.png"
                rows.append(row)
                continue
        else:
            sim_skip += 1

        # ── 出图 ──────────────────────────────────────────────────────────────
        ok = plot_bscan.process(stem, resize=(256, 256))
        if ok:
            plot_ok += 1
        else:
            plot_fail += 1

        # ── 删除 .out ─────────────────────────────────────────────────────────
        if not keep_out and (do_sim or ok):
            n = delete_out_files(stem)
            if n:
                out_del += n
                print(f"  [清理] 删除 {n} 个 .out 文件")

        row = parse_stem(stem)
        row["filename"] = f"bscan_preview/{stem}_bscan.png"
        rows.append(row)

    # ── Write labels.csv ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("[3/3] 写入 labels.csv ...")
    fieldnames = ["filename", "label", "bag_type", "soil", "depth_m", "cx_m", "clutter"]
    with open(LABELS_CSV, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_time = time.time() - t0
    print(f"\n{'=' * 65}")
    print(f"全流程完成  ({fmt_time(total_time)})")
    print(f"  仿真: {sim_ok} 成功 / {sim_fail} 失败 / {sim_skip} 跳过")
    print(f"  出图: {plot_ok} 成功 / {plot_fail} 失败")
    if not keep_out:
        print(f"  清理: 共删除 {out_del} 个 .out 文件")
    print(f"  标签: {LABELS_CSV}  ({len(rows)} 行)")
    print(f"  图像: {plot_bscan.PREVIEW_DIR}")
    n0 = sum(1 for r in rows if r["label"] == 0)
    n1 = sum(1 for r in rows if r["label"] == 1)
    print(f"  类别分布: noleak={n0}  leak={n1}")


if __name__ == "__main__":
    main()
