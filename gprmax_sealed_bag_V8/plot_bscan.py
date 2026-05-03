"""
从 gprMax 多个 A-scan .out 文件读取 Ez，拼成 B-scan 并保存灰度 PNG。
处理流程与参考数据集 export_bscan.py 完全一致：
  1. 读取各 .out 文件的 Ez，按道序号排列
  2. Dewow（移动平均去趋势）
  3. 背景去除（减去各时刻均值）
  4. 顶部静音（压制直达波）
  5. 时间增益（补偿深度衰减）
  6. 百分位数裁剪 + 映射到 uint8 灰度
  7. Lanczos 缩放，保存 PNG

用法:
  python plot_bscan.py bag_small_flat_d020_x25
  python plot_bscan.py bag_small_flat_d020_x25 --resize 256 256
  python plot_bscan.py --all
"""

import sys
import os
import re
import glob
import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

SCRIPT_DIR  = Path(__file__).parent.resolve()
PREVIEW_DIR = SCRIPT_DIR / "bscan_preview"
PREVIEW_DIR.mkdir(exist_ok=True)


# ── 信号处理函数（与参考 export_bscan.py 保持一致）───────────────────────────

def _dewow(data: np.ndarray, window: int = 9) -> np.ndarray:
    """移动平均去趋势（去除低频漂移）。"""
    if window <= 1:
        return data
    kernel = np.ones(window, dtype=np.float32) / window
    out = np.empty_like(data)
    for i in range(data.shape[1]):
        out[:, i] = data[:, i] - np.convolve(data[:, i], kernel, mode="same")
    return out


def _background_remove(data: np.ndarray) -> np.ndarray:
    """按时刻减去所有道的均值（去除水平条带背景）。"""
    return data - data.mean(axis=1, keepdims=True)


def _mute_top(data: np.ndarray, mute_ratio: float = 0.08,
              fade_ratio: float = 0.03) -> np.ndarray:
    """压制顶部直达波：先清零，再线性淡入。"""
    if mute_ratio <= 0:
        return data
    n = data.shape[0]
    mute_n = max(0, min(n, int(round(n * mute_ratio))))
    out = data.copy()
    out[:mute_n, :] = 0.0
    fade_n = int(round(n * max(0.0, fade_ratio)))
    if fade_n > 0 and mute_n < n:
        end = min(n, mute_n + fade_n)
        ramp = np.linspace(0.0, 1.0, end - mute_n, endpoint=False, dtype=np.float32)
        out[mute_n:end, :] *= ramp[:, None]
    return out


def _time_gain(data: np.ndarray, alpha: float = 1.8) -> np.ndarray:
    """指数时间增益，补偿深部能量衰减。"""
    if alpha <= 0:
        return data
    t = np.linspace(0.0, 1.0, data.shape[0], dtype=np.float32)
    return data * np.exp(alpha * t)[:, None]


def _to_uint8(data: np.ndarray, clip_pct: float = 99.5) -> np.ndarray:
    """百分位数裁剪后映射到 [0, 255] uint8。"""
    clip_val = np.percentile(np.abs(data), clip_pct)
    if clip_val < 1e-12:
        clip_val = 1e-12
    data = np.clip(data / clip_val, -1.0, 1.0)
    return ((data + 1.0) * 127.5).astype(np.uint8)


# ── 读取 .out 文件 ────────────────────────────────────────────────────────────

def _read_trace(path: Path, component: str = "Ez") -> np.ndarray:
    with h5py.File(path, "r") as f:
        rx = f["rxs"]["rx1"]
        key = component if component in rx else list(rx.keys())[0]
        return np.asarray(rx[key][:], dtype=np.float32)


def _collect_traces(stem: str, component: str = "Ez") -> np.ndarray | None:
    """
    找到 stem1.out … stemN.out，按数字序号排序后拼成 (n_time, n_traces) 数组。
    """
    files = list(SCRIPT_DIR.glob(f"{stem}*.out"))
    numbered = []
    for f in files:
        suffix = f.stem[len(stem):]
        if re.fullmatch(r"\d+", suffix):
            numbered.append((int(suffix), f))

    if not numbered:
        print(f"  [错误] 未找到 {stem}*.out 文件")
        return None

    numbered.sort()
    print(f"  找到 {len(numbered)} 道 A-scan")

    traces = [_read_trace(f, component) for _, f in numbered]
    min_len = min(len(t) for t in traces)
    return np.column_stack([t[:min_len] for t in traces])   # (n_time, n_traces)


# ── 主处理函数 ────────────────────────────────────────────────────────────────

def process(stem: str,
            resize: tuple[int, int] | None = (256, 256),
            dewow_window: int = 9,
            mute_ratio: float = 0.08,
            fade_ratio: float = 0.03,
            gain_alpha: float = 1.8,
            clip_pct: float = 99.5,
            component: str = "Ez") -> bool:

    data = _collect_traces(stem, component)
    if data is None:
        return False

    print(f"  原始 B-scan shape: {data.shape}")

    data = _dewow(data, window=dewow_window)
    data = _background_remove(data)
    data = _mute_top(data, mute_ratio=mute_ratio, fade_ratio=fade_ratio)
    data = _time_gain(data, alpha=gain_alpha)
    img_u8 = _to_uint8(data, clip_pct=clip_pct)

    img = Image.fromarray(img_u8, mode="L")
    if resize is not None:
        img = img.resize(resize, Image.Resampling.LANCZOS)

    out_path = PREVIEW_DIR / f"{stem}_bscan.png"
    img.save(out_path)
    print(f"  [保存] {out_path}  size={img.size}")
    return True


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="gprMax A-scan → 灰度 B-scan PNG")
    parser.add_argument("stems", nargs="*",
                        help="输出文件 stem，如 bag_small_flat_d020_x25")
    parser.add_argument("--all", action="store_true", help="处理目录下所有场景")
    parser.add_argument("--resize", nargs=2, type=int, default=[256, 256],
                        metavar=("W", "H"), help="输出尺寸（默认 256 256）")
    parser.add_argument("--no-resize", action="store_true", help="保持原始分辨率")
    parser.add_argument("--gain-alpha", type=float, default=1.8)
    parser.add_argument("--clip-pct",   type=float, default=99.5)
    parser.add_argument("--mute-ratio", type=float, default=0.08)
    parser.add_argument("--component",  type=str,   default="Ez")
    args = parser.parse_args()

    resize = None if args.no_resize else tuple(args.resize)

    if args.all:
        all_out = list(SCRIPT_DIR.glob("*.out"))
        stems = sorted({
            f.stem[:re.search(r"\d+$", f.stem).start()]
            for f in all_out
            if re.search(r"\d+$", f.stem)
        })
        print(f"找到 {len(stems)} 个场景")
    elif args.stems:
        stems = args.stems
    else:
        parser.print_help()
        return

    ok = 0
    for s in stems:
        print(f"\n处理: {s}")
        if process(s, resize=resize, gain_alpha=args.gain_alpha,
                   clip_pct=args.clip_pct, mute_ratio=args.mute_ratio,
                   component=args.component):
            ok += 1

    print(f"\n完成 {ok}/{len(stems)}  →  {PREVIEW_DIR}")


if __name__ == "__main__":
    main()
