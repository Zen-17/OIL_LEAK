"""
批量后处理所有 GPRMax 仿真场景，生成 B-scan 训练图像。
输出：
  - training_images/sealed_bag/*.png  (224×224 训练图)
  - bscan_preview/*_preview.png       (原始尺寸预览图)
  - labels.csv                         (标签文件)
"""

import os
import glob
import csv
import re
import sys

import numpy as np
import h5py
import cv2

# ── 路径配置 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
IMG_DIR       = os.path.join(SCRIPT_DIR, "training_images", "sealed_bag")
PREVIEW_DIR   = os.path.join(SCRIPT_DIR, "bscan_preview")
LABELS_CSV    = os.path.join(SCRIPT_DIR, "labels.csv")

os.makedirs(IMG_DIR,     exist_ok=True)
os.makedirs(PREVIEW_DIR, exist_ok=True)

N_SCANS = 21  # 每个场景的扫描道数

# ── 袋子类型元数据 ────────────────────────────────────────────────────────────
BAG_META = {
    "small_flat": {"bag_x": 0.17,  "bag_z": 0.025, "orientation": "flat"},
    "small_vert": {"bag_x": 0.025, "bag_z": 0.17,  "orientation": "vert"},
    "large_flat": {"bag_x": 0.22,  "bag_z": 0.025, "orientation": "flat"},
    "large_vert": {"bag_x": 0.025, "bag_z": 0.15,  "orientation": "vert"},
}

# ── 文件名解析 ────────────────────────────────────────────────────────────────
def parse_bag_filename(stem):
    """
    从文件名（不含扩展名）解析场景参数。
    格式：bag_{type}_d{depth_cm}_x{cx_cm}
    返回 dict 或 None（baseline 场景）。
    """
    if stem == "baseline":
        return None
    m = re.match(
        r"bag_(small_flat|small_vert|large_flat|large_vert)_d(\d+)_x(\d+)",
        stem
    )
    if not m:
        return None
    bag_type  = m.group(1)
    depth_cm  = int(m.group(2))
    cx_cm     = int(m.group(3))
    meta      = BAG_META[bag_type]
    return {
        "bag_type":    bag_type,
        "depth_cm":    depth_cm,
        "cx_cm":       cx_cm,
        "bag_x":       meta["bag_x"],
        "bag_z":       meta["bag_z"],
        "orientation": meta["orientation"],
    }

# ── HDF5 读取 ─────────────────────────────────────────────────────────────────
def load_bscan_from_hdf5(stem):
    """
    自动匹配 stem 对应的所有 HDF5 输出文件，拼成 B-scan 矩阵。
    GPRMax 每个 -n 运行生成 stem1.out, stem2.out, ... stemN.out（HDF5 格式）。
    返回 shape (n_time, n_scans) 的 numpy 数组，失败返回 None。
    """
    pattern = os.path.join(SCRIPT_DIR, f"{stem}*.out")
    files   = sorted(glob.glob(pattern))

    if not files:
        return None

    traces = []
    for fpath in files:
        try:
            with h5py.File(fpath, "r") as hf:
                # GPRMax 3.x 输出路径: /rxs/rx1/Ez 或 /rxs/rx1/{field}
                rx_group = hf["rxs"]["rx1"]
                # 优先取 Ez（TM 极化）
                field_key = "Ez" if "Ez" in rx_group else list(rx_group.keys())[0]
                trace = rx_group[field_key][:]  # shape: (n_time,)
                traces.append(trace)
        except Exception as e:
            print(f"  [警告] 读取 {fpath} 失败: {e}")

    if not traces:
        return None

    # 截断到最短道（防止时间步不一致）
    min_len = min(len(t) for t in traces)
    traces  = [t[:min_len] for t in traces]

    bscan = np.column_stack(traces)  # (n_time, n_scans)
    return bscan

# ── 预处理 ────────────────────────────────────────────────────────────────────
def preprocess_bscan(bscan):
    """
    背景去除 + 列归一化。
    返回值域 [-1, 1] 的 float32 数组，shape 不变。
    """
    # 背景去除：减去所有道均值
    mean_trace = np.mean(bscan, axis=1, keepdims=True)
    bscan      = bscan - mean_trace

    # 列归一化：每列除以该列最大绝对值
    col_max = np.max(np.abs(bscan), axis=0, keepdims=True)
    col_max[col_max == 0] = 1.0   # 防止除零
    bscan = bscan / col_max

    return bscan.astype(np.float32)

# ── 图像保存 ──────────────────────────────────────────────────────────────────
def save_training_image(bscan_norm, out_path):
    """将 [-1,1] B-scan 缩放为 0~255 uint8，resize 到 224×224，保存 PNG。"""
    img_float = (bscan_norm + 1.0) / 2.0 * 255.0
    img_u8    = np.clip(img_float, 0, 255).astype(np.uint8)
    img_224   = cv2.resize(img_u8, (224, 224), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(out_path, img_224)

def save_preview_image(bscan_norm, out_path):
    """保存原始尺寸预览图（灰度）。"""
    img_float = (bscan_norm + 1.0) / 2.0 * 255.0
    img_u8    = np.clip(img_float, 0, 255).astype(np.uint8)
    cv2.imwrite(out_path, img_u8)

# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    # 收集所有场景：bag_*.in + baseline.in
    in_files = sorted(glob.glob(os.path.join(SCRIPT_DIR, "bag_*.in")))
    in_files += glob.glob(os.path.join(SCRIPT_DIR, "baseline.in"))

    rows      = []   # labels.csv 行
    success   = 0
    skipped   = 0

    for in_path in in_files:
        stem = os.path.splitext(os.path.basename(in_path))[0]
        info = parse_bag_filename(stem)

        print(f"\n处理: {stem}")

        bscan = load_bscan_from_hdf5(stem)
        if bscan is None:
            print(f"  [跳过] 未找到 HDF5 输出文件: {stem}*.out")
            skipped += 1
            continue

        # 检查道数
        if bscan.shape[1] != N_SCANS:
            print(f"  [警告] 期望 {N_SCANS} 道，实际 {bscan.shape[1]} 道，继续处理")

        bscan_norm = preprocess_bscan(bscan)

        # 保存训练图
        img_name    = stem + ".png"
        img_path    = os.path.join(IMG_DIR, img_name)
        save_training_image(bscan_norm, img_path)

        # 保存预览图
        prev_path = os.path.join(PREVIEW_DIR, stem + "_preview.png")
        save_preview_image(bscan_norm, prev_path)

        # 构造标签行
        if info is None:
            # baseline
            row = {
                "filename":    img_name,
                "label":       0,
                "bag_type":    "none",
                "depth_cm":    0,
                "cx_cm":       0,
                "bag_x_m":     0,
                "bag_z_m":     0,
                "orientation": "none",
            }
        else:
            row = {
                "filename":    img_name,
                "label":       1,
                "bag_type":    info["bag_type"],
                "depth_cm":    info["depth_cm"],
                "cx_cm":       info["cx_cm"],
                "bag_x_m":     info["bag_x"],
                "bag_z_m":     info["bag_z"],
                "orientation": info["orientation"],
            }

        rows.append(row)
        success += 1
        print(f"  [完成] {img_name}  shape={bscan.shape}")

    # 写 labels.csv
    fieldnames = ["filename", "label", "bag_type", "depth_cm",
                  "cx_cm", "bag_x_m", "bag_z_m", "orientation"]
    with open(LABELS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 50)
    print(f"处理完成！")
    print(f"  成功处理: {success} 个场景")
    print(f"  跳过:     {skipped} 个场景（缺少 HDF5 输出）")
    print(f"  训练图像: {IMG_DIR}")
    print(f"  预览图像: {PREVIEW_DIR}")
    print(f"  标签文件: {LABELS_CSV}")


if __name__ == "__main__":
    main()
