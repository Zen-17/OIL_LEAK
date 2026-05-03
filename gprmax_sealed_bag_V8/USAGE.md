# gprmax_sealed_bag_V8 使用说明

## 项目概述

基于 GPRMax 的油类泄漏 B-scan 图像数据集生成流水线，用于训练二分类模型（有泄漏 / 无泄漏）。

**环境要求**
```
conda activate gprMax
```

---

## 脚本一览

| 脚本 | 用途 |
|------|------|
| `generate_cases.py` | 生成仿真配置文件 (.in) |
| `run_all.py` | **主流水线**：生成 → 仿真 → 出图 → labels.csv |
| `plot_bscan.py` | 单独处理 .out 文件 → B-scan PNG |
| `run_pipeline.py` | 旧版流水线（仅支持旧命名格式，不推荐） |
| `process_all.py` | 旧版批量后处理（仅支持旧命名格式，不推荐） |

---

## 一、generate_cases.py — 生成仿真配置文件

### 功能

生成 1704 个 GPRMax `.in` 仿真配置文件，覆盖两类场景：

- **leak（有泄漏，label=1）**：PE 袋装食用油埋入土壤，共 1152 个
  - 4 种袋型 × 6 种土壤 × 8 种深度 × 6 种横向位置
- **noleak（无泄漏，label=0）**：不含油，共 552 个
  - `bare`（纯土壤，12 种介电常数变体）× 12
  - `rock`（石块杂波，产生双曲线但非油）× 360
  - `pipe`（空 PVC 管杂波）× 180

### 文件命名规则

```
leak_{bag_type}_{soil_key}_d{depth_cm:03d}_x{cx_cm:02d}.in
noleak_bare_{soil_key}.in
noleak_rock{size}_{soil_key}_d{depth_cm:03d}_x{cx_cm:02d}.in
noleak_pipe_{soil_key}_d{depth_cm:03d}_x{cx_cm:02d}.in
```

### 命令

```bash
# 生成全部 408 个 .in 文件（写入当前目录）
python generate_cases.py

# 预览文件列表和数量，不写入文件
python generate_cases.py --list
```

---

## 二、run_all.py — 主流水线

### 功能

三步全流程：

1. **[1/3] 生成**：调用 `generate_cases.py` 生成全部 `.in` 文件
2. **[2/3] 仿真**：对每个 `.in` 文件调用 `python -m gprMax`，每个场景扫描 21 道
3. **[3/3] 出图**：读取 `.out` 文件，经信号处理后保存为 256×256 灰度 PNG，并写入 `labels.csv`

### 输出

```
bscan_preview/{stem}_bscan.png    # 256×256 灰度 B-scan 图像
labels.csv                        # 字段：filename, label, bag_type, soil, depth_m, cx_m, clutter
```

### 命令

```bash
# ── 完整流程 ──────────────────────────────────────────────
# 全流程：生成 .in → 仿真（CPU）→ 出图
python run_all.py

# 全流程：生成 .in → 仿真（GPU 0）→ 出图（需要 CUDA Toolkit + pycuda）
python run_all.py --gpu 0

# ── 跳过阶段 ──────────────────────────────────────────────
# 跳过生成，用已有 .in 文件直接仿真 + 出图
python run_all.py --no-gen

# 跳过生成 + 仿真，仅对已有 .out 文件出图并写 labels.csv
python run_all.py --no-gen --no-sim

# 跳过仿真，用已有 .out 文件出图（适合重新调整图像处理参数）
python run_all.py --no-sim

# ── 按类别过滤 ────────────────────────────────────────────
# 只处理 leak_*.in 文件（有泄漏场景）
python run_all.py --only leak

# 只处理 noleak_*.in 文件（无泄漏场景）
python run_all.py --only noleak

# ── 组合用法 ──────────────────────────────────────────────
# 跳过生成，GPU 仿真全部文件
python run_all.py --no-gen --gpu 0

# 只对 leak 场景重新出图（不重新仿真）
python run_all.py --only leak --no-gen --no-sim

# 保留 .out 文件不删除（调试用）
python run_all.py --no-gen --keep-out
```

### 选项说明

| 选项 | 说明 |
|------|------|
| `--no-gen` | 跳过第 1 步（不重新生成 .in 文件） |
| `--no-sim` | 跳过第 2 步（不重新仿真，直接用已有 .out 文件） |
| `--only leak` | 只处理 `leak_*.in` 文件 |
| `--only noleak` | 只处理 `noleak_*.in` 文件 |
| `--gpu <ID>` | 使用指定 GPU 仿真（需 CUDA Toolkit 已安装，`nvcc` 在 PATH 中） |
| `--keep-out` | 出图后**保留** .out 文件（默认：出图成功后自动删除） |

---

## 三、plot_bscan.py — 单独出图

### 功能

读取 GPRMax `.out` 文件（HDF5 格式），经以下信号处理后输出 PNG：

1. Dewow（移动平均去趋势）
2. 背景去除（减去各时刻所有道均值）
3. 顶部静音（压制直达波，前 8% 时间窗清零）
4. 时间增益（指数增益补偿深度衰减）
5. 百分位数裁剪 → uint8 映射
6. Lanczos 缩放至目标尺寸，保存灰度 PNG

### 命令

```bash
# 处理单个场景（自动找 stem1.out … stem21.out）
python plot_bscan.py leak_small_flat_soil_dry_d010_x15

# 指定输出尺寸（默认 256×256）
python plot_bscan.py leak_small_flat_soil_dry_d010_x15 --resize 128 128

# 保持原始分辨率，不缩放
python plot_bscan.py leak_small_flat_soil_dry_d010_x15 --no-resize

# 处理目录下所有场景（自动识别所有 stem）
python plot_bscan.py --all

# 调整信号处理参数
python plot_bscan.py --all --gain-alpha 2.0 --clip-pct 99.0 --mute-ratio 0.05
```

### 选项说明

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--resize W H` | `256 256` | 输出图像尺寸 |
| `--no-resize` | — | 保持原始时间×道数分辨率 |
| `--gain-alpha` | `1.8` | 时间增益指数（越大深部越亮） |
| `--clip-pct` | `99.5` | 幅度裁剪百分位数 |
| `--mute-ratio` | `0.08` | 顶部静音比例（0~1） |
| `--component` | `Ez` | 读取的场分量（通常无需修改） |

---

## 四、典型工作流

### ▶ 立即执行（当前推荐）

.in 文件已生成（1704 个，天线贴地已修正），直接跳过生成步骤开始仿真：

```bash
conda activate gprMax
cd D:\GPRMax\gprmax_sealed_bag_V8

# CPU 仿真（无需额外依赖）
python run_all.py --no-gen

# GPU 仿真（需已安装 CUDA Toolkit 12.x，nvcc 在 PATH 中）
python run_all.py --no-gen --gpu 0
```

执行流程：408 个场景逐一处理，每个场景完成后立即删除 .out 文件。
预计耗时：1704 个场景，CPU 约十数小时；GPU（RTX 4070，安装 CUDA Toolkit 后）约 4～6 小时。

---

### 全流程重新生成（含重新生成 .in 文件）

```bash
conda activate gprMax
cd D:\GPRMax\gprmax_sealed_bag_V8
python run_all.py          # CPU
python run_all.py --gpu 0  # GPU
```

### 仅重新出图（已有 .out 文件）

```bash
python run_all.py --no-gen --no-sim
```

### 单场景调试

```bash
# 手动仿真一个文件（21 道）
conda activate gprMax
python -m gprMax noleak_bare_s10.in -n 21

# 出图查看效果
python plot_bscan.py noleak_bare_s10
```

---

## 五、输出目录结构

```
gprmax_sealed_bag_V8/
├── *.in                        # 408 个仿真配置文件
├── *1.out ~ *21.out            # 仿真输出（HDF5，每场景 21 个文件）
├── bscan_preview/
│   └── {stem}_bscan.png        # 256×256 灰度 B-scan 图像
└── labels.csv                  # 数据集标签文件
```

### labels.csv 字段

| 字段 | 说明 |
|------|------|
| `filename` | 图像相对路径，如 `bscan_preview/leak_..._bscan.png` |
| `label` | `1` = 有泄漏，`0` = 无泄漏 |
| `bag_type` | 袋型（`small_flat` / `small_vert` / `large_flat` / `large_vert`） |
| `soil` | 土壤类型（`soil_dry` / `soil_med` / `soil_wet` / `s03`…`s25`） |
| `depth_m` | 埋深（米） |
| `cx_m` | 横向中心位置（米） |
| `clutter` | 杂波类型（`bare` / `rock_sm` / `rock_md` / `pipe` / 空字符串） |

---

## 六、仿真参数说明

| 参数 | 值 | 说明 |
|------|----|------|
| 域尺寸 | 0.6 × 1.533 × 0.003 m | 2D 模式（Z 方向单格） |
| 格间距 | 3 mm | DX = DY = DZ = 0.003 m |
| 时间窗 | 30 ns | 对应最大探测深度约 1.5 m |
| 中心频率 | 1.61 GHz | Gaussian 脉冲 |
| 天线高度 | y = 1.497 m | 贴地（最后一个土壤格，soil top = 1.500 m） |
| 天线偏移 | 0.1 m | 发射-接收间距 |
| 扫描道数 | 21 道 | 步进 0.02 m，覆盖 x = 0.05 ~ 0.45 m |
| PML | 10 格 | 吸收边界 |
