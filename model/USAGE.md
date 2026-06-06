# model/ 使用说明

## 项目概述

基于 EfficientNet-B0 的 GPR B-scan 双头分类器，用于阶段一异常检测：

- **Head A**：输出异常概率 P ∈ [0, 1]（有无油类泄漏）
- **Head B**：输出 5 类目标类型（油类 / 金属管线 / 非金属管 / 石块 / 背景）

**环境要求**
```
conda activate torch_gpu
```

---

## 脚本一览

| 脚本 | 用途 |
|------|------|
| `efficientnet_dual.py` | 模型架构定义 + 损失函数 |
| `train.py` | 训练脚本（两阶段迁移学习） |
| `export_onnx.py` | 导出 ONNX 格式（部署前置步骤） |

---

## 前置条件

### 1. 生成训练数据

训练脚本默认从 `gprmax_sealed_bag_V8/labels.csv` 加载数据。
若该文件不存在，先完成仿真流水线：

```bash
conda activate gprMax
cd D:\GPRMax\gprmax_sealed_bag_V8
python run_all.py --no-gen       # CPU 仿真
# 或
python run_all.py --no-gen --gpu 0  # GPU 仿真
```

### 2. 确认图像已生成

```
gprmax_sealed_bag_V8/
├── labels.csv              ← 训练标签（1704 行）
└── bscan_preview/
    └── *.png               ← 256×256 灰度 B-scan 图像（1704 张）
```

---

## 一、efficientnet_dual.py — 模型架构

### 网络结构

```
输入 (B, 3, 224, 224)
        ↓
EfficientNet-B0 主干（ImageNet 预训练）
        ↓
GlobalAvgPool → Dropout → 特征向量 (B, 1280)
        ↓              ↓
    头 A               头 B
Linear(1280, 1)   Linear(1280, 5)
   + sigmoid          (logits)
        ↓              ↓
  P(异常) ∈[0,1]   [油, 金属管, 非金属管, 石块, 背景]
```

### 损失函数

```
总损失 = 0.5 × BCELoss(头A) + 0.5 × CrossEntropyLoss(头B)
```

### 标签来源

标签从 `labels.csv` 的两列读取，互相独立：

| labels.csv 字段 | 值 | 对应标签 |
|---|---|---|
| `label` | `1` | Head A = 1（有泄漏） |
| `label` | `0` | Head A = 0（无泄漏） |
| `clutter` = `""` | leak 场景 | Head B = 0（oil） |
| `clutter` = `bare` | 纯土壤 | Head B = 4（background） |
| `clutter` = `rock_sm/md` | 石块 | Head B = 3（rock） |
| `clutter` = `pipe` | 非金属管 | Head B = 2（nonmetal_pipe） |

---

## 二、train.py — 训练

### 两阶段训练策略

**Phase 1（冻结主干）**：冻结 EfficientNet-B0 前 75% 的层，仅训练后 1/4 层和双头，快速收敛。

**Phase 2（全量微调）**：解冻所有层，以较小学习率精调，适应 GPR 域特征。

### 数据增强

| 方法 | 参数 |
|------|------|
| 随机水平翻转 | p=0.5（目标左右对称，物理上合法） |
| 亮度/对比度扰动 | ±20% |
| 随机裁剪 resize | scale=0.85~1.0 → 224×224 |
| 高斯噪声 | std=0.02（模拟真实采集噪声） |

### 命令

```bash
conda activate torch_gpu
cd D:\GPRMax

# 默认参数（推荐首次运行）
python model/train.py

# 指定数据集路径
python model/train.py --data gprmax_sealed_bag_V8/labels.csv

# 调整训练轮数和学习率
python model/train.py --epochs-frozen 10 --epochs-full 20 --lr 1e-3 --lr-full 2e-4

# 不使用预训练权重（离线机器）
python model/train.py --no-pretrain

# 自定义 batch size 和 dataloader 工作进程数
python model/train.py --batch 32 --workers 4
```

### 选项说明

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--data` | `gprmax_sealed_bag_V8/labels.csv` | 标签文件路径 |
| `--epochs-frozen` | `10` | Phase 1 训练轮数（冻结主干） |
| `--epochs-full` | `20` | Phase 2 训练轮数（全量微调） |
| `--batch` | `32` | Batch size |
| `--lr` | `1e-3` | Phase 1 学习率 |
| `--lr-full` | `2e-4` | Phase 2 学习率 |
| `--val-ratio` | `0.15` | 验证集比例（15%） |
| `--workers` | `4` | DataLoader 工作进程数 |
| `--no-pretrain` | — | 跳过 ImageNet 预训练权重 |

### 输出文件

```
model/
├── best_phase1.pth        # Phase 1 最优权重（按验证集 Head-A 准确率保存）
└── efficientnet_dual.pth  # Phase 2 最优权重（最终使用此文件）
```

### 训练日志示例

```
Device : cuda
Dataset: 1704 samples
  Head-A  noleak=552  leak=1152
  Head-B  [0] oil              1152
  Head-B  [2] nonmetal_pipe     180
  Head-B  [3] rock              360
  Head-B  [4] background         12

Train: 1448   Val: 256

============================================================
Phase 1  frozen backbone  10 epochs  lr=0.001
  ep01  tr=0.6823/0.621  val=0.5914/A=0.734/B=0.701
  ep02  tr=0.5201/0.758  val=0.4832/A=0.812/B=0.779  ✓ saved
  ...

============================================================
Phase 2  full fine-tune   20 epochs  lr=0.0002
  ep01  tr=0.3847/0.861  val=0.3201/A=0.887/B=0.854  ✓ saved
  ...
Best val Head-A accuracy: 0.923
```

---

## 三、export_onnx.py — 导出 ONNX

### 功能

将训练好的 `.pth` 权重导出为 ONNX 格式，用于后续 RKNN 模型转换。

导出参数：
- 输入：`input`，shape `(1, 3, 224, 224)`，固定 batch=1
- 输出：`prob_a` shape `(1,)`，`logits_b` shape `(1, 5)`
- opset version：12（RKNN-Toolkit2 兼容）

### 命令

```bash
conda activate torch_gpu
cd D:\GPRMax

# 使用默认路径（model/efficientnet_dual.pth → model/efficientnet_dual.onnx）
python model/export_onnx.py

# 指定权重和输出路径
python model/export_onnx.py --weights model/efficientnet_dual.pth --output model/efficientnet_dual.onnx

# 验证导出结果
python -c "import onnx; m=onnx.load('model/efficientnet_dual.onnx'); onnx.checker.check_model(m); print('ONNX OK')"
```

### 选项说明

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--weights` | `model/efficientnet_dual.pth` | 输入权重文件 |
| `--output` | `model/efficientnet_dual.onnx` | 输出 ONNX 文件路径 |
| `--opset` | `12` | ONNX opset 版本 |

---

## 四、典型工作流

### ▶ 完整训练流程

```bash
# 1. 确认仿真数据已就绪
conda activate gprMax
cd D:\GPRMax\gprmax_sealed_bag_V8
python run_all.py --no-gen          # 生成 bscan_preview/ 和 labels.csv

# 2. 训练模型
conda activate torch_gpu
cd D:\GPRMax
python model/train.py

# 3. 导出 ONNX
python model/export_onnx.py
```

### 仅重新训练（数据已就绪）

```bash
conda activate torch_gpu
cd D:\GPRMax
python model/train.py
```

### 调整超参数重新训练

```bash
# 增加轮数、减小 batch（显存不足时）
python model/train.py --epochs-frozen 15 --epochs-full 30 --batch 16

# 验证集准确率不足时，降低学习率
python model/train.py --lr 5e-4 --lr-full 1e-4
```

---

## 五、输出目录结构

```
model/
├── efficientnet_dual.py   # 模型架构
├── train.py               # 训练脚本
├── export_onnx.py         # ONNX 导出
├── USAGE.md               # 本文件
├── best_phase1.pth        # Phase 1 最优权重（训练后生成）
├── efficientnet_dual.pth  # Phase 2 最优权重（训练后生成）
└── efficientnet_dual.onnx # ONNX 模型（导出后生成）
```

---

## 六、模型参数说明

| 参数 | 值 | 说明 |
|------|----|------|
| 主干网络 | EfficientNet-B0 | torchvision 内置，无需额外依赖 |
| 预训练权重 | ImageNet1K | `EfficientNet_B0_Weights.IMAGENET1K_V1` |
| 特征维度 | 1280 | GlobalAvgPool 输出 |
| 输入尺寸 | 224 × 224 | 训练和推理均使用此尺寸 |
| Head A | Linear(1280, 1) + sigmoid | 异常二分类 |
| Head B | Linear(1280, 5) | 目标类型 5 分类 |
| 触发阈值 | P > 0.65 | 超过此值触发阶段二验证 |
| 高置信阈值 | P > 0.85 | 配合空间一致性判为红色告警 |
