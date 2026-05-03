# 🛢️ GPR-OilScan: 基于探地雷达B-scan图像的浅层油类泄漏深度学习检测系统

> **平台**：瑞芯微 RK3568 (1 TOPS NPU) | **方法**：GPRmax仿真 + 深度学习分类 | **目标**：浅层地表油类泄漏识别

---

## 📋 目录

1. [项目概述](#1-项目概述)
2. [系统架构](#2-系统架构)
3. [技术原理](#3-技术原理)
4. [环境配置](#4-环境配置)
5. [GPRmax仿真数据生成](#5-gprmax仿真数据生成)
6. [数据预处理流程](#6-数据预处理流程)
7. [模型设计与选型](#7-模型设计与选型)
8. [训练策略](#8-训练策略)
9. [模型压缩与量化](#9-模型压缩与量化)
10. [RK3568部署](#10-rk3568部署)
11. [真实数据补充与迁移学习](#11-真实数据补充与迁移学习)
12. [性能评估指标](#12-性能评估指标)
13. [项目目录结构](#13-项目目录结构)
14. [已知挑战与解决方案](#14-已知挑战与解决方案)
15. [参考文献](#15-参考文献)

---

## 1. 项目概述

### 1.1 背景与动机

地下储油罐、输油管道的油类泄漏是严重的环境安全隐患。传统检测方法（钻孔取样、化学分析）存在破坏性强、成本高、响应慢等缺陷。探地雷达（Ground Penetrating Radar, GPR）作为一种无损探测技术，通过发射高频电磁波并接收反射信号，能够在不开挖的情况下对浅层地表介质进行成像。

油类物质（汽油、柴油、原油等）与正常土壤介质的**介电常数（Relative Permittivity）**存在显著差异：

| 介质 | 相对介电常数 εᵣ | 电导率 σ (mS/m) |
|------|----------------|-----------------|
| 干燥砂土 | 3–6 | 0.01–1 |
| 湿润砂土 | 10–30 | 0.1–10 |
| 粘土 | 5–40 | 2–1000 |
| 汽油/柴油 | **1.8–2.2** | <0.01 |
| 原油 | **2.0–2.5** | 0.01–0.1 |
| 含油污染土壤 | **4–8**（取决于含量）| 变化显著 |
| 淡水 | 81 | 0.5 |

油类物质的低介电常数特性会在B-scan图像中产生特征性的**双曲线反射异常**，但由于油层往往以弥散态分布于土壤孔隙中，其反射特征比管道、空洞等目标更为复杂，这正是本项目的核心挑战。

### 1.2 项目目标

- **主要目标**：通过分析GPR B-scan灰度图像，二分类判断是否存在油类泄漏（`oil_detected` / `no_oil`）
- **部署目标**：在瑞芯微RK3568（1 TOPS NPU）上实现实时推理（目标 ≥ 5 FPS）
- **精度目标**：综合F1-Score ≥ 85%（最终含真实数据微调后 ≥ 90%）

### 1.3 技术路线总览

```
GPRmax仿真 ──────────────────────────────┐
  ├─ 多场景参数配置                         │
  ├─ 批量B-scan生成                        │
  └─ 仿真数据集构建                         │
                                          ▼
                                    数据预处理
                                      ├─ 背景去除
                                      ├─ 增益校正
                                      ├─ 归一化
                                      └─ 数据增强
                                          │
                                          ▼
                                   模型训练（PC/GPU）
                                      ├─ 骨干网络选型
                                      ├─ 迁移学习
                                      ├─ 知识蒸馏
                                      └─ QAT量化感知训练
                                          │
                              ┌───────────┴───────────┐
                              │                       │
                         RKNN转换                 真实数据
                         INT8量化                  微调补充
                              │                       │
                              └───────────┬───────────┘
                                          ▼
                                   RK3568 NPU部署
                                      └─ 实时检测推理
```

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         离线训练阶段（x86 PC / GPU 服务器）          │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌─────────────────┐   │
│  │  GPRmax 仿真  │───▶│  数据预处理   │───▶│   模型训练/蒸馏  │   │
│  │  批量脚本生成  │    │  背景去除    │    │  MobileNetV2   │   │
│  │  .in 配置文件 │    │  增益+归一化  │    │  + 注意力模块   │   │
│  └──────────────┘    └──────────────┘    └────────┬────────┘   │
│                                                    │             │
│                                           ┌────────▼────────┐   │
│                                           │  RKNN-Toolkit2  │   │
│                                           │  模型转换+量化    │   │
│                                           │  INT8 校准      │   │
│                                           └────────┬────────┘   │
└────────────────────────────────────────────────────┼────────────┘
                                                      │ .rknn 模型文件
┌────────────────────────────────────────────────────▼────────────┐
│                         在线推理阶段（RK3568 嵌入式平台）           │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌─────────────────┐   │
│  │  GPR 硬件接口 │───▶│  实时预处理   │───▶│  RKNN Runtime   │   │
│  │  原始A-scan  │    │  B-scan合成  │    │  NPU推理引擎    │   │
│  │  数据采集    │    │  灰度图生成   │    │  1 TOPS 加速    │   │
│  └──────────────┘    └──────────────┘    └────────┬────────┘   │
│                                                    │             │
│                                           ┌────────▼────────┐   │
│                                           │   结果输出/告警   │   │
│                                           │  置信度 + 热力图  │   │
│                                           └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 技术原理

### 3.1 GPR B-scan成像原理

GPR天线沿地面移动，在每个位置发射脉冲并记录反射波形（A-scan）。将所有A-scan按时间轴拼接即得B-scan图像：

- **横轴**：天线位置（扫描距离，m）
- **纵轴**：双程走时（ns），可换算为深度
- **亮度**：反射信号强度

点目标（管道截面、油团聚集体）在B-scan中呈现**双曲线**特征。弥散性油污染层则可能表现为：
- 层状强反射界面（油水界面）
- 异常高程反射（低εᵣ导致传播速度加快）
- 绕射双曲线群（孤立油团）

### 3.2 油类介质对电磁波的影响

```
电磁波速度: v = c / √εᵣ

干燥砂土 (εᵣ=5):  v ≈ 0.134 m/ns
油污染土 (εᵣ=3):  v ≈ 0.173 m/ns  ← 速度更快，反射层深度偏浅
含水土壤 (εᵣ=25): v ≈ 0.060 m/ns

反射系数: R = (√εᵣ₁ - √εᵣ₂) / (√εᵣ₁ + √εᵣ₂)
```

油类低介电常数导致：
1. 油层顶界面产生**负极性反射**（速度增大）
2. 油层底界面产生**正极性反射**
3. 相对于干燥土壤，油污染区域电磁波传播速度反而更快

### 3.3 为什么选择B-scan灰度图像

| 输入形式 | 优点 | 缺点 |
|---------|------|------|
| 原始时域A-scan信号 | 信息完整 | 序列处理复杂，对齐困难 |
| 频域特征 | 区分介质类型 | 需要专业信号处理知识 |
| **B-scan灰度图像** ✅ | 空间纹理特征丰富，CNN天然适用，可视化直观 | 需要合理的增益处理 |
| RGB伪彩色图像 | 视觉对比度高 | 颜色映射引入主观偏差 |

---

## 4. 环境配置

### 4.1 训练环境（PC端）

```bash
# Python 环境
Python >= 3.8
CUDA >= 11.3（GPU训练）
cuDNN >= 8.2

# 核心依赖
pip install torch==2.0.1 torchvision==0.15.2
pip install numpy scipy matplotlib h5py
pip install scikit-learn scikit-image pillow
pip install tensorboard
pip install rknn-toolkit2  # 用于模型转换

# GPRmax
pip install gprMax  # 或从源码安装
# conda install -c conda-forge gprmax
```

### 4.2 部署环境（RK3568端）

```bash
# 板端系统：Ubuntu 20.04 / Debian 11 (aarch64)
# Python >= 3.8

# RKNN Runtime（板端推理）
pip install rknn-toolkit-lite2

# 依赖库
pip install numpy opencv-python-headless pillow
```

### 4.3 推荐硬件配置

| 组件 | 规格 |
|------|------|
| 开发机 GPU | NVIDIA RTX 3060 或以上 |
| RK3568 内存 | 4 GB LPDDR4 或以上 |
| 存储 | 32 GB eMMC + TF卡（数据集） |
| GPR天线 | 500 MHz 或 1 GHz 屏蔽天线 |

---

## 5. GPRmax仿真数据生成

### 5.1 仿真设计哲学：最大化Sim-to-Real相似度

仿真数据与真实数据之间存在**域偏移（Domain Gap）**，这是GPR深度学习最大的挑战之一。为最小化此偏移，需要在仿真中精确模拟以下因素：

```
真实场景复杂性
    ├── 土壤异质性（fractal_box + soil_peplinski）
    ├── 地表粗糙度（#rough_surface）
    ├── 天线离地高度变化（±1–3 cm抖动）
    ├── 油污染的几何形态多样性（层状/团状/弥散）
    ├── 土壤含水量变化（影响背景εᵣ）
    ├── 环境噪声（高斯噪声叠加）
    └── 多层介质结构（沥青/素土/砂砾层）
```

### 5.2 物理参数设置

#### 5.2.1 核心材料参数

```python
# 材料介电参数（参考文献值）
MATERIALS = {
    # 背景土壤类型（随机选择一种）
    'dry_sand':        {'er': 4.0,  'sigma': 0.001, 'tag': 'background'},
    'moist_sand':      {'er': 15.0, 'sigma': 0.01,  'tag': 'background'},
    'clay':            {'er': 25.0, 'sigma': 0.05,  'tag': 'background'},
    'loam':            {'er': 12.0, 'sigma': 0.02,  'tag': 'background'},
    'gravel':          {'er': 6.0,  'sigma': 0.005, 'tag': 'background'},

    # 油类目标
    'gasoline':        {'er': 2.0,  'sigma': 0.001, 'tag': 'oil'},
    'diesel':          {'er': 2.2,  'sigma': 0.001, 'tag': 'oil'},
    'crude_oil':       {'er': 2.5,  'sigma': 0.005, 'tag': 'oil'},
    'oil_sat_soil':    {'er': 5.0,  'sigma': 0.003, 'tag': 'oil'},  # 油饱和土

    # 其他地下结构（干扰项）
    'concrete':        {'er': 6.0,  'sigma': 0.01,  'tag': 'structure'},
    'pvc_pipe':        {'er': 3.0,  'sigma': 0.001, 'tag': 'structure'},
    'air_void':        {'er': 1.0,  'sigma': 0.0,   'tag': 'structure'},
}
```

#### 5.2.2 天线频率选择

```
目标深度: 0.1m – 2.0m（浅层泄漏重点关注 < 1m）

频率选择依据:
  ┌─────────────────────────────────────────────┐
  │ 分辨率（λ/4）= v / (4×f)                    │
  │                                             │
  │ 500 MHz: 分辨率 ≈ 6.7cm（εᵣ=5土壤）          │
  │          穿透深度 ≈ 3-5m                     │
  │                                             │
  │ 1 GHz:   分辨率 ≈ 3.4cm                     │
  │          穿透深度 ≈ 1-2m  ← 浅层油污推荐      │
  └─────────────────────────────────────────────┘

推荐：仿真时同时生成 500MHz 和 1GHz 两套数据
      训练时合并（需标注频率作为辅助输入或分别训练）
```

### 5.3 GPRmax输入文件模板

#### 5.3.1 无油类污染（负样本）

```
## negative_sample_template.in
## 无油污染标准土壤模型

#title: No-oil baseline - moist sandy soil

## 仿真域设置 (x, y, z 单位: m)
## 2D模式: z方向网格数为1
#domain: 1.0 0.6 0.002

## 空间步长 (满足 Δ < λmin/10 约束)
## 1GHz天线: λmin ≈ 0.03m (εᵣ=9) → Δ ≤ 0.003m
#dx_dy_dz: 0.002 0.002 0.002

## 时间窗口 (双程走时覆盖目标深度)
## 深度1m，εᵣ=15土壤: t = 2×1.0/v = 2×1.0×√15/c ≈ 25.8ns → 取30ns
#time_window: 30e-9

## 材料定义: #material: εᵣ  σ  μᵣ  σ*  名称
#material: 15.0 0.01 1.0 0.0 moist_sand

## 土壤体（使用Peplinski混合模型获得更真实的频散特性）
## #soil_peplinski: 沙粒比 粘粒比 体积密度 含水量(低) 含水量(高) 名称
#soil_peplinski: 0.5 0.15 1.7 0.05 0.25 soil_bg

## 地表以下全部填充背景土壤
#fractal_box: 0 0 0 1.0 0.58 0.002 1.5 1.0 1.0 1 50 soil_bg z

## 地表空气层（天线离地10mm）
#box: 0 0.58 0 1.0 0.60 0.002 free_space

## 激励天线（Ricker子波，1GHz中心频率）
## 天线沿x方向扫描，步长5mm，共200道 → x从0.05到1.0
#waveform: ricker 1 1e9 mypulse

## 赫兹偶极子天线（2D简化模型）
## 发射: x=0.05, y=0.591 (离地10mm), 极化方向z
#hertzian_dipole: z 0.05 0.591 0 mypulse

## 接收: 发射右侧固定偏移 20mm（天线间距）
#rx: 0.07 0.591 0

## B-scan输出：每次运行移动天线位置
#src_steps: 0.005 0 0
#rx_steps: 0.005 0 0

## 输出字段
#rx_array: 0.07 0.591 0 0.005 0 0 200

## 边界条件（完美匹配层）
#pml_cells: 10
```

#### 5.3.2 含油层污染（正样本）

```
## positive_sample_oil_layer.in
## 含水平油污染层模型

#title: Oil layer contamination - 0.3m depth

#domain: 1.0 0.6 0.002
#dx_dy_dz: 0.002 0.002 0.002
#time_window: 30e-9

## 背景土壤（含水砂土）
#soil_peplinski: 0.5 0.15 1.7 0.05 0.25 soil_bg
#fractal_box: 0 0 0 1.0 0.58 0.002 1.5 1.0 1.0 1 50 soil_bg z

## 油污染层（埋深0.28-0.38m，厚度约10cm）
## 油饱和土: εᵣ≈5.0, σ≈0.003
#material: 5.0 0.003 1.0 0.0 oil_sat_soil

## 不规则油污形状（使用多个矩形叠加模拟弥散边界）
#box: 0.1 0.22 0 0.85 0.30 0.002 oil_sat_soil
#box: 0.05 0.21 0 0.60 0.29 0.002 oil_sat_soil
#box: 0.40 0.20 0 0.90 0.28 0.002 oil_sat_soil

## 空气层
#box: 0 0.58 0 1.0 0.60 0.002 free_space

## 天线设置（同负样本）
#waveform: ricker 1 1e9 mypulse
#hertzian_dipole: z 0.05 0.591 0 mypulse
#rx: 0.07 0.591 0
#src_steps: 0.005 0 0
#rx_steps: 0.005 0 0
#rx_array: 0.07 0.591 0 0.005 0 0 200
#pml_cells: 10
```

### 5.4 批量仿真脚本

```python
# generate_dataset.py
"""
批量生成GPRmax仿真数据集
策略：系统性参数扫描 + 随机扰动 = 高覆盖率数据集
"""

import os
import subprocess
import random
import numpy as np
from itertools import product

# ─────────────────────────────────────────────────────────
# 参数空间定义（笛卡尔积 + 随机采样）
# ─────────────────────────────────────────────────────────
PARAM_SPACE = {
    # 背景土壤
    'soil_type': [
        ('dry_sand',   4.0,  0.001),
        ('moist_sand', 15.0, 0.010),
        ('clay',       25.0, 0.050),
        ('loam',       12.0, 0.020),
    ],
    # 天线频率
    'frequency': [500e6, 1e9],
    # 油层深度范围 (m)
    'oil_depth': [0.2, 0.3, 0.4, 0.5, 0.7, 1.0],
    # 油层厚度 (m)
    'oil_thickness': [0.05, 0.10, 0.20],
    # 油类型
    'oil_type': [
        ('gasoline',     2.0, 0.001),
        ('diesel',       2.2, 0.001),
        ('crude_oil',    2.5, 0.005),
        ('oil_sat_soil', 5.0, 0.003),
    ],
    # 是否包含非油类干扰目标
    'add_clutter': [True, False],
}

# 仿真数量目标
N_POSITIVE = 3000  # 含油样本
N_NEGATIVE = 3000  # 无油样本
N_RANDOM_AUGMENT = 5  # 每个基础配置随机扰动次数

def generate_input_file(config: dict, output_path: str, label: str) -> str:
    """生成gprMax输入文件"""
    soil_name, soil_er, soil_sigma = config['soil_type']
    freq = config['frequency']
    
    # 计算网格步长 (λ/10 准则)
    v_min = 3e8 / np.sqrt(max(soil_er, 25.0))  # 最慢速度对应最高εᵣ
    lambda_min = v_min / freq
    dx = min(lambda_min / 15, 0.003)  # 取较严格的约束
    dx = round(dx, 4)
    
    # 时间窗口：覆盖2m深度双程走时 + 20%余量
    t_window = 2 * 2.0 / (3e8 / np.sqrt(soil_er)) * 1.2
    
    # 随机扰动参数
    ant_height = 0.58 + random.uniform(-0.01, 0.02)  # 天线高度抖动
    noise_amp = random.uniform(0, 0.05)               # 噪声幅度
    
    lines = [
        f"#title: {label} - {soil_name} - f={freq/1e6:.0f}MHz",
        f"",
        f"#domain: 1.0 {ant_height + 0.03:.3f} 0.002",
        f"#dx_dy_dz: {dx} {dx} 0.002",
        f"#time_window: {t_window:.2e}",
        f"",
        f"## 背景土壤（Peplinski混合模型）",
        f"#soil_peplinski: 0.5 0.15 1.7 0.05 0.25 soil_bg",
        f"#fractal_box: 0 0 0 1.0 {ant_height:.3f} 0.002 1.5 1.0 1.0 1 50 soil_bg z",
    ]
    
    if label == 'oil':
        oil_name, oil_er, oil_sigma = config['oil_type']
        depth = config['oil_depth']
        thickness = config['oil_thickness']
        
        # 油层顶部y坐标 = ant_height - depth
        y_top = ant_height - depth
        y_bot = y_top - thickness
        
        # 添加随机横向不均匀性（模拟真实泄漏形态）
        x_start = random.uniform(0.05, 0.2)
        x_end = random.uniform(0.75, 0.95)
        
        lines += [
            f"",
            f"## 油污染层",
            f"#material: {oil_er} {oil_sigma} 1.0 0.0 {oil_name}",
            f"#box: {x_start:.3f} {y_bot:.3f} 0 {x_end:.3f} {y_top:.3f} 0.002 {oil_name}",
        ]
        
        # 有概率添加第二个分离油团（模拟泄漏扩散）
        if random.random() > 0.5:
            x2s = random.uniform(0.1, 0.4)
            x2e = random.uniform(0.5, 0.8)
            y2t = y_top + random.uniform(0, 0.05)
            y2b = y2t - thickness * random.uniform(0.3, 0.8)
            if y2b > 0 and y2t < ant_height:
                lines.append(
                    f"#box: {x2s:.3f} {y2b:.3f} 0 {x2e:.3f} {y2t:.3f} 0.002 {oil_name}"
                )
    
    # 添加随机干扰目标（负样本中的地下结构）
    if config.get('add_clutter') and label == 'no_oil':
        clutter_depth = random.uniform(0.1, 0.8)
        y_clutter = ant_height - clutter_depth
        if y_clutter > 0.05:
            x_c = random.uniform(0.2, 0.8)
            radius = random.uniform(0.02, 0.08)
            # 圆柱管道（截面圆形，用cylinder命令）
            lines += [
                f"",
                f"## 干扰目标（地下管道）",
                f"#material: 3.0 0.001 1.0 0.0 pvc_pipe",
                f"#cylinder: {x_c:.3f} {y_clutter:.3f} 0 {x_c:.3f} {y_clutter:.3f} 0.002 {radius:.3f} pvc_pipe",
            ]
    
    lines += [
        f"",
        f"## 地表空气层",
        f"#box: 0 {ant_height:.3f} 0 1.0 {ant_height+0.03:.3f} 0.002 free_space",
        f"",
        f"## 天线",
        f"#waveform: ricker 1 {freq:.2e} mypulse",
        f"#hertzian_dipole: z 0.05 {ant_height + 0.015:.3f} 0 mypulse",
        f"#rx: 0.07 {ant_height + 0.015:.3f} 0",
        f"#src_steps: 0.005 0 0",
        f"#rx_steps: 0.005 0 0",
        f"",
        f"#pml_cells: 10",
    ]
    
    content = "\n".join(lines)
    with open(output_path, 'w') as f:
        f.write(content)
    
    return output_path


def run_simulation(input_file: str, n_traces: int = 180) -> str:
    """运行单次gprMax仿真"""
    cmd = [
        "python", "-m", "gprMax",
        input_file,
        "-n", str(n_traces),
        "--gpu"  # 如有GPU则启用CUDA加速
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"仿真失败: {result.stderr}")
        return None
    # 返回输出HDF5文件路径
    return input_file.replace('.in', '.out')


def h5_to_bscan_image(h5_path: str, output_img_path: str,
                       apply_gain: bool = True,
                       add_noise: float = 0.0) -> np.ndarray:
    """
    将gprMax输出的HDF5文件转换为B-scan灰度图像

    Args:
        h5_path: .out 文件路径
        output_img_path: 保存的PNG路径
        apply_gain: 是否应用时变增益
        add_noise: 添加高斯噪声的标准差（0表示不加）
    """
    import h5py
    from PIL import Image

    with h5py.File(h5_path, 'r') as f:
        # gprMax3输出格式
        bscan = f['rxs']['rx1']['Ez'][:]  # shape: (n_samples, n_traces)

    # 背景去除（减去水平均值，消除直达波和地表反射）
    mean_trace = np.mean(bscan, axis=1, keepdims=True)
    bscan = bscan - mean_trace

    # 时变增益补偿（补偿几何扩散和介质衰减）
    if apply_gain:
        n_samples = bscan.shape[0]
        t = np.linspace(0, 1, n_samples)
        # 指数增益: g(t) = exp(α·t)，α由土壤电导率决定
        gain = np.exp(2.0 * t)
        bscan = bscan * gain[:, np.newaxis]

    # 可选添加噪声（模拟真实采集噪声）
    if add_noise > 0:
        bscan += np.random.normal(0, add_noise * np.max(np.abs(bscan)), bscan.shape)

    # 归一化到 [0, 255]
    abs_max = np.percentile(np.abs(bscan), 99)  # 使用99分位避免极值影响
    bscan_norm = np.clip(bscan / abs_max, -1, 1)
    bscan_uint8 = ((bscan_norm + 1) / 2 * 255).astype(np.uint8)

    # 调整图像尺寸到统一大小 (H=224, W=224) 供模型输入
    img = Image.fromarray(bscan_uint8)
    img = img.resize((224, 224), Image.LANCZOS)
    img.save(output_img_path)

    return np.array(img)


def main():
    os.makedirs("dataset/positive", exist_ok=True)
    os.makedirs("dataset/negative", exist_ok=True)
    os.makedirs("dataset/sim_inputs", exist_ok=True)

    idx = 0
    # 正样本生成
    for soil, freq, oil, depth, thickness in product(
        PARAM_SPACE['soil_type'][:2],   # 两种土壤
        PARAM_SPACE['frequency'],        # 两种频率
        PARAM_SPACE['oil_type'],         # 四种油
        PARAM_SPACE['oil_depth'][:4],    # 四种深度
        PARAM_SPACE['oil_thickness']     # 三种厚度
    ):
        for repeat in range(N_RANDOM_AUGMENT):
            config = {
                'soil_type': soil,
                'frequency': freq,
                'oil_type': oil,
                'oil_depth': depth + random.uniform(-0.05, 0.05),
                'oil_thickness': thickness,
                'add_clutter': random.random() > 0.7,
            }
            in_file = f"dataset/sim_inputs/pos_{idx:05d}.in"
            generate_input_file(config, in_file, 'oil')
            out_file = run_simulation(in_file)
            if out_file:
                h5_to_bscan_image(
                    out_file,
                    f"dataset/positive/pos_{idx:05d}.png",
                    add_noise=random.uniform(0, 0.03)
                )
            idx += 1

    print(f"正样本生成完成: {idx} 个")
    idx = 0
    # 负样本生成（全部土壤类型 + 随机干扰物）
    for soil, freq, add_clutter in product(
        PARAM_SPACE['soil_type'],
        PARAM_SPACE['frequency'],
        [True, False]
    ):
        for repeat in range(N_RANDOM_AUGMENT * 3):
            config = {
                'soil_type': soil,
                'frequency': freq,
                'add_clutter': add_clutter,
            }
            in_file = f"dataset/sim_inputs/neg_{idx:05d}.in"
            generate_input_file(config, in_file, 'no_oil')
            out_file = run_simulation(in_file)
            if out_file:
                h5_to_bscan_image(
                    out_file,
                    f"dataset/negative/neg_{idx:05d}.png",
                    add_noise=random.uniform(0, 0.05)
                )
            idx += 1

    print(f"负样本生成完成: {idx} 个")


if __name__ == '__main__':
    main()
```

### 5.5 仿真参数变异矩阵（确保场景多样性）

| 变量 | 范围/取值 | 说明 |
|------|----------|------|
| 土壤类型 | 干砂/湿砂/粘土/壤土/砾石 | 覆盖主要地质类型 |
| 土壤含水量 | 5% – 35% | Peplinski模型参数 |
| 土壤分形维数 | 1.5 – 2.5 | 控制土壤非均质程度 |
| 天线频率 | 500 MHz, 1 GHz | |
| 天线离地高度 | 8 – 30 mm | ±扰动模拟实测抖动 |
| 天线间距（偏移） | 15 – 25 mm | |
| 油层埋深 | 0.1 – 1.5 m | 重点 < 1m |
| 油层厚度 | 2 – 30 cm | |
| 油层形态 | 水平层/不规则团/多段分布 | |
| 油类型 | 汽油/柴油/原油/油饱和土 | |
| 含水量–油共存 | LNAPL/DNAPL界面 | 轻质/重质非水相液体 |
| 地面粗糙度 | RMS高度 0 – 5 mm | rough_surface命令 |
| 干扰目标 | 无/PVC管/金属管/空洞 | 负样本难度提升 |
| 环境噪声 | SNR 15 – 40 dB | 后处理叠加 |

---

## 6. 数据预处理流程

### 6.1 标准处理管线

```python
# preprocessing.py
import numpy as np
import cv2
from PIL import Image
from scipy import signal

class GPRPreprocessor:
    """GPR B-scan图像标准化处理器"""

    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def background_removal(self, bscan: np.ndarray) -> np.ndarray:
        """
        背景去除（消除直达波和地表强反射）
        方法：减去水平方向均值（等效于高通时间门控）
        """
        # 方法1：均值去除（适合均匀背景）
        mean_trace = np.mean(bscan, axis=1, keepdims=True)
        bscan_br = bscan - mean_trace

        # 方法2（可选）：SVD低秩分量去除（适合非均匀背景）
        # U, S, Vt = np.linalg.svd(bscan, full_matrices=False)
        # S[:3] = 0  # 去除前3个奇异值（背景分量）
        # bscan_br = U @ np.diag(S) @ Vt

        return bscan_br

    def time_varying_gain(self, bscan: np.ndarray,
                           alpha: float = 2.0) -> np.ndarray:
        """
        时变增益（TVG）：补偿深度衰减
        g(t) = t^n 或 exp(αt)，根据地层衰减选择
        """
        n_samples = bscan.shape[0]
        t_norm = np.linspace(0.01, 1.0, n_samples)
        gain = np.exp(alpha * t_norm)
        return bscan * gain[:, np.newaxis]

    def bandpass_filter(self, bscan: np.ndarray,
                         dt: float, f_low: float, f_high: float) -> np.ndarray:
        """带通滤波（去除低频漂移和高频噪声）"""
        fs = 1.0 / dt
        sos = signal.butter(4, [f_low, f_high],
                           btype='bandpass', fs=fs, output='sos')
        return signal.sosfilt(sos, bscan, axis=0)

    def normalize_to_grayscale(self, bscan: np.ndarray) -> np.ndarray:
        """
        归一化为灰度图（0–255）
        使用百分位裁剪避免离群值影响
        """
        p_low = np.percentile(bscan, 1)
        p_high = np.percentile(bscan, 99)
        bscan_clipped = np.clip(bscan, p_low, p_high)
        bscan_norm = (bscan_clipped - p_low) / (p_high - p_low + 1e-10)
        return (bscan_norm * 255).astype(np.uint8)

    def process(self, raw_bscan: np.ndarray,
                dt: float = 1e-11) -> np.ndarray:
        """完整预处理流水线"""
        # Step 1: 背景去除
        bscan = self.background_removal(raw_bscan)
        # Step 2: 带通滤波（保留目标频带）
        # bscan = self.bandpass_filter(bscan, dt, 200e6, 1.5e9)
        # Step 3: 时变增益
        bscan = self.time_varying_gain(bscan, alpha=1.5)
        # Step 4: 归一化
        gray = self.normalize_to_grayscale(bscan)
        # Step 5: 缩放至统一尺寸
        img = Image.fromarray(gray)
        img = img.resize(self.target_size, Image.LANCZOS)
        return np.array(img)
```

### 6.2 数据增强策略

```python
# augmentation.py
import albumentations as A
import cv2

# 仿真阶段（线下增强，扩充数据集）
SIMULATION_AUGMENT = A.Compose([
    A.HorizontalFlip(p=0.5),          # 水平翻转（扫描方向镜像）
    A.VerticalFlip(p=0.1),            # 垂直翻转（时间轴反转，谨慎使用）
    A.RandomBrightnessContrast(       # 模拟不同增益设置
        brightness_limit=0.2,
        contrast_limit=0.2, p=0.5),
    A.GaussNoise(var_limit=(5, 30), p=0.4),   # 模拟采集噪声
    A.RandomCrop(height=200, width=200, p=0.3),# 随机裁剪
    A.Resize(224, 224),
    A.ElasticTransform(               # 轻微弹性变形（模拟介质非均质）
        alpha=20, sigma=5, p=0.2),
    A.CoarseDropout(                  # 随机遮挡（模拟信号丢失）
        max_holes=8, max_height=20,
        max_width=10, p=0.2),
])

# 训练时在线增强（轻量级）
TRAIN_ONLINE_AUGMENT = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
    A.GaussNoise(var_limit=(2, 15), p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,
                       rotate_limit=2, p=0.3),
])
```

### 6.3 数据集划分

```
总数据集（仿真）: ~6000 张（正负各~3000）
│
├── 训练集: 70%  → 4200 张（在线增强 × 5 = 21000 有效样本）
├── 验证集: 15%  → 900 张（不做增强）
└── 测试集: 15%  → 900 张（仅预处理，评估泛化性）

注：仿真数据/真实数据比例建议最终保持 3:1 到 5:1
```

---

## 7. 模型设计与选型

### 7.1 RK3568 算力约束分析

```
RK3568 NPU 规格:
  算力: 1 TOPS（INT8）
  内存带宽: ~12.8 GB/s (LPDDR4)
  支持精度: INT8, INT16
  最大模型大小建议: < 20 MB (INT8后)

目标推理性能:
  输入尺寸: 224×224×1 (灰度)
  目标延迟: < 200ms/帧 (5+ FPS)
  内存占用: < 100 MB

→ 约束条件: MACs < 300M, 参数量 < 5M
```

### 7.2 候选模型对比

| 模型 | 参数量 | MACs | Top-1(ImageNet) | 备注 |
|------|--------|------|-----------------|------|
| ResNet-50 | 25.6M | 4.1G | 76.1% | ❌ 过重 |
| MobileNetV2 (1.0) | 3.4M | 300M | 72.0% | ✅ 推荐基线 |
| **MobileNetV3-Small** | **2.9M** | **56M** | **67.4%** | ✅ **首选** |
| EfficientNet-Lite0 | 4.7M | 407M | 75.1% | ✅ 可考虑 |
| ShuffleNetV2-0.5x | 1.4M | 41M | 60.6% | ⚠️ 精度偏低 |
| SqueezeNet1.1 | 1.2M | 355M | 58.2% | ⚠️ 精度偏低 |

> **推荐方案**：以 **MobileNetV3-Small** 为主干 + **CBAM注意力模块** + **知识蒸馏**

### 7.3 最终模型架构

```python
# model.py
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )
        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # 通道注意力
        ca = self.channel_att(x).view(x.size(0), -1, 1, 1)
        x = x * ca
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_att(torch.cat([avg_out, max_out], dim=1))
        return x * sa


class GPROilDetector(nn.Module):
    """
    GPR油类泄漏检测模型
    主干: MobileNetV3-Small（预训练ImageNet权重迁移）
    改进: CBAM注意力 + 灰度适配 + 二分类头
    """
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()

        # 加载预训练主干
        backbone = mobilenet_v3_small(pretrained=pretrained)

        # ── 修改输入层：RGB(3通道) → 灰度(1通道) ──
        # 策略：将原始3通道权重在通道维度求平均，初始化1通道卷积
        original_weight = backbone.features[0][0].weight  # shape: [16,3,3,3]
        self.input_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            backbone.features[0][1],  # BN
            backbone.features[0][2],  # Hardswish
        )
        with torch.no_grad():
            self.input_conv[0].weight = nn.Parameter(
                original_weight.mean(dim=1, keepdim=True)
            )

        # 主干特征提取（去掉原始第一层）
        self.features = backbone.features[1:]

        # CBAM注意力（插入在最后特征图上）
        self.cbam = CBAM(channels=96)  # MobileNetV3-Small最后特征图96通道

        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 分类头（简化版，减少参数量）
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, 64),
            nn.Hardswish(),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (B, 1, 224, 224) 灰度图
        x = self.input_conv(x)
        x = self.features(x)
        x = self.cbam(x)
        x = self.gap(x)
        return self.classifier(x)

    def get_cam(self, x):
        """返回Class Activation Map用于可视化"""
        x = self.input_conv(x)
        x = self.features(x)
        x = self.cbam(x)
        feature_map = x  # (B, 96, H, W)
        x = self.gap(x)
        logits = self.classifier(x)
        return logits, feature_map
```

### 7.4 知识蒸馏方案

```python
# distillation.py
"""
教师模型: ResNet-50 (在PC/GPU训练，精度优先)
学生模型: GPROilDetector/MobileNetV3-Small (部署目标)
"""

class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.7):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, labels):
        # 硬标签损失（交叉熵）
        hard_loss = self.ce_loss(student_logits, labels)

        # 软标签损失（KL散度）
        soft_student = torch.log_softmax(student_logits / self.T, dim=1)
        soft_teacher = torch.softmax(teacher_logits / self.T, dim=1)
        soft_loss = self.kl_loss(soft_student, soft_teacher) * (self.T ** 2)

        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss
```

---

## 8. 训练策略

### 8.1 训练流程

```
阶段一：仿真数据预训练（第1–50轮）
  ├─ 使用ImageNet预训练权重初始化
  ├─ 冻结主干前7层，只训练后几层+分类头（前10轮）
  ├─ 解冻全部层，端到端微调（第10轮起）
  ├─ 学习率: 余弦退火 (1e-3 → 1e-6)
  ├─ 优化器: AdamW, weight_decay=1e-4
  └─ 损失函数: Focal Loss（处理难样本）

阶段二：知识蒸馏（第51–80轮）
  ├─ 教师模型: 已训练的ResNet-50（精度优先版本）
  ├─ 学生模型: GPROilDetector（部署版本）
  └─ 蒸馏损失: α=0.7 硬标签 + 0.3 软标签

阶段三：量化感知训练QAT（第81–100轮）
  ├─ 插入伪量化节点（模拟INT8量化误差）
  ├─ 极小学习率: 1e-5
  └─ 为RKNN部署做精度保留优化
```

### 8.2 关键训练参数

```python
# train_config.py
TRAIN_CONFIG = {
    # 数据
    'input_size': (224, 224),
    'batch_size': 64,           # GPU训练批大小
    'num_workers': 8,

    # 模型
    'num_classes': 2,           # oil / no_oil
    'pretrained': True,         # ImageNet预训练权重

    # 优化
    'epochs': 100,
    'lr_init': 1e-3,
    'lr_min': 1e-6,
    'weight_decay': 1e-4,
    'optimizer': 'adamw',

    # 损失函数
    'loss': 'focal',            # Focal Loss
    'focal_gamma': 2.0,
    'focal_alpha': 0.75,        # 正样本权重（如正负不平衡）

    # 正则化
    'dropout': 0.3,
    'label_smoothing': 0.1,
    'mixup_alpha': 0.2,         # Mixup数据增强

    # 早停
    'early_stopping_patience': 15,
    'monitor_metric': 'val_f1',

    # 蒸馏
    'distill_temperature': 4.0,
    'distill_alpha': 0.7,

    # 输出
    'save_dir': 'checkpoints/',
    'log_dir': 'runs/',
}
```

### 8.3 类别不平衡处理

```python
# 采样策略（针对仿真数据正负样本可能不均衡的情况）
from torch.utils.data import WeightedRandomSampler

def get_balanced_sampler(dataset):
    class_counts = [len(dataset.neg_files), len(dataset.pos_files)]
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = torch.zeros(len(dataset))
    for i, (_, label) in enumerate(dataset):
        sample_weights[i] = weights[label]
    return WeightedRandomSampler(sample_weights, len(dataset), replacement=True)
```

---

## 9. 模型压缩与量化

### 9.1 压缩技术路线

```
原始FP32模型 (MobileNetV3-Small + CBAM)
     约 3.5MB / ~56M MACs
          │
          ├── 结构化剪枝（可选，削减20-30%通道）
          │     工具: torch.nn.utils.prune
          │     策略: L1范数通道剪枝
          │     目标: 压缩至 45M MACs
          │
          ├── 量化感知训练（QAT）
          │     框架: PyTorch FX Graph Mode Quantization
          │     精度: INT8
          │     校准: 使用200张代表性仿真图像
          │
          └── RKNN格式转换
                工具: RKNN-Toolkit2
                最终大小: ~1.2MB (INT8)
                目标延迟: < 150ms on RK3568 NPU
```

### 9.2 RKNN模型转换

```python
# convert_to_rknn.py
from rknn.api import RKNN
import numpy as np
from PIL import Image

def convert_model(onnx_path: str, rknn_path: str,
                  calibration_images: list):
    """
    将ONNX模型转换为RKNN格式（INT8量化）

    Args:
        onnx_path: 导出的ONNX模型路径
        rknn_path: 输出RKNN模型路径
        calibration_images: 校准图像路径列表（建议200+张）
    """
    rknn = RKNN(verbose=True)

    # 1. 配置转换参数
    rknn.config(
        mean_values=[[127.5]],          # 灰度图归一化（单通道）
        std_values=[[127.5]],
        target_platform='rk3568',
        quantized_dtype='asymmetric_quantized-8',  # INT8非对称量化
        quantized_algorithm='normal',              # 量化算法
        quantized_method='layer',                  # 逐层量化
        optimization_level=3,                      # 最高优化等级
    )

    # 2. 加载ONNX模型
    ret = rknn.load_onnx(
        model=onnx_path,
        input_size_list=[[1, 1, 224, 224]]  # 灰度图
    )
    assert ret == 0, "模型加载失败"

    # 3. 构建（含量化）
    # 准备校准数据集
    dataset_file = 'calibration_dataset.txt'
    with open(dataset_file, 'w') as f:
        for img_path in calibration_images:
            f.write(img_path + '\n')

    ret = rknn.build(
        do_quantization=True,
        dataset=dataset_file,
        rknn_batch_size=1
    )
    assert ret == 0, "模型构建失败"

    # 4. 精度分析（验证量化误差）
    ret = rknn.accuracy_analysis(
        inputs=['sample_input.npy'],
        output_dir='./accuracy_analysis',
        target=None  # None表示在PC上模拟
    )

    # 5. 导出RKNN模型
    ret = rknn.export_rknn(rknn_path)
    assert ret == 0, "模型导出失败"

    # 6. 性能评估
    rknn.init_runtime(target=None)  # 模拟器评估
    perf = rknn.eval_perf(is_print=True)

    rknn.release()
    print(f"模型已保存至: {rknn_path}")
    return perf


# PyTorch → ONNX 导出
def export_to_onnx(model_path: str, onnx_path: str):
    import torch
    from model import GPROilDetector

    model = GPROilDetector(num_classes=2)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dummy_input = torch.randn(1, 1, 224, 224)  # 灰度图
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=12,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        do_constant_folding=True,
    )
    print(f"ONNX模型已导出: {onnx_path}")
```

---

## 10. RK3568部署

### 10.1 板端推理代码

```python
# inference_rk3568.py
"""
RK3568 NPU 推理引擎
使用 RKNN-Toolkit-Lite2 进行板端部署
"""
import numpy as np
import cv2
from rknnlite.api import RKNNLite
from preprocessing import GPRPreprocessor

class OilLeakDetector:
    """油类泄漏实时检测器"""

    LABELS = {0: 'no_oil', 1: 'oil_detected'}
    RISK_COLORS = {
        'no_oil': (0, 255, 0),       # 绿色：安全
        'oil_detected': (0, 0, 255), # 红色：告警
    }

    def __init__(self, model_path: str):
        self.preprocessor = GPRPreprocessor(target_size=(224, 224))
        self.model = self._load_model(model_path)
        print(f"[OilLeakDetector] 模型加载完成: {model_path}")

    def _load_model(self, path: str) -> RKNNLite:
        rknn_lite = RKNNLite(verbose=False)
        ret = rknn_lite.load_rknn(path)
        assert ret == 0, f"模型加载失败: {path}"

        # 初始化NPU运行时
        # core_mask=RKNNLite.NPU_CORE_0 for RK3568单核NPU
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        assert ret == 0, "NPU运行时初始化失败"
        return rknn_lite

    def preprocess(self, raw_bscan: np.ndarray) -> np.ndarray:
        """原始B-scan → 模型输入张量"""
        gray = self.preprocessor.process(raw_bscan)
        # 转换为NCHW格式
        tensor = gray.astype(np.float32)[np.newaxis, np.newaxis, :, :]
        # 归一化到 [-1, 1]
        tensor = (tensor / 127.5) - 1.0
        return tensor

    def inference(self, tensor: np.ndarray) -> dict:
        """执行NPU推理"""
        outputs = self.model.inference(inputs=[tensor])
        logits = outputs[0][0]  # shape: (2,)
        probs = self._softmax(logits)
        pred_class = int(np.argmax(probs))
        return {
            'label': self.LABELS[pred_class],
            'confidence': float(probs[pred_class]),
            'prob_oil': float(probs[1]),
            'prob_no_oil': float(probs[0]),
        }

    @staticmethod
    def _softmax(x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def detect(self, raw_bscan: np.ndarray) -> dict:
        """完整检测流程（含预处理）"""
        tensor = self.preprocess(raw_bscan)
        result = self.inference(tensor)
        return result

    def release(self):
        self.model.release()


# ─────────────────────────────────────────────────────────
# 使用示例
# ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    import h5py

    detector = OilLeakDetector('models/oil_detector_int8.rknn')

    # 从GPR硬件采集的HDF5文件读取
    with h5py.File('field_data/scan_001.h5', 'r') as f:
        raw_bscan = f['bscan'][:]

    result = detector.detect(raw_bscan)
    print(f"检测结果: {result['label']}")
    print(f"置信度: {result['confidence']:.3f}")
    print(f"油类泄漏概率: {result['prob_oil']:.3f}")

    if result['label'] == 'oil_detected' and result['confidence'] > 0.85:
        print("⚠️  警告：检测到油类泄漏风险！建议进一步确认。")

    detector.release()
```

### 10.2 性能优化技巧

```python
# 针对RK3568 NPU的优化建议

# 1. 批量推理（减少NPU调度开销）
# 对于连续扫描场景，累积4帧后批量推理
BATCH_SIZE = 1  # RK3568建议batch=1以降低延迟

# 2. 图像缓存预分配
input_buffer = np.zeros((1, 1, 224, 224), dtype=np.float32)

# 3. NPU/CPU并行：预处理在CPU，推理在NPU
import threading
from queue import Queue
preprocess_queue = Queue(maxsize=3)
result_queue = Queue(maxsize=3)

# 4. 使用OpenCV硬件加速预处理
# RK3568支持RGA（Raster Graphic Acceleration）加速图像缩放
# 可通过 rga-python 调用
```

---

## 11. 真实数据补充与迁移学习

### 11.1 数据采集规范

```
野外采集标准流程：
  1. 场地准备
     ├─ 记录场地土壤类型、含水量（TDR仪器测量）
     ├─ 测量天气条件（温度、湿度）
     └─ GPS坐标记录

  2. GPR扫描
     ├─ 使用500MHz或1GHz屏蔽天线
     ├─ 扫描线间距: 0.5m
     ├─ 天线移动速度: < 0.5 m/s（手推匀速）
     └─ 重复扫描3次取平均

  3. 真值标注
     ├─ 化学分析取样（钻孔TPH检测）作为金标准
     ├─ 标注B-scan图像中的疑似区域
     └─ 专家复核（≥2名工程师确认）

  4. 数据格式
     └─ 原始数据: .DT1/.HD (GSSI) 或 .rd3/.rad (MALA)
        → 转换脚本统一处理为HDF5格式
```

### 11.2 域适应微调策略

```python
# fine_tune_real.py
"""
策略：渐进式微调 + 域对齐
真实数据量少（通常<200张）→ 需要防止过拟合
"""

def fine_tune_with_real_data(
    pretrained_model_path: str,
    real_data_dir: str,
    output_path: str,
    real_data_ratio: float = 0.3,  # 真实数据占训练集比例
):
    """
    混合训练策略：
    - 真实数据与仿真数据以 3:7 比例混合
    - 使用较低学习率，防止遗忘仿真知识
    - 真实数据施加更大的增强（扩充有限的真实样本）
    """
    model = GPROilDetector(num_classes=2)
    checkpoint = torch.load(pretrained_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 冻结底层特征提取（保留仿真学到的通用特征）
    for name, param in model.named_parameters():
        if 'features.0' in name or 'features.1' in name:
            param.requires_grad = False

    # 对真实数据使用更激进的增强
    real_augment = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(0.3, 0.3, p=0.7),
        A.GaussNoise(var_limit=(10, 50), p=0.6),
        A.RandomCrop(190, 190, p=0.4),
        A.Resize(224, 224),
    ])

    # 训练参数（微调阶段）
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,           # 比预训练低10倍
        weight_decay=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=30, eta_min=1e-6
    )
    # 早停: 真实数据验证集F1
    # 推荐: 30轮微调 + Early Stopping(patience=10)
```

### 11.3 主动学习策略（持续改进）

```
当真实数据逐渐积累时，采用主动学习最大化标注效率：

1. 不确定性采样：选择模型置信度在 [0.4, 0.6] 区间的样本优先标注
2. 多样性采样：使用聚类确保标注样本覆盖不同土壤/场景类型
3. 周期性重训练：每积累50张真实标注样本，触发一次微调
4. 错误分析：重点分析假阳性（误报油污）和假阴性（漏检油污）案例
```

---

## 12. 性能评估指标

### 12.1 模型精度指标

```python
# 使用F1-Score作为主要指标（比准确率更适合检测任务）

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score
)

EVALUATION_METRICS = {
    '主要指标': 'F1-Score (Macro)',
    '召回率': '漏检率低（油污未检出的代价更高）',
    '精确率': '误报率可接受范围',
    'AUC-ROC': '模型判别能力评估',
    'AP': '平均精度（单类别）',
}

# 目标性能（分阶段）
PERFORMANCE_TARGETS = {
    '仿真数据测试集':   {'F1': 0.90, 'Recall': 0.92, 'Precision': 0.88},
    '含真实数据微调后': {'F1': 0.88, 'Recall': 0.90, 'Precision': 0.86},
    'RK3568 INT8量化': {'F1_degradation': '< 2%'},  # 量化精度损失
}
```

### 12.2 推理性能指标

| 指标 | 目标值 | 测量方法 |
|------|--------|---------|
| 单帧推理延迟 | < 200ms | RK3568实测 |
| 吞吐量 | ≥ 5 FPS | 连续推理压测 |
| NPU内存占用 | < 50 MB | rknn_perf工具 |
| 模型文件大小 | < 5 MB | ls -lh |
| CPU占用（预处理）| < 30% | top命令 |

### 12.3 混淆矩阵分析框架

```
                     预测: no_oil    预测: oil
真实: no_oil   [  TN(误报少) |  FP(误报)  ]
真实: oil      [  FN(漏检)   |  TP(正确)  ]

关键原则：
  FN（漏检）危害 >> FP（误报）危害
  → 在精度-召回率权衡中，倾向于提高召回率
  → 决策阈值: 0.4（而非默认0.5）——降低油污漏检风险
```

---

## 13. 项目目录结构

```
gpr-oil-scan/
│
├── README.md                          # 本文件
│
├── simulation/                        # GPRmax仿真模块
│   ├── templates/                     # .in 文件模板
│   │   ├── positive_oil_layer.in      # 油层正样本模板
│   │   ├── positive_oil_blob.in       # 油团正样本模板
│   │   └── negative_baseline.in       # 负样本基线模板
│   ├── generate_dataset.py            # 批量仿真生成脚本
│   ├── h5_to_image.py                 # HDF5转图像工具
│   └── param_config.yaml              # 仿真参数配置文件
│
├── preprocessing/                     # 数据预处理模块
│   ├── preprocessor.py                # GPRPreprocessor类
│   ├── augmentation.py                # 数据增强配置
│   └── dataset.py                     # PyTorch Dataset类
│
├── models/                            # 模型定义与权重
│   ├── gpr_oil_detector.py            # 主模型（MobileNetV3+CBAM）
│   ├── teacher_model.py               # 教师模型（ResNet-50）
│   ├── distillation.py                # 知识蒸馏训练逻辑
│   ├── checkpoints/                   # 训练检查点
│   │   ├── best_f1.pth                # 最佳F1模型权重
│   │   └── latest.pth                 # 最新检查点
│   └── exported/
│       ├── oil_detector.onnx          # ONNX格式
│       └── oil_detector_int8.rknn     # 最终部署模型
│
├── training/                          # 训练脚本
│   ├── train.py                       # 主训练入口
│   ├── train_distill.py               # 蒸馏训练脚本
│   ├── fine_tune_real.py              # 真实数据微调脚本
│   └── train_config.py                # 训练超参数配置
│
├── evaluation/                        # 评估与可视化
│   ├── evaluate.py                    # 模型评估脚本
│   ├── visualize_cam.py               # CAM热力图可视化
│   └── plot_metrics.py                # 训练曲线绘制
│
├── deployment/                        # 部署相关
│   ├── convert_to_onnx.py             # PyTorch→ONNX
│   ├── convert_to_rknn.py             # ONNX→RKNN转换
│   ├── inference_rk3568.py            # 板端推理主程序
│   └── calibration_data/              # INT8量化校准图像
│
├── dataset/                           # 数据集（不入Git）
│   ├── simulation/
│   │   ├── positive/                  # 仿真正样本 (~3000张)
│   │   └── negative/                  # 仿真负样本 (~3000张)
│   ├── real/                          # 真实采集数据（后续补充）
│   │   ├── positive/
│   │   └── negative/
│   └── splits/
│       ├── train.txt
│       ├── val.txt
│       └── test.txt
│
├── docs/                              # 技术文档
│   ├── simulation_guide.md            # 仿真详细指南
│   ├── deployment_guide.md            # 部署操作手册
│   └── data_collection_sop.md        # 野外采集SOP
│
├── requirements.txt                   # PC端Python依赖
├── requirements_rk3568.txt            # 板端Python依赖
└── .gitignore
```

---

## 14. 已知挑战与解决方案

### 14.1 仿真-真实域偏移（最关键挑战）

| 挑战 | 根本原因 | 解决方案 |
|------|---------|---------|
| 天线近场效应 | 简化天线模型 | 使用gprMax内置天线模型库（如MALA/GSSI） |
| 土壤频散特性 | 简单Debye模型不足 | 使用多极Debye模型拟合真实频散 |
| 耦合干扰 | 真实天线存在互耦 | 仿真中建立真实天线几何模型 |
| 地表杂波 | 地面不平整 | 使用`#rough_surface`命令添加粗糙度 |
| 多路径效应 | 地上物体反射 | 在仿真域边界添加散射体 |

### 14.2 小数据集过拟合

```
问题：真实标注数据少（< 200张）时容易过拟合

解决方案组合：
  1. Mixup / CutMix 数据增强
  2. Dropout (0.3–0.5)
  3. Label Smoothing (0.1)
  4. 迁移学习（冻结低层特征）
  5. 早停（patience=10）
  6. 集成学习（3个模型投票）
```

### 14.3 RK3568 INT8量化精度损失

```
问题：FP32→INT8量化后精度下降 > 3%

解决方案：
  1. QAT（量化感知训练）优于PTQ
  2. 使用有代表性的校准集（200+张多样化图像）
  3. 对量化敏感层（如CBAM注意力）使用FP16
  4. 分析RKNN精度工具定位问题层
  5. 调整量化算法（normal/mmse/percentile）
```

### 14.4 弥散性油污特征不明显

```
问题：弥散态油污（低浓度）在B-scan中无明显双曲线，难以识别

解决方案：
  1. 仿真中增加低浓度油饱和土（εᵣ=7–10）的训练样本
  2. 使用差分B-scan（与历史基准比较）突出异常
  3. 多频联合分析（500MHz + 1GHz融合判断）
  4. 在分类头添加不确定性估计（MC-Dropout）
     → 输出 "高置信油" / "低置信疑似" / "无油" 三类
```

---

## 15. 参考文献

1. Warren, C., Giannopoulos, A., & Giannakis, I. (2016). **gprMax: Open source software to simulate electromagnetic wave propagation for Ground Penetrating Radar**. *Computer Physics Communications*, 209, 163–170.

2. Sandler, M., et al. (2018). **MobileNetV2: Inverted Residuals and Linear Bottlenecks**. *CVPR 2018*.

3. Howard, A., et al. (2019). **Searching for MobileNetV3**. *ICCV 2019*.

4. Woo, S., et al. (2018). **CBAM: Convolutional Block Attention Module**. *ECCV 2018*.

5. Hinton, G., Vinyals, O., & Dean, J. (2015). **Distilling the Knowledge in a Neural Network**. *NIPS Deep Learning Workshop*.

6. Peplinski, N.R., Ulaby, F.T., & Dobson, M.C. (1995). **Dielectric Properties of Soils in the 0.3–1.3-GHz Range**. *IEEE TGRS*, 33(3), 803–807.

7. Daniels, D.J. (2004). **Ground Penetrating Radar** (2nd ed.). IET, London.

8. Giannakis, I., et al. (2016). **A Realistic FDTD Numerical Modeling Framework of Ground Penetrating Radar for Landmine Detection**. *IEEE JSTARS*, 9(1), 37–51.

9. Tong, Z., et al. (2022). **Lightweight CNN model for automatic detection of subsurface voids using GPR B-scan data**. *Construction and Building Materials*.

10. Rockchip. (2023). **RKNN-Toolkit2 User Guide**. Rockchip Electronics Co., Ltd.

---

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/your-org/gpr-oil-scan.git
cd gpr-oil-scan

# 2. 安装依赖（PC端）
pip install -r requirements.txt

# 3. 生成仿真数据（需先安装gprMax）
python simulation/generate_dataset.py --n_pos 3000 --n_neg 3000

# 4. 训练模型
python training/train.py --config training/train_config.py

# 5. 转换为RKNN格式
python deployment/convert_to_onnx.py --checkpoint models/checkpoints/best_f1.pth
python deployment/convert_to_rknn.py --onnx models/exported/oil_detector.onnx

# 6. 板端推理（在RK3568上运行）
python deployment/inference_rk3568.py --model models/exported/oil_detector_int8.rknn \
                                       --input field_data/scan_001.h5
```

---

*本项目持续迭代中。随着真实采集数据的积累，模型性能将持续提升。*

*如有技术问题或合作意向，请提交 Issue 或联系项目负责人。*
