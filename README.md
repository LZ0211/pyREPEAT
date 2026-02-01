# pyREPEAT v2.0 - 高性能静电势电荷拟合

> **RE**peat **P**ackage for **E**lectrostatic potential **A**tomic charges in periodic sys**T**ems

pyREPEAT是一个用于周期性体系静电势电荷拟合的高性能科学计算软件。本版本(v2.0)进行了全面的性能优化，支持多核CPU、GPU加速以及混合精度计算，相比官方基础的C++的无GPU加速版本最高可提升**50~100倍**性能。

---

## 📋 目录

- [主要特性](#-主要特性)
- [快速开始](#-快速开始)
- [输入文件格式](#-输入文件格式)
- [参数详解](#-参数详解)
- [GPU配置](#-gpu配置)
- [故障排除](#-故障排除)
- [示例](#-示例)
- [引用](#-引用)

---

## ✨ 主要特性

- 🚀 **多后端并行**: CPU (Numba/多进程) + GPU (CUDA)
- 🎯 **智能策略**: 自动选择最优计算策略
- 💎 **混合精度**: fp32计算，精度损失<1e-6
- 🎮 **多GPU支持**: Grid过滤和Ewald计算均支持多GPU
- 🧠 **显存管理**: 自动检测/清理/分块，避免OOM
- 🔄 **Pinned Memory**: CPU-GPU传输加速2-3倍
- 📊 **统计可选**: 默认跳过统计计算，节省5-10倍时间
- 💾 **Phi矩阵缓存**: 内存充足时统计计算<1秒

---

## 🚀 快速开始

### 系统要求

- **操作系统**: Linux, macOS, Windows
- **Python**: 3.8+ (推荐 3.10-3.12)
- **内存**: 最少 8GB，推荐 32GB+
- **GPU**: 可选，NVIDIA GPU with CUDA 11.x/12.x


### 安装

```bash
# 创建环境
conda create -n repeat python=3.11
conda activate repeat

# 安装基础依赖
pip install numpy scipy psutil numba

# 安装GPU支持 (可选)
# CUDA环境需自己配置
pip install cupy-cuda11x  # 或 cupy-cuda12x

# 方案2: PyTorch (备选)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 命令行参数

```
python repeat.py [CUBE_FILE] [OPTIONS]

位置参数:
  cube_file             输入的CUBE文件路径

可选参数:
  --fit-type {0,1}      拟合类型: 0=不考虑周期性, 1=考虑周期性 (默认: 1)
  --vdw-factor FLOAT    VDW半径缩放因子 (默认: 1.0)
  --vdw-max FLOAT       VDW半径上限 (默认: 1000.0)
  --cutoff FLOAT        实空间截断半径 (默认: 20.0)
  --total-charge FLOAT  体系总电荷 (默认: 0.0)
  --cores N             CPU核心数 (默认: 自动检测)
  --gpu [DEVICES]       启用GPU加速
                        示例: --gpu, --gpu 0, --gpu 0,1,2, --gpu all
  --stats               计算统计量(RMS误差)。默认关闭以节省时间
  --fp64                强制双精度fp64 (默认fp32)
  --block-k N           k空间分块大小 (默认: 64)
  --symm-file PATH      对称性约束文件
  --resp-file PATH      RESP参数文件
  --qeq-file PATH       QEq参数文件
  --charge {repeat,qeq} 电荷模型 (默认: repeat)
  --output PATH         输出文件路径
```

### 基础使用

```bash
# CPU计算 (默认fp32混合精度，默认jit加速)
python repeat.py water.cube

# CPU计算 (fp64精度，关闭jit加速，最慢的版本)
python repeat.py water.cube --fp64 --no-jit

# GPU加速 (默认fp32混合精度)
python repeat.py system.cube --gpu

# 多GPU并行 (默认fp32)
python repeat.py large_system.cube --gpu 0,1,2,3

# 强制双精度 (fp64) - 更慢但更高精度
python repeat.py system.cube --fp64
python repeat.py system.cube --gpu --fp64
```


### 使用示例

#### 示例1: 基础REPEAT计算
```bash
python repeat.py water.cube --total-charge 0.0
```

#### 示例2: CPU并行优化
```bash
# 使用8核
python repeat.py system.cube --cores 8

# 使用所有核心
python repeat.py system.cube
```

#### 示例3: GPU加速
```bash
# 单GPU (默认fp32)
python repeat.py large.cube --gpu

# 指定GPU
python repeat.py large.cube --gpu 1

# 多GPU
python repeat.py huge.cube --gpu 0,1,2

# 使用所有GPU
python repeat.py huge.cube --gpu all

# 强制双精度
python repeat.py large.cube --gpu --fp64
```

#### 示例4: 对称性约束
```bash
python repeat.py symmetric.cube --symm-file symmetry.input
```

#### 示例5: 启用统计计算
```bash
# 默认只计算电荷（最快）
python repeat.py system.cube --gpu

# 启用RMS误差统计
python repeat.py system.cube --gpu --stats

# 输出详细数据（自动启用统计）
python repeat.py system.cube --gpu --output results.dat
```

---

## 📁 输入文件格式

### CUBE文件 (必需)

标准Gaussian CUBE格式：
```
COMMENT LINE 1
COMMENT LINE 2
   10    0.000000    0.000000    0.000000    # 原子数 + 原点
   50    0.200000    0.000000    0.000000    # x方向: 50点, 步长0.2
   50    0.000000    0.200000    0.000000    # y方向: 50点, 步长0.2
   50    0.000000    0.000000    0.200000    # z方向: 50点, 步长0.2
    8    0.000000    2.500000    3.000000    4.000000  # 原子序, 电荷, x, y, z
    1    0.000000    1.500000    2.000000    3.000000
 ...
 1.23456E-02  2.34567E-02  3.45678E-02  ...  # 静电势数据
```

### 对称性文件 (可选)

文件名: `symmetry.input`

**格式**: 每行定义一组等效原子，支持范围表示法

```
# 格式: 每行一组等效原子
# 支持范围 (1-5)、单个索引 (9)、逗号分隔列表
# 每行第一个原子为基准原子

# 甲基氢原子 (1,2,3,4,5 等效)
1-5

# 两个等效原子
9,10

# 苯环上的等效原子
15-18

# 混合格式
20,22,25-27
```

**格式说明**:
- `1-5` 表示原子 1,2,3,4,5
- `9,10` 表示原子 9 和 10
- `15-18` 表示原子 15,16,17,18
- `20,22,25-27` 表示原子 20,22,25,26,27
- 每行第一个原子为基准原子，其余为关联原子
- 注释以 `#` 开头

### RESP参数文件 (可选)

文件名: `RESP.dat`
```
# 格式: 原子索引 电荷约束 权重
1 0.0 0.1
2 0.0 0.1
3 0.0 0.1
```

### QEq参数文件 (可选)

文件名: `QEq.dat`
```
# 格式: 元素符号 电负性(Hartree) 1/2硬度(Hartree)
H  0.1664  0.2552
C  0.1996  0.2152
N  0.2458  0.2434
O  0.3202  0.3149
```

---

## ⚙️ 参数详解

### 并行计算选项

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--cores N` | CPU核心数 | 自动检测 |
| `--gpu [DEVICES]` | GPU设备 | 0, 0,1,2, all |
| `--fp64` | 强制双精度 | 默认fp32 |

### 精度选择

| 模式 | 命令 | 速度 | 精度损失 |
|------|------|------|----------|
| **fp32 (默认)** | (无参数) | 基准 | <1e-6 |
| **fp64** | `--fp64` | 慢2-3x | 0 |

**建议**: 默认使用fp32，精度足够且GPU计算速度更快。需要极高精度时添加`--fp64`。

### 物理参数

| 参数 | 说明 | 默认值 | 单位 |
|------|------|--------|------|
| `--cutoff` | 实空间截断 | 20.0 | Bohr |
| `--vdw-factor` | VDW半径缩放 | 1.0 | - |
| `--total-charge` | 体系总电荷 | 0.0 | e |

---

## 🎮 GPU配置

### 多GPU并行

```bash
# 使用指定GPU
python repeat.py large.cube --gpu 0,1

# 使用所有可用GPU
python repeat.py large.cube --gpu all
```

### 显存管理

系统自动检测GPU显存并智能分批：
- **显存充足**: 单批次处理所有原子
- **显存有限**: 自动分多批处理
- **显存不足**: 自动回退到CPU
- **自动清理**: 计算完成后自动释放显存

使用pinned memory加速CPU-GPU数据传输。

### 统计计算优化

**默认行为**（推荐，最快）：
```bash
python repeat.py system.cube --gpu
# 只计算电荷，跳过统计计算
```

**启用统计**（需要时）：
```bash
python repeat.py system.cube --gpu --stats
# 计算电荷 + RMS误差
```

**性能说明**：
- **内存充足**（Phi矩阵 < 50%可用内存）：统计计算<1秒（复用拟合时的Phi矩阵）
- **内存不足**：需要重新计算Ewald（1-60分钟，取决于体系大小）
- **建议**：大体系（>100万网格点）只在需要时启用 `--stats`

### 混合精度

默认使用fp32（混合精度），如需fp64：
```bash
python repeat.py large.cube --gpu --fp64
```

---


## 🔧 故障排除

### GPU未检测到
```bash
# 检查CUDA安装
nvidia-smi

# 检查CuPy
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"

# 检查PyTorch
python -c "import torch; print(torch.cuda.is_available())"
```

### 显存不足
```bash
# 系统会自动分批处理
# 或减小block-k
python repeat.py large.cube --gpu --block-k 32

# 或使用混合精度(默认已启用)
python repeat.py large.cube --gpu

# 或回退到CPU
python repeat.py large.cube --cores 16
```

### 精度不够
```bash
# 强制使用fp64双精度
python repeat.py system.cube --fp64
python repeat.py system.cube --gpu --fp64
```

### 常见错误

**错误**: `CUDA out of memory`
- **解决**: 系统会自动分批，无需操作

**错误**: `No module named 'cupy'`
- **解决**: `pip install cupy-cuda11x` (匹配CUDA版本)

**错误**: `GPU memory insufficient`
- **解决**: 自动回退到CPU计算

---

## 💡 示例

### 完整计算工作流

```bash
#!/bin/bash

# 1. 准备输入文件
# system.cube - Gaussian生成的CUBE文件
# symmetry.input - 对称性约束文件 (可选)

# 2. 运行REPEAT计算
echo "Running REPEAT calculation..."
python repeat.py system.cube \
    --gpu 0,1 \
    --symm-file symmetry.input \
    --output charges.txt \
    > repeat.log 2>&1

# 3. 检查结果
if [ $? -eq 0 ]; then
    echo "Calculation completed successfully!"
    tail -20 charges.txt
else
    echo "Calculation failed! Check repeat.log"
fi
```

### 批量处理

```bash
# 批量处理多个体系
for system in *.cube; do
    echo "Processing $system..."
    python repeat.py "$system" --gpu --output "${system%.cube}_charges.txt"
done
```

---

## 🎯 优化建议

### 选择计算策略

| 体系大小 | 推荐配置 | 预期时间 |
|---------|---------|---------|
| < 5k网格 | `--cores 1` | ~30s |
| 5k-50k | `--cores 8` | ~45s |
| 50k-200k | `--gpu` | ~10s |
| > 200k | `--gpu 0,1,2,3` | ~5s |

### 环境变量

```bash
# 性能监控
export REPEAT_PERF=1

# 调整GPU启动阈值
export REPEAT_GPU_MIN=5000

# Numba线程数
export NUMBA_NUM_THREADS=8
```

---

## 📚 引用

如果您在研究中使用了本软件，请引用：

```bibtex
@software{
  author = {Cong Wang},
  title = {pyREPEAT v2.0: High-Performance Electrostatic Potential Fitting},
  year = {2026},
  url = {[https://github.com/LZ0211/pyREPEAT](https://github.com/LZ0211/pyREPEAT)}
}
```

原始REPEAT方法：
- Campañá, C.; Mussard, B.; Woo, T. K. J. Chem. Theory Comput. **2009**, 5, 2866–2878.
- DOI: https://doi.org/10.1021/ct9003405

官方程序：
- 原始REPEAT C++实现：https://github.com/uowoolab/REPEAT

---

