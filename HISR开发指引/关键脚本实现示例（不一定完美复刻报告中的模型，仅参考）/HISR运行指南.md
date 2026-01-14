<
![CDATA[
# HISR 项目运行指南

## 📋 项目概述

HISR (Hierarchical Invariant Sketch Resolution) 是一个基于图神经网络的数据流重构方法，用于从 Sketch 结构中恢复原始键值频率。

## 🔧 环境要求

### 系统要求
- Python >= 3.8 (推荐 3.9+)
- macOS/Linux/Windows

### 依赖包安装

```bash
# 安装基础依赖
pip install torch>=2.0.0 numpy>=1.21.0 tqdm>=4.60.0

# 安装模型相关依赖
pip install timm>=0.6.0 transformers>=4.20.0

# 安装数据处理依赖
pip install ftfy>=6.0.0 regex>=2.0.0 tokenizers>=0.13.0

# 安装UCL项目依赖
pip install scipy scikit-learn bitarray
```

或使用 requirements.txt 安装：

```bash
pip install -r requirements.txt
```

## 📁 项目目录结构

```
HISR开发指引/关键脚本实现示例（不一定完美复刻报告中的模型，仅参考）/
├── hisr/                          # HISR 核心模块
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── bucketize.py          # 分桶逻辑
│   │   ├── local_operator.py     # 局部算子（图提取）
│   │   └── prefix.py             # 前缀层次结构
│   └── model/
│       ├── __init__.py
│       ├── encoder_bipartite.py  # 二部图编码器
│       └── decoder_prefix_tree.py # 前缀树解码器
├── ucl_refs/                     # UCL 参考代码
│   ├── __init__.py
│   ├── load_data.py              # 数据加载
│   ├── hash_function.py          # 哈希函数
│   ├── ucl_sketch.py             # UCL Sketch
│   ├── cm_sketch.py              # CM Sketch
│   ├── bloom_filter.py           # 布隆过滤器
│   ├── heavy_filter.py           # 重元素过滤器
│   ├── common.py                 # 通用工具
│   └── mertrics.py               # 评估指标
├── train_eval_hisr.py            # 主训练脚本
└── requirements.txt              # 依赖列表
```

## 🚀 快速开始

### 1. 准备数据

确保数据集位于正确路径：

```bash
cd "HISR开发指引/关键脚本实现示例（不一定完美复刻报告中的模型，仅参考）"
```

Kosarak 数据集应该位于：
```
../参考代码仓库/uclpe-sketch-master/data/kosarak.dat
```

### 2. 基础运行（小规模测试）

```bash
python3 train_eval_hisr.py \
  --data kosarak \
  --data_path "../参考代码仓库/uclpe-sketch-master/data/kosarak.dat" \
  --break_number 5000 \
  --train_steps 50 \
  --num_samples 5 \
  --interval 200
```

### 3. 中等规模运行

```bash
python3 train_eval_hisr.py \
  --data kosarak \
  --data_path "../参考代码仓库/uclpe-sketch-master/data/kosarak.dat" \
  --break_number 10000 \
  --train_steps 100 \
  --num_samples 10 \
  --interval 100
```

### 4. 大规模运行

```bash
python3 train_eval_hisr.py \
  --data kosarak \
  --data_path "../参考代码仓库/uclpe-sketch-master/data/kosarak.dat" \
  --break_number 100000 \
  --train_steps 2000 \
  --num_samples 128 \
  --interval 1000
```

## ⚙️ 参数说明

### 数据参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data` | 数据集类型 (kosarak/retail/network/synthetic) | network |
| `--data_path` | 数据文件路径 | 必需 |
| `--key_size` | 键的字节大小 | 8 |
| `--break_number` | 使用的数据项数量 | 1000000 |

### Sketch 参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--slot_num` | Heavy Filter 槽数量 | 20000 |
| `--width` | CM Sketch 宽度 | 20000 |
| `--depth` | CM Sketch 深度 | 3 |
| `--bf_width` | 布隆过滤器宽度 | 50000 |
| `--bf_hash` | 布隆过滤器哈希函数数 | 3 |

### 采样参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--interval` | 采样间隔 | 1000 |
| `--num_samples` | 采样数量 | 128 |

### 训练参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--device` | 设备 (cuda/cpu) | 自动检测 |
| `--num_envs` | 环境数量 | 3 |
| `--train_steps` | 训练步数 | 2000 |
| `--lr` | 学习率 | 0.001 |
| `--irm_lambda` | IRM 惩罚系数 | 1.0 |
| `--inv_lambda` | 不变对齐系数 | 0.1 |
| `--sparse_lambda` | 稀疏正则化系数 | 0.01 |

### 分桶参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--bucket_len_stage1` | Stage1 分桶大小 | 2048 |
| `--bucket_len_stage2` | Stage2 分桶大小 | 1024 |
| `--bucket_len_stage3` | Stage3 分桶大小 | 512 |
| `--topk_l1` | Stage1 Top-K | 256 |
| `--topk_l2` | Stage2 Top-K | 512 |

## 🔍 运行输出说明

### 阶段输出

1. **数据加载**：显示读取的数据项总数
   ```
   Successfully read in 8019015 items.
   ```

2. **环境构建**：构建多个 sketch 视图
   ```
   [HISR] Inserting traces (ref view)
   [HISR] Inserting traces (view s=1)
   [HISR] Inserting traces (view s=2)
   ```

3. **训练阶段**：
   ```
   [HISR] Training L1_all
   ```
   - L1_all: 所有 CM 键的全局扫描
   - L2: 热点 /16 前缀子网
   - L3: 热点 /24 前缀子网

4. **训练日志**（每200步）：
   ```
   step=200 loss=0.1234 risk=0.1123 irm=0.0011 inv=0.0022
   ```

5. **评估结果**：
   ```
   [HISR] Stage=L1_all metrics={'AAE': 1.23, 'ARE': 0.45, 'WMRD': 0.67, 'EAE': 0.89}
   ```

### 评估指标说明

- **AAE (Average Absolute Error)**: 平均绝对误差
- **ARE (Average Relative Error)**: 平均相对误差
- **WMRD (Weighted Mean Relative Difference)**: 加权平均相对差异
- **EAE (Entropy Absolute Error)**: 熵绝对误差

## ⚠️ 常见问题及解决方案

### 1. 模块导入错误

**问题**：`ModuleNotFoundError: No module named 'hisr.data.bucketize'`

**解决**：确保目录结构正确，执行以下命令：

```bash
cd "HISR开发指引/关键脚本实现示例（不一定完美复刻报告中的模型，仅参考）"
mkdir -p hisr/data hisr/model
touch hisr/__init__.py hisr/data/__init__.py hisr/model/__init__.py

cp bucketize.py hisr/data/
cp local_operator.py hisr/data/
cp prefix.py hisr/data/
cp encoder_bipartite.py hisr/model/
cp decoder_prefix_tree.py hisr/model/
```

### 2. UCL 依赖模块缺失

**问题**：`ModuleNotFoundError: No module named 'ucl_refs.load_data'`

**解决**：创建 ucl_refs 目录并复制依赖：

```bash
mkdir -p ucl_refs
touch ucl_refs/__init__.py

cp "../参考代码仓库/uclpe-sketch-master/load_data.py" ucl_refs/
cp "../参考代码仓库/uclpe-sketch-master/Sketching/hash_function.py" ucl_refs/
cp "../参考代码仓库/uclpe-sketch-master/UCL_sketch/ucl_sketch.py" ucl_refs/
cp "../参考代码仓库/uclpe-sketch-master/Sketching/cm_sketch.py" ucl_refs/
cp "../参考代码仓库/uclpe-sketch-master/Sketching/bloom_filter.py" ucl_refs/
cp "../参考代码仓库/uclpe-sketch-master/UCL_sketch/heavy_filter.py" ucl_refs/
cp "../参考代码仓库/uclpe-sketch-master/Utils/common.py" ucl_refs/
cp "../参考代码仓库/uclpe-sketch-master/Utils/mertrics.py" ucl_refs/
```

然后修改导入路径：

```bash
cd ucl_refs
sed -i '' 's/from Utils\.common import/from .common import/' cm_sketch.py bloom_filter.py heavy_filter.py
sed -i '' 's/from Sketching\.hash_function import/from .hash_function import/' cm_sketch.py bloom_filter.py
```

### 3. Sketch 宽度不匹配错误

**问题**：`ValueError: all the input array dimensions except for the concatenation axis must match exactly`

**原因**：CM Sketch 初始化时使用 `calNextPrime` 调整宽度，但数组初始化使用了原始配置值

**解决**：已在 `train_eval_hisr.py` 中修复，使用 `ref.cm.width` 而不是 `cfg.width`

### 4. 整数溢出错误

**问题**：`RuntimeError: value cannot be converted to type int64_t without overflow`

**原因**：哈希函数中的整数乘法溢出

**解决**：已在 `encoder_bipartite.py` 中修复，使用 float64 类型进行计算

### 5. 数据文件未找到

**问题**：`FileNotFoundError` 或数据读取失败

**解决**：检查数据路径是否正确：

```bash
ls -la "../参考代码仓库/uclpe-sketch-master/data/"
```

确保 kosarak.dat 文件存在

## 📊 性能优化建议

### 1. 使用 GPU 加速

如果系统有 NVIDIA GPU：

```bash
# 安装 CUDA 版本的 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 脚本会自动检测并使用 GPU
```

### 2. 调整数据集大小

- **快速测试**：`--break_number 5000`
- **中等规模**：`--break_number 50000`
- **完整数据集**：`--break_number 1000000`（kosarak 完整数据约 990K 行）

### 3. 调整训练步数

- **调试**：`--train_steps 10`
- **快速验证**：`--train_steps 100`
- **正常训练**：`--train_steps 2000`
- **充分训练**：`--train_steps 10000`

## 🐛 调试技巧

### 1. 检查模块导入

```python
python3 -c "from hisr.data.bucketize import BucketIndex; print('OK')"
python3 -c "from hisr.model.encoder_bipartite import BipartiteGNNEncoder; print('OK')"
python3 -c "from ucl_refs.load_data import readTraces; print('OK')"
python3 -c "from ucl_refs.ucl_sketch import UCLSketch; print('OK')"
```

### 2. 验证数据加载

```python
from ucl_refs.load_data import readTraces
size, traces = readTraces("../参考代码仓库/uclpe-sketch-master/data/kosarak.dat", "kosarak")
print(f"Loaded {size} items")
```

### 3. 测试 Sketch

```python
from ucl_refs.ucl_sketch import UCLSketch
sketch = UCLSketch(20000, 20000, 3, 50000, 3, 8, decode_mode="CM")
sketch.insert(b'test_key', 5)
print(f"Query result: {sketch.query(b'test_key')}")
```

## 📝 开发笔记

### 关键设计要点

1. **环境定义**：`e = (τ, s, π)` - 时间窗口、哈希种子、键空间阶段
2. **三阶段分桶策略**：
   - Stage 1 (L1): 全局扫描，所有 CM 键
   - Stage 2 (L2): 热点 /16 前缀子网
   - Stage 3 (L3): 热点 /24 前缀子网
3. **损失函数**：测量一致性 + IRM 惩罚 + 不变对齐 + 稀疏正则化
4. **解码器**：质量守恒的层次化前缀树分裂

### 已知限制

1. **内存使用**：大桶大小会增加内存消耗
2. **训练速度**：CPU 训练较慢，建议使用 GPU
3. **数据集格式**：目前支持 kosarak、retail 等格式

## 📚 参考资料

- UCL-Sketch: `../参考代码仓库/uclpe-sketch-master/`
- 论文: `HISR构建报告V5.pdf`
- 代码注释: 各源文件头部注释

## 🆘 获取帮助

如遇到问题，请检查：
1. Python 版本 >= 3.8
2. 所有依赖包已安装
3. 目录结构正确
4. 数据文件路径正确

---

**最后更新**: 2026-01-15
**版本**: 1.0