# HISR - Hierarchical Invariant Sketch Resolution

HISR (层次化不变性网络测量框架) 是一个基于深度学习的不变性网络流测量系统。

## 项目概述

HISR通过结合图神经网络、不变性学习和传统Sketch技术，实现了在高噪声Sketch观测下的精确频率估计。项目采用层次化键空间建模和多环境不变学习策略，能够在不同网络条件下保持稳定的测量精度。

### 核心创新点

- **Prefix-Scale多尺度建模**: 将"尺度"定义为键空间的前缀粒度（/16→/24→/32），采用粗到细的Zoom-in策略
- **环境不变性学习**: 通过IRM（Invariant Risk Minimization）学习跨环境不变的恢复机制，提升OOD鲁棒性
- **热点聚焦机制**: 先定位粗粒度热点区域，再在热点内细化恢复，降低计算开销和碰撞不确定性
- **质量守恒解码**: 前缀树分裂比例确保父子节点质量守恒，保证非负性和一致性

## 项目结构

```
HISR/
├── HISR开发指引/                     # 主要开发文档和代码
│   ├── 关键脚本实现示例（不一定完美复刻报告中的模型，仅参考）/  # 核心算法实现
│   │   ├── prefix.py                 # Prefix层级处理模块
│   │   ├── bucketize.py              # Bucket分区模块
│   │   ├── local_operator.py         # 局部操作符和二部图提取
│   │   ├── encoder_bipartite.py      # 二部图编码器
│   │   ├── decoder_prefix_tree.py    # 前缀树解码器
│   │   ├── train_eval_hisr.py        # 完整训练评估脚本
│   │   ├── hisr_integrated.ipynb     # 整合版Jupyter Notebook ⭐
│   │   ├── requirements.txt          # Python依赖
│   │   ├── 环境配置指南.md            # 详细环境配置说明
│   │   └── 代码脚本简介.txt          # 代码结构说明
│   ├── 参考代码仓库/
│   │   └── uclpe-sketch-master/      # UCL参考实现
│   └── 主要参考工作/                 # 相关论文文档
├── README.md                        # 本文件
└── run_hisr_experiment.py          # 项目入口脚本
```

## 核心特性

- **层次化键空间建模**: 支持多级前缀粒度 (/16, /24, /32)
- **环境不变表征学习**: 通过IRM学习跨环境不变特征
- **二部图神经网络**: Counter-Key交替消息传递
- **质量守恒解码**: 层级化质量分裂保证守恒约束
- **自监督训练**: 基于测量一致性的自监督学习

## 快速开始

### 方式1: 使用整合版Notebook (推荐) ⭐

直接打开并运行 `HISR开发指引/关键脚本实现示例（不一定完美复刻报告中的模型，仅参考）/hisr_integrated.ipynb`

```bash
cd "HISR开发指引/关键脚本实现示例（不一定完美复刻报告中的模型，仅参考）"
jupyter notebook hisr_integrated.ipynb
```

Notebook包含完整的10步实验流程：
1. 环境设置和依赖检查
2. 路径配置
3. 数据文件检查
4. 配置文件生成
5. 模块导入测试
6. 训练执行
7. 结果验证
8. 批量实验配置
9. 可视化支持
10. 结果汇总

优势：
- ✅ 无需配置外部依赖
- ✅ 包含所有核心模块
- ✅ 提供完整演示
- ✅ 交互式学习
- ✅ 自动化实验流程

### 方式2: 使用独立训练脚本

```bash
cd "HISR开发指引/关键脚本实现示例（不一定完美复刻报告中的模型，仅参考）"
python train_eval_hisr.py \
  --data network \
  --data_path /path/to/your/dataset.dat \
  --break_number 1000000
```

**注意**: 需要先配置UCL-sketch路径，详见"环境配置"章节。

### 方式3: 使用项目入口

```bash
python run_hisr_experiment.py
```

## 环境配置

### 基础环境

```bash
# 创建虚拟环境（推荐）
python -m venv hisr_env
source hisr_env/bin/activate  # Linux/macOS
# 或 hisr_env\Scripts\activate  # Windows

# 安装基础依赖
pip install torch numpy tqdm
```

### 完整环境（用于train_eval_hisr.py）

```bash
# 安装所有依赖
pip install -r "HISR开发指引/关键脚本实现示例（不一定完美复刻报告中的模型，仅参考）/requirements.txt"

# 配置UCL-sketch路径（如需使用）
export PYTHONPATH="${PYTHONPATH}:$(pwd)/HISR开发指引/参考代码仓库/uclpe-sketch-master"
```

详细配置说明请参考: `HISR开发指引/关键脚本实现示例（不一定完美复刻报告中的模型，仅参考）/环境配置指南.md`

## 核心模块说明

### 数据流

1. **prefix.py**: 
   - 实现键的层次化前缀表示
   - 支持u64/IPv4等键类型
   - 提供Prefix Hierarchy管理

2. **bucketize.py**:
   - 将候选键分区为逻辑桶
   - 支持sorted/hash/prefix三种策略
   - 提供稳定的键-桶映射

3. **local_operator.py**:
   - 提取桶局部二部图 (Key ↔ Counter)
   - 实现测量操作符 A_{e,b}
   - 支持稀疏图表示

### 模型组件

4. **encoder_bipartite.py**:
   - 二部图GNN编码器
   - T层交替消息传递 (Counter→Key, Key→Counter)
   - 输出不变表征 z_c 和特异表征 z_v

5. **decoder_prefix_tree.py**:
   - 基于前缀树的层次化解码器
   - 质量守恒分裂机制
   - 确保非负性和守恒性

### 训练框架

6. **train_eval_hisr.py**:
   - 完整的训练+评估管道
   - 多环境采样和IRM训练
   - 三阶段分层处理 (L1/L2/L3)

## 损失函数

- **Measurement Loss**: `||A x̂ - y||²` - 测量一致性
- **Sparsity Loss**: `Σ log(1+|x̂|/ε)` - 稀疏性促进
- **IRM Penalty**: `Σ ||∇_w R_e||²` - 不变性正则化
- **Invariant Alignment**: `Var(z_c)` - 跨环境对齐

## 参数说明

### 模型参数
- `d_node`: 节点嵌入维度 (默认128)
- `d_msg`: 消息嵌入维度 (默认128)
- `d_z`: 不变表征维度 (默认128)
- `num_layers`: GNN层数 (默认3)

### 数据参数
- `bucket_len_stage1/2/3`: 各阶段桶长度
- `topk_l1/l2`: 热点prefix选择数量
- `break_number`: 最大处理数据量

### 训练参数
- `lr`: 学习率 (默认1e-3)
- `train_steps`: 训练步数 (默认2000)
- `irm_lambda`: IRM系数 (默认1.0)
- `inv_lambda`: 不变性系数 (默认0.1)
- `sparse_lambda`: 稀疏性系数 (默认0.01)

## 核心算法流程

### 整体流程

```
输入: Sketch观测 y, 候选键集 K
  ↓
1. Prefix层次化 → /16, /24, /32 分组
  ↓
2. Bucket分区 → K → {B_1, ..., B_m}
  ↓
3. 对每个Bucket b:
   a. 提取二部图 G_b = (U_b, V_{e,b}, E_b)
   b. 编码: z_c, z_v = Φ_θ(G_b)
   c. 解码: x̂_b = D_θ(z_c, Tree_b)
  ↓
4. 多环境IRM训练 → 最优参数 θ*
  ↓
输出: 频率估计 x̂
```

### 分层Zoom-in流程

HISR采用三阶段分层恢复策略（Path-A）：

```
阶段1: L1 (/16) - 粗粒度聚合
  ↓
目标: 估计各/16前缀的聚合流量
  ↓
操作: 在所有候选键上运行HISR，输出Top-K热点/16前缀
  ↓
阶段2: L2 (/24) - 中粒度细化
  ↓
目标: 在热点/16内估计各/24前缀流量
  ↓
操作: 仅在选定的热点/16对应的候选键集合上运行HISR，输出Top-K热点/24
  ↓
阶段3: L3 (/32) - 细粒度恢复
  ↓
目标: 在热点/24内恢复具体IP（/32）的精确频率
  ↓
操作: 仅在热点/24对应的候选键集合上运行HISR，输出最终频率估计
```

**优势**:
- 显式缩小候选集合，降低求解复杂度
- 聚焦热点区域，提升计算效率
- 粗粒度信号强，不易被哈希碰撞淹没
- 逐级细化，减少误差累积

## 评估指标

### 基础指标
- **AAE**: Average Absolute Error (平均绝对误差)
- **ARE**: Average Relative Error (平均相对误差)
- **WMRD**: Weighted Mean Relative Difference (加权相对差异)
- **EAE**: Entropy Absolute Error (熵绝对误差)

### 层次化指标
- **L1 Error**: /16前缀粒度的估计误差
- **L2 Error**: /24前缀粒度的估计误差
- **L3 Error**: /32具体IP粒度的估计误差

### 热点性能指标
- **Hotspot Recall@K**: Top-K热点前缀的召回率
- **候选覆盖率**: Zoom-in后实际处理的候选键比例

### OOD泛化指标
- **OOD Gap**: ID与OOD场景下的误差相对增幅
- **环境间方差**: 多环境输出的一致性度量

## 使用示例

### 基础演示 (无需外部数据)

```python
# 在hisr_integrated.ipynb中运行
from prefix import PrefixHierarchy
from bucketize import build_buckets

# 创建示例keys
keys = [i.to_bytes(8, 'little') for i in range(100)]

# 初始化hierarchy
hierarchy = PrefixHierarchy(levels_bits=(16, 24, 32))

# 创建buckets
bucket_index = build_buckets(keys, bucket_len=32, strategy="prefix")
```

### 完整训练（三阶段Zoom-in）

```bash
# 阶段1: L1粗粒度训练
python train_eval_hisr.py \
  --data network \
  --data_path data/network.dat \
  --break_number 100000 \
  --bucket_len_stage1 2048 \
  --lr 1e-3 \
  --train_steps 5000 \
  --device cuda

# 阶段2: L2中粒度训练（在热点/16上）
python train_eval_hisr.py \
  --data network \
  --data_path data/network.dat \
  --break_number 100000 \
  --bucket_len_stage2 1024 \
  --topk_l1 10 \
  --lr 1e-3 \
  --train_steps 5000 \
  --device cuda

# 阶段3: L3细粒度训练（在热点/24上）
python train_eval_hisr.py \
  --data network \
  --data_path data/network.dat \
  --break_number 100000 \
  --bucket_len_stage3 512 \
  --topk_l2 20 \
  --lr 1e-3 \
  --train_steps 5000 \
  --device cuda
```

### 多环境训练（IRM）

```bash
python train_eval_hisr.py \
  --data network \
  --data_path data/network.dat \
  --num_envs 5 \
  --irm_lambda 1.0 \
  --inv_lambda 0.1 \
  --train_steps 10000 \
  --device cuda
```

## 实验设置

### 研究问题

1. **RQ1 (多尺度有效性)**: 在键空间具备层级结构的情况下，将恢复目标按前缀粒度分层是否能显著提高长尾与OOD条件下的估计稳定性？

2. **RQ2 (Zoom-in效率)**: 先在粗粒度定位热点区域，再在热点内细化恢复，是否能在相同空间预算下提升精度/计算效率比？

3. **RQ3 (不变学习收益)**: 当哈希视角或键空间阶段发生变化时，IRM-style的跨环境约束是否能减少模型对碰撞模式的依赖，从而降低OOD误差增幅？

### 实验配置

#### 环境定义
- **哈希视角**: 不同hash seed/view诱导不同测量算子
- **键空间阶段**: 按前缀集合划分训练/测试集
- **采样窗口**: 按时间段落划分（可选）

#### OOD场景
- **ID**: 训练/测试使用相同seed和前缀分布
- **OOD-seed**: 测试切换到未见hash seed
- **OOD-prefix-phase**: 测试引入未见前缀集合

### 对比方法

- **UCL-sketch**: inverseNet + 自监督重建（强学习基线）
- **传统Sketch**: CM-sketch, C-sketch, Elastic, UnivMon, Nitro
- **HISR (ours)**: Prefix-scale分层 + 多环境不变性

### 消融实验

- **Flat**: 移除前缀树，直接恢复目标尺度
- **no-IRM**: 移除IRM不变性项
- **only-L1/L2**: 仅做粗粒度聚合，不进行细化
- **不同前缀配置**: 更换层级粒度
- **不同K值**: 改变热点预算

## 输出产物

训练完成后，项目将生成以下输出：

```
关键脚本实现示例/
├── checkpoints/              # 模型权重文件
│   ├── encoder_l1.pth
│   ├── encoder_l2.pth
│   └── encoder_l3.pth
├── results/                  # 评估结果
│   ├── metrics.json          # 各场景和尺度下的指标
│   ├── curves.csv            # 误差-开销曲线
│   └── training_log.txt      # 训练日志
└── plots/                    # 可视化图表
    ├── error_comparison.pdf  # 误差对比图
    ├── hotspot_recall.pdf    # 热点召回率
    ├── scale_decomposition.pdf # 尺度分解图
    └── ood_gap.pdf           # OOD泛化能力
```

## 故障排除

### UCL-sketch导入失败
```bash
# 检查路径
ls "HISR开发指引/参考代码仓库/uclpe-sketch-master"

# 设置PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/HISR开发指引/参考代码仓库/uclpe-sketch-master"
```

### CUDA内存不足
```bash
# 使用CPU或减小batch
python train_eval_hisr.py --device cpu --bucket_len_stage1 1024
```

### 训练不收敛
```bash
# 降低学习率
python train_eval_hisr.py --lr 1e-4 --irm_lambda 0.5
```

## 文档

- **代码简介**: `HISR开发指引/关键脚本实现示例（不一定完美复刻报告中的模型，仅参考）/代码脚本简介.txt`
- **环境配置**: `HISR开发指引/关键脚本实现示例（不一定完美复刻报告中的模型，仅参考）/环境配置指南.md`
- **整合教程**: `HISR开发指引/关键脚本实现示例（不一定完美复刻报告中的模型，仅参考）/hisr_integrated.ipynb`
- **实验流程**: `HISR开发指引/HISR实验流程.ipynb`
- **实验方案**: `HISR开发指引/01 HISR实验方案_基于UCL数据集.docx`

## 技术栈

- **深度学习**: PyTorch >= 2.0
- **数值计算**: NumPy >= 1.21
- **科学计算**: SciPy >= 1.7
- **机器学习**: scikit-learn >= 1.0
- **进度条**: tqdm >= 4.60
- **数据格式**: bytes/int64
- **可视化**: matplotlib (可选)
- **数据处理**: pandas (可选)

## 引用

如使用本项目，请引用相关论文（具体信息待补充）。

## 许可证

[待添加许可证信息]

## 贡献指南

欢迎提交Issue和Pull Request！

## 联系方式

[待添加联系信息]