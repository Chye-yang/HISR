# HISR项目使用指南

## 📋 项目概述

HISR (Hierarchical Invariant Sketch Resolution) 是一个基于多层不变学习的网络流量测量框架，实现了论文中的核心算法。

## 🏗️ 项目结构

```
关键脚本实现示例（不一定完美复刻报告中的模型，仅参考）/
├── train_eval_hisr.py         # 主训练评估脚本
├── bucketize.py               # 逻辑分桶模块
├── encoder_bipartite.py       # 二分图编码器
├── decoder_prefix_tree.py     # 前缀树解码器
├── prefix.py                  # 层次化键空间
├── local_operator.py          # 桶本地算子
├── 代码脚本简介.txt           # 详细设计文档
└── 演示文件/
    ├── HISR实验流程.ipynb     # Jupyter Notebook
    ├── run_hisr_experiment.py # 命令行脚本
    └── demo_hisr_minimal.py   # 最小演示
```

## 🚀 快速开始

### 方法1: 使用命令行脚本（推荐）
```bash
# 检查环境
python run_hisr_experiment.py --check_only

# 运行小型实验
python run_hisr_experiment.py --break_number 50000 --train_steps 50

# 运行完整实验
python run_hisr_experiment.py --break_number 100000 --train_steps 100
```

### 方法2: 使用Jupyter Notebook
```bash
jupyter notebook HISR实验流程.ipynb
```

### 方法3: 最小功能测试
```bash
python demo_hisr_minimal.py
```

## 🔧 核心模块说明

### 1. 数据预处理模块
- **prefix.py**: 层次化前缀处理（/16 → /24 → /32）
- **bucketize.py**: 逻辑分桶策略（三种分桶方式）
- **local_operator.py**: 桶本地二分图算子提取

### 2. 神经网络模型
- **encoder_bipartite.py**: 二分图消息传递编码器
- **decoder_prefix_tree.py**: 质量守恒前缀树解码器

### 3. 训练框架
- **train_eval_hisr.py**: 多环境训练与评估入口
- 支持IRM惩罚、不变量对齐等先进训练策略

## ⚙️ 配置参数

主要配置参数（可通过命令行或配置文件修改）：
- `--break_number`: 处理的数据包数量
- `--train_steps`: 训练步数  
- `--bucket_len_stage1`: 第一阶段桶大小
- `--topk_l1`: L1层级Top-K选择
- `--num_envs`: 环境数量
- `--irm_lambda`: IRM惩罚系数

## 📊 输出结果

训练完成后会生成：
- `checkpoints/`: 模型检查点
- `training_log.txt`: 训练日志
- 性能指标输出（AAE、ARE、WMRD等）

## 🎯 实验流程

1. **环境检查**: 验证依赖包和路径设置
2. **数据加载**: 自动检测可用数据集
3. **模块测试**: 验证所有核心模块可正常导入
4. **训练执行**: 多环境HISR训练
5. **结果验证**: 检查输出文件和性能指标

## 🔍 故障排除

### 常见问题
1. **模块导入错误**: 检查Python路径设置
2. **数据文件缺失**: 使用合成数据模式 (`--data synthetic`)
3. **内存不足**: 减小 `--break_number` 参数
4. **训练时间过长**: 减小 `--train_steps` 参数

### 调试建议
```bash
# 详细调试信息
python run_hisr_experiment.py --break_number 10000 --train_steps 10

# 如果遇到导入问题，先运行最小测试
python demo_hisr_minimal.py
```

## 📈 性能优化

- 对于大型数据集，适当增加 `--break_number`
- 调整分桶参数优化内存使用
- 使用GPU加速训练（如可用）

## 🤝 扩展开发

项目采用模块化设计，易于扩展：
- 新增编码器：继承 `BipartiteGNNEncoder`
- 新增解码器：继承 `PrefixTreeDecoder`  
- 新增损失函数：在 `train_eval_hisr.py` 中添加

## 📚 参考文献

- [HISR构建报告V5.pdf](00%20HISR构建报告V5.pdf)
- [代码脚本简介.txt](代码脚本简介.txt)
- UCL-sketch参考实现

---

**开始使用**: 选择上述任一方法开始您的HISR实验！