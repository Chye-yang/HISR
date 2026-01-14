## Notebook 思路（`experiment_demo.ipynb`）

下面为 `experiment_demo.ipynb` 的核心流程与实现思路，便于快速理解和重复实验：

- **目标**：在若干数据集上（网络流、Retail、Kosarak、Zipf 合成）比较 UCL-sketch 与多种基线（CM-sketch、C-sketch、Elastic、UnivMon、Nitro 等）的频率估计性能。评估指标包括 AAE、ARE、WMRD 等。
- **Step 1 — 导入与依赖**：加载 `load_data.readTraces`、各类 sketch 实现（位于 `Sketching/`）、训练工具 `Utils.training.learningSolver` 与 `UCL_sketch.ucl_sketch.UCLSketch`。
- **Step 2 — 参数配置**：通过 argparse 设置数据路径、哈希/桶参数、采样间隔 `--interval`、样本数 `--num_samples`、训练/保存选项与随机种子等。
- **Step 3 — 初始化**：构建 `UCLSketch` 与一组基线 sketch（CM、C、Nitro、Elastic、UnivMon 等），并打印内存/尺寸信息以便对齐空间预算。
- **Step 4 — 插入数据流并采样**：遍历 trace，将每个键插入 `ucl_sketch` 并统计 ground truth（字典形式）；在末端按间隔位置采样 `ucl_sketch.get_current_state()`，将采样状态收集到 `samples`（形状为 [num_samples, hash_num, bucket_dim]）。
- **Step 5 — 保存 ground-truth**：将完整的 `ground_truth` 写入根目录 `ground_truth.txt`，并按批（默认 5000 条）拆分写入 `ground_truth_files/ground_truth_*.txt`。
- **Step 6 — 导出采样快照**：将 `samples` 原始保存为 `export/cm_samples.npy`，扁平化后保存为 `export/cm_samples.csv`，并写入 `export/cm_samples_meta.json`（包含样本数、hash_num、bucket_dim）。
- **Step 7 — 理想学习准备**：按频率排序将低频键与高频键分离（用于“理想学习”对比实验），分别插入到各个基线 sketch，以得到基线的参考状态。
- **Step 8 — 评估与绘图（Notebook 下游）**：基于保存的样本与 ground-truth 计算 AAE/ARE/WMRD、绘制对比曲线，并保存或导出评估结果（Notebook 中含示例代码与可视化）。
- **关键输出文件**：`ground_truth.txt`、`ground_truth_files/ground_truth_*.txt`、`export/cm_samples.npy`、`export/cm_samples.csv`、`export/cm_samples_meta.json`，以及 `network_flow_output.csv/.json`（由 `convert_flow.py` 生成）。

可直接在 Notebook 中逐格执行以复现实验，或改写为脚本化流程以便批量跑参。

- 🪐 一个简洁的 PyTorch 实现（目录: UCL_sketch）。
- ⚡️ 预处理好的 13 字节长的 5 元组网络包数据切片。
- 💥 一个自包含的 Jupyter notebook（`experiment_demo.ipynb`），用于运行与评估多种 sketch 算法：如 CM-sketch、C-sketch、理想学习版 CM-sketch、理想学习版 C-sketch、Univmon、Elastic Sketch、NitroSketch、SeqSketch 以及我们的 UCL-sketch。
- 🛸 其它实用函数与文档，例如用于评估的 WMRD（加权平均相对差）等指标。

## 环境搭建

首先，克隆并进入仓库：

```bash
git clone https://github.com/Y-debug-sys/UCL-sketch.git
cd UCL-sketch
```

仓库提供了 `environment.yml`，可用来创建 Conda 环境：

```bash
快速查看示例（在仓库根目录运行）：
conda activate UCL-sketch
```

## 运行示例

仓库中提供了一个运行脚本 `main.py`，可用于在提供的 IP trace 上训练 UCL-sketch，同时也便于扩展到其他流式数据集：例如 `Kosarak` 与 `Retail`。下载数据并将 `.dat` 文件放到 `data/` 目录后，可按如下方式运行：

```bash
python main.py --config_path ./configs/{your_config_name}.yaml --data_path ./data/{your_dataset_name}.dat --ckpt ./checkpoints --data network
```

此外，你也可以通过合成的 Zipf 分布数据做实验：

```bash
python main.py --config_path ./configs/{your_config_name}.yaml --skewness {your_skew_value} --ckpt ./checkpoints --data synthetic
```

关于与基线方法的对比与评估（AAE、ARE、WMRD 等），请参阅我们的 Jupyter 示例（`run_sketches.ipynb`）。

## Sketch 导出数据集

以下是在本仓库中由 sketch 流程或示例 notebook 导出的数据文件及其位置：

- `network_flow_output.csv` — 由 `convert_flow.py` 生成的 CSV 格式流量样本文件，路径: [network_flow_output.csv](network_flow_output.csv)
- `network_flow_output.json` — 由 `convert_flow.py` 生成的 JSON 格式流量样本文件，路径: [network_flow_output.json](network_flow_output.json)
- `ground_truth.txt` — 单一的 ground-truth 文件（整体），路径: [ground_truth.txt](ground_truth.txt)
- `ground_truth_files/` — 存放分割的 ground-truth 子文件（`ground_truth_1.txt` 至 `ground_truth_19.txt`），路径: [ground_truth_files](ground_truth_files)

快速查看示例（在仓库根目录运行）：

```bash
head -n 8 network_flow_output.csv
head -n 8 ground_truth.txt
ls -1 ground_truth_files | head -n 20
```

说明：这些文件的生成逻辑位于 `convert_flow.py`（参见对应行）与示例 notebook `experiment_demo.ipynb` 中。

## 使用导出样本进行学习

下面说明如何把在运行 `experiment_demo.ipynb`（或 `main.py`）中采样到的 sketch 状态用于训练模型，以及训练后如何在 UCL-sketch 中推断并查询频率：

- **样本格式**：采样保存为 NumPy 数组，形状为 `(num_samples, depth, width)`（在本仓库中默认为 `samples`），并可导出为 `export/cm_samples.npy` / `export/cm_samples.csv`，其元信息保存在 `export/cm_samples_meta.json`。
- **矩阵与索引**：从当前 `UCLSketch` 获取压缩感知矩阵与索引：调用 `A, index = ucl_sketch.get_current_state(return_A=True)`。其中 `A` 为密集矩阵（用于训练时重建），`index` 用于标记样本中对应的键位置。
- **数据加载**：训练器 `Utils.training.learningSolver.train(sketchShots, phiMatrix, index)` 接收三个输入：
	- `sketchShots`：即上面形状的 `samples`（NumPy 数组），由 `load_data.sketchDataset` 封装为 PyTorch 的 DataLoader（默认 `batch_size` 来自配置）。
	- `phiMatrix`：即 `A`，用于把模型输出映射回 sketch 空间以计算重建损失。
	- `index`：键的位置信息，用于内部转换与数据增强（`transform`）。
- **训练细节**：
	- 模型：`Utils.net_params.inverseNet`（或 `inverseNet_ablation`），将 sketch 快照映射到键频率空间。
	- 优化器：Adam，初始学习率默认 `0.001`（见 `configs/config.yaml`）。
	- 损失：MSE（重建误差），并在训练中结合对自回归输出的正则项与重建后的自监督损失（见 `Utils/training.py`）。
	- 训练轮数、早停、批大小等超参从 `configs/config.yaml` 或通过 `main.py` 的 `Model_Args` 配置传入（默认 `train_epochs: 300, patience: 30, batch_size: 32`）。
	- 数据增强：训练时会对模型输出做归一（instance norm）与局部扰动（`transform`），以提高鲁棒性。
	- 模型检查点：训练过程使用早停并把最佳参数保存到配置的 `checkpoints` 目录（默认由 `main.py` 的 `--ckpt` 指定）。
- **训练调用示例（来自 `main.py`）**：

```python
# 在插入并采样后：
samples = np.empty([0, ucl_sketch.cm.depth, ucl_sketch.cm.width])
# ... 插入流并按间隔采样到 samples ...
A, index = ucl_sketch.get_current_state(return_A=True)
solver = learningSolver(model_args, A.shape[1])
solver.train(samples, A, index)
```

- **推断与查询**：训练好模型后，可用 `solver.test(test_sample)` 得到恢复的键频率向量（float），常对结果取整并作为查询参数传入 `ucl_sketch.query(key, x)`，以结合 heavy filter 与 CM 部分输出得到最终估计。

```python
test_sample = ucl_sketch.get_current_state(return_A=False)
pred_y = solver.test(test_sample)
pred_x = np.ceil(pred_y.squeeze()).astype(np.int32)
ans = ucl_sketch.query(key, pred_x)
```




