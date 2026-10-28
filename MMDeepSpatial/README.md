# MMDeepSpatial：E0 / E1 / E2 表示验证

本目录与 `HM_DeepSpital` 同级，全部新增代码独立存放，读取相同 MERFISH H5AD。
本次实现用户指定的**前三个表达表示实验（Phase 1）**，编号以本目录为准：

| Notebook | 表示和距离 | 问题 | 默认物理 GPU |
|---|---|---|---|
| `E0_Euclidean.ipynb` | 普通 MLP AE / 欧氏距离；可切换 PCA | 基础表示能否区分稀有细胞 | 7 |
| `E1_Hyperbolic.ipynb` | Poincaré 球 AE / 测地距离 | 负曲率约束及距离是否带来优势 | 5 |
| `E2_Joint.ipynb` | 欧氏 + Poincaré 联合 AE / 标定后的乘积距离 | 融合是否优于单几何 | 0 |

这是 MManiST 思路启发的最小 AE 对照，不是 MManiST 图网络/Pareto 优化的复现。
当前阶段产物是冻结编码器和表示评价；尚未改变 UOT、GiT、Flow Matching 或生成三维组织。
附带研究计划仅作为设计参考，未执行其中后续阶段或把“给智能体的约束”当作额外用户指令。

## 运行

在服务器已有 CUDA PyTorch 环境中安装本目录 `requirements.txt`。无需安装 DeepSpatial、scanpy、Lightning 或 POT。

```bash
cd /path/to/DeepSpatial/MMDeepSpatial
python -m pip install -r requirements.txt
python -m ipykernel install --user --name mmdeepspatial --display-name MMDeepSpatial
```

在 Jupyter 中给三个 notebook **各自选择独立 kernel**。各 notebook 第一格设置 `CUDA_VISIBLE_DEVICES`，之后程序内部统一用 `cuda`（即被映射后的 `cuda:0`）。
保留本项目已有物理卡配置 7/5/0；调度器若已分配设备，应在第一格修改 `GPU_ID`，或设为 `None` 保留调度器环境。
如果当前 kernel 已导入 torch，第一格主动报错，重启该 kernel 再从头运行。仅有三张 GPU 并不意味着应启用 DDP；本实现每组独立单卡。

三个 notebook 的配置区结构一致。默认：5 张 `merfish_0` 到 `merfish_4`；分别隐藏 1、2、3；每折 seeds **[42, 0, 1, 2, 3]**；100 epochs；总潜在维度 32；hidden 128；batch 512；稀有类训练丰度 **≤5%**。
每张卡依次运行该组的 15 个训练任务。可先在三组中同步改为 `HOLDOUTS=[2], SEEDS=[0], EPOCHS=2` 检查服务器，再使用新输出目录正式运行。

默认数据路径 `DeepSpatial/data/merfish_mouse_hypothalamus/**/merfish_[0-4].h5ad`。本地没有这份原始数据，运行时请确认路径。
文件按自然顺序排序；holdout 是排序后的零起始索引，配置格会打印对应文件。需要至少三张切片，留出只能是中间切片。
`LABEL_KEY='cell_class'` 必须是可信细胞类型注释，不能以空间域标签代替。没有标签的输入会直接报错。

默认从 `X` 读取非负表达，仅做 `log1p`，不擅自执行测序深度归一化。若输入已是 log 表达或标准化表达，三组均改 `TRANSFORM='identity'`；有原始 counts layer 时可设 `LAYER='counts'`。
训练切片决定交集基因、去零方差基因、可选方差筛选、均值和标准差。默认使用全部变量基因（MERFISH 靶向面板不默认筛 HVG）。测试缺失训练基因会报错，不用测试基因改写训练特征。

E0 默认 `E0_METHOD='ae'`，适合容量对照；设成 `'pca'` 可额外跑 PCA 基线，输出目录自动区分。PCA 使用 CPU，不会因为指定 GPU 而加速。
本轮映射修改参考了上一轮相同切片上的诊断，因此仍属于探索性实验。曲率和融合权重预先固定，不用隐藏切片早停或自动选参；最终评价需要独立切片/样本。多 seed 衡量优化波动，不能代替生物学重复。

## 模型和公平性

三组使用相同的 `genes → hidden → dim` 编码器和 `dim → hidden → genes` 解码器，AdamW、训练步数、shuffle 和初始化 seed 一致。
**同一输入基因数下，三组 AE 可训练参数量完全相同**。E0 全部 32 维欧氏，E1 全部 32 维双曲，E2 拆成 16 维欧氏 + 16 维双曲，共享 MLP 主干；没有把总维度或容量翻倍。
PCA 是计算开销参考，不声称与 AE 同容量。AE 不使用 cell type、稀有类重采样、空间位置或图；这次隔离表达表示机制。

E1/E2 的 MLP 双曲分支输出 `v` 先经过平滑径向压缩：
`t = v / sqrt(1 + c * ||v||² / L²)`，再经 geoopt `PoincareBall.expmap0(t)` 映入球内。
固定曲率 `-c`，默认 `c=1`，`L=max_tangent_radius=1.5`，三组 notebook 中统一设置。
该映射保留有限输入的径向次序与径向梯度，使 `sqrt(c)*||h|| < tanh(L) ≈ 0.905`，避免上一轮大量点被硬裁到半径 0.98。
原有 `0.98/sqrt(c)` 裁剪只保留为数值保护；不增加损失项或参数，E0 编码与训练目标保持原样。
新映射仍可能趋向自己的平滑上限，因此同时记录 `near_mapping_limit_fraction`（半径达到 `tanh(0.95*L)` 的比例），不能仅凭边界率为零声称径向信息已恢复。
解码器直接接收球内坐标，重建训练标准化表达；没有先 logmap 再解码而抵消流形映射，也不加入组别特定的解码缩放。
优化的是欧氏网络权重，因此使用普通 AdamW；没有可训练流形参数，不需要 RiemannianAdam。
损失仅为表达 MSE；本版不额外加入图或测地对比损失，所以结论范围是“球约束 AE 表示及其几何距离”，不是双曲图传播机制。
检查 `training.csv` 的 `raw_tangent_radius_mean/p95/max`、`ball_radius_fraction_median/p95/std`、`near_mapping_limit_fraction`，以及主汇总中的训练/测试半径统计，再结合重建误差和几何检索结果判断。新 checkpoint 保存映射配置；旧 checkpoint 缺少该字段时按原映射加载。

检索距离：E0 `dE/sE`；E1 `dH/sH`；E2 `sqrt(alpha*(dE/sE)^2+(1-alpha)*(dH/sH)^2)`，默认 `alpha=0.5`。
`sE/sH` 是训练切片随机跨切片配对距离的正值中位数，编码器训练后冻结。双曲距离使用 float64 的稳定解析式，并在测试中与 geoopt 距离核对。
距离和精确 top-k 分块计算，不保存整张 N×M 矩阵；计算时间仍为 O(Ntest×Ntrain)。可减小 query/reference chunk 降显存，不要以裁掉稀有类来加速。

## 评价与输出

每折只有可见切片用于表达 AE/PCA、预处理统计量、距离尺度和线性 probe 拟合。随后才打开隐藏 H5AD 评价。
线性 probe 是相同的 StandardScaler + LogisticRegression，固定 `C=1`、`class_weight='balanced'`；这种权重仅在 probe 使用，三组一致。
双曲 probe 使用原点切空间坐标，E2 拼接欧氏/切空间坐标。probe 未必能反映非线性几何优势，必须结合使用测地距离的检索指标。

- `rare_macro_auprc`：有训练支持且测试存在的稀有类型 AP 宏平均；同时保存各类型测试 prevalence。
- macro-F1、Recall、common macro-F1、整体 accuracy：统一 argmax 决策，不在测试上选阈值。
- retrieval hit@1/5/15、kNN purity@1/5/15：隐藏切片查询可见切片参考库。它们是类型一致性代理，不能解释为真实逐细胞配对正确率。
- coverage：每种类型在训练切片间的最近同类型距离 95% 分位数作为冻结半径，测试到训练同类型参考的最近距离是否在半径内。每类最多固定随机取 2048 个训练校准查询，不裁剪测试细胞。
- 单训练切片类型无法跨切片校准覆盖率，记 NaN 和原因；训练未见类型单列，present/absent、分母、可评价类型数全部保留。测试未见类型不被伪装成“稀有类成功”；整体 accuracy 仍计入其错误。
- train time、test encode time、evaluation time、训练峰值显存、参数量、重建 MSE，以及每 epoch 梯度范数、球边界占比、非有限批次数。

下一轮输出在 **`outputs/representations_v2/{E0_ae,E1_ae,E2_ae}/holdout_*/seed_*/`**；保留 `representations_v1` 全部结果：

```text
encoder.pt 或 pca.joblib       preprocessing.npz       geometry.json
probe.joblib                  probe_probabilities.npz
embeddings.npz                train_cells.csv         test_cells.csv
train_type_counts_by_slice.csv                         test_type_counts.csv
training.csv 或 pca_variance.csv                       test_predictions.csv
per_type.csv                  summary.csv
manifest.json                 environment.json        completed.json / failed.json
diagnostics_summary.csv       diagnostics_per_type.csv
diagnostic_probes.joblib       diagnostic_probabilities.npz
```

`diagnostics=True` 默认开启。所有诊断复用已训练的编码器，不增加编码器训练任务：

| arm / variant | 表示与距离 | probe |
|---|---|---|
| E0 / `main` | 原欧氏表示 | 主 probe |
| E0 / `l2_normalized` | L2 归一化表示＋欧氏距离，与余弦近邻排序一致 | 在归一化训练表示上拟合 |
| E1 / `main` | 双曲表示＋测地距离 | 主 probe |
| E1 / `angular` | 将同一个双曲嵌入归一化，去掉径向差异 | 在角度训练表示上拟合 |
| E2 / `main` | 16E＋16H，固定 alpha=0.5 | 联合主 probe |
| E2 / `euclidean_only` | 只用已有欧氏 16 维，等价于检索 alpha=1 | 单分支 probe |
| E2 / `hyperbolic_only` | 只用已有双曲 16 维，等价于检索 alpha=0 | 单分支切空间 probe |
| E2 / `alpha_0.25`, `alpha_0.75` | 同一联合表示，改变检索距离权重 | 保持联合主 probe 不变 |

所有诊断 probe 和距离尺度都在打开测试切片前拟合。分支对照用于分析已有编码器的信息分配，不等同于独立训练的同容量单几何模型。
alpha 对照只改变距离，因此其 AUPRC/F1 与 `main` 相同；看 retrieval、purity、coverage 判断距离融合影响。端点使用分支行的检索结果，不重复评价；分支行的 AUPRC/F1 来自各自 probe，应结合 `probe_view` 解读。
诊断同时保留逐类型预测数量、Precision、Recall、F1、AP、测试细胞数与稀有度；不删除极小样本类型，也不自动选择最佳 alpha。诊断会增加检索与 probe 评价耗时，单独记录 `diagnostic_fit_seconds` 和 `diagnostic_evaluation_seconds`。

输入文件 SHA256、源码 SHA256、Git commit、依赖版本、可见文件列表及 GPU 都有记录。测试文件的字节哈希仅用于溯源，不参与任何拟合或超参。
同配置/源码/输入且已完成的任务再次运行会跳过；残留失败/不同配置输出会报错，选择新 `OUTPUT_ROOT` 后重跑，不覆盖原结果。
失败会立即中止该 notebook 并保存 `failed.json`，其他独立 kernel 不受影响。部分完成任务不会被当成全实验结果。

三组跑完后，执行任一 notebook 最后的比较格（默认关闭，改 `COMPARE_WHEN_READY=True`）：

```bash
python -m mmdeepspatial.compare outputs/representations_v2
```

比较前检查三组配置、源码、输入、split、预处理完全一致及所有任务已完成。
`comparison/` 保存每个 seed/fold 的 paired delta、完整逐类表、各折 seed 平均和折间 mean/std。
另外保存 `diagnostic_all_runs.csv`、`diagnostic_all_cell_types.csv`、`diagnostic_seed_averaged_by_fold.csv`、`diagnostic_fold_mean_std.csv` 和 `diagnostic_paired_deltas.csv`。诊断配对表分别相对 `E0/main`、`E0/l2_normalized` 计算同 split/seed 的改变量，保留所有固定权重，不做测试集模型选择。
同一组织的多个留出折并不独立；这些 mean/std 是描述统计，不能充当独立动物重复的显著性证据。

冻结编码器可供后续 UOT 成本实验读取：

```python
from mmdeepspatial.experiment import load_encoder, encode_adata
model, prep, geometry = load_encoder("outputs/representations_v2/E2_ae/holdout_2/seed_0")
z = encode_adata(visible_adata, model, prep)
# geometry.distances(torch.as_tensor(z0), torch.as_tensor(z1))
# 大数据仍应分块；当前没有自动接入原 DeepSpatial UOT。
```

`shuffled_probe=True` 可额外启用训练标签置换 probe 负对照，编码器及正式评价标签保持不变。不要用该选项改变三组主协议的一致性。

## 工程验证

```bash
cd /path/to/DeepSpatial/MMDeepSpatial
python -m unittest discover -s tests -v
```

检查包含合成 H5AD 上三种 AE 和 PCA 的短训练→保存→冻结加载→评价、容量一致、双曲距离/梯度/边界、分块 top-k、隐藏切片读取时序、训练基因/统计量隔离、缺失类型与空稀有集合、完成后跳过与拒绝覆盖、三组配对汇总，以及 notebook 结构/语法。
新增检查覆盖大范数输入的径向梯度、归一化与分支视图、诊断拟合的防泄露时序、checkpoint 往返和诊断配对汇总。这些是合成数据工程测试，不是 MERFISH 生物学实验结果；正式结果由服务器 notebook 产生。

几何组件接口参考：[geoopt 官方文档](https://geoopt.readthedocs.io/en/latest/manifolds.html)。
