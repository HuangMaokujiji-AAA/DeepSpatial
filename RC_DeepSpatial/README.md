# RC DeepSpatial：MERFISH 稀有细胞恢复

本目录与 `HM_DeepSpital`、`MMDeepSpatial` 同级，代码和输出独立。复用此前五张 `merfish_0`–`merfish_4.h5ad`，默认留出 1、2、3；物理 Z 沿用原配置的 0/50/100/150/200 µm。输入只读，默认不复制原数据，不修改原 DeepSpatial 或 HM/MM 实验。

实现了上一轮方案的三个部分：A 有向运输边际校正与独立的类别风险混合；B 自身/双端生态位条件残差；C 分模态 RMS 预算、来源类型门控和自然分布锚点。没有预设效果已提升。

## 运行

推荐打开 `examples/run_merfish.ipynb`，选择之前的 MERFISH CUDA Python 环境。第一格先设置物理 GPU，随后 `cuda:0` 表示该进程看见的第一张卡；更改 GPU 后重启内核。

终端从 DeepSpatial 仓库根目录运行：

```bash
# 完整默认实验：3 个留出折、seed 42、7 个实验臂。
CUDA_VISIBLE_DEVICES=0 python RC_DeepSpatial/examples/run_merfish.py

# 机制对照：同一输出根目录、相同源码/配置下添加对照。
CUDA_VISIBLE_DEVICES=0 python RC_DeepSpatial/examples/run_merfish.py \
  --resume --arms ABC_zero ABC_shuffle ABC_single ABC_no_geometry \
  ABC_no_anchor AB_risk_only AB_marginal_only

# 完整检查点续跑。100 epochs 表示总轮数，不再追加 100 轮。
CUDA_VISIBLE_DEVICES=0 python RC_DeepSpatial/examples/run_merfish.py --resume

# 汇总已完成实验；不重新训练、不按测试指标自动选参。
python RC_DeepSpatial/examples/run_merfish.py --stage compare
```

默认输出为 `RC_DeepSpatial/outputs/merfish_v1/`。已有目录不覆盖；同配置/源码/可见输入可用 `--resume` 跳过完成组或续跑未完成组。配置、源码或可见输入变化后使用新的 `--output`。基线/模型/预测文件已完成后会核对其哈希。

单卡先查看配对和数据诊断：

```bash
python RC_DeepSpatial/examples/run_merfish.py --stage audit \
  --holdouts 2 --arms control ABC --output RC_DeepSpatial/outputs/pairing_audit
```

`audit` 读取可见切片并计算真实规模局部 UOT、邻域、训练注释器和诊断，不训练流模型、不读取隐藏真值。全局原方法对照仍是二次规模矩阵，保存多个全局耦合需要相应内存与磁盘。

阶段入口 `--stage`：`audit`、`base`、`train`、`generate`、`evaluate`、`all`、`compare`。分阶段接续使用 `--resume`。`base` 训练 baseline/control；`train` 训练指定臂并自动准备 control；`generate` 只生成；`evaluate` 才打开隐藏切片。

扩展随机种子和分 GPU 运行：

```bash
CUDA_VISIBLE_DEVICES=0 python RC_DeepSpatial/examples/run_merfish.py --holdouts 1 --seeds 42 0 1 2 3
CUDA_VISIBLE_DEVICES=1 python RC_DeepSpatial/examples/run_merfish.py --holdouts 2 --seeds 42 0 1 2 3
CUDA_VISIBLE_DEVICES=2 python RC_DeepSpatial/examples/run_merfish.py --holdouts 3 --seeds 42 0 1 2 3
```

每个进程负责不同 fold/seed；同一 fold/seed 的所有实验臂须由同一进程顺序管理。没有加入 DDP，也不允许两个进程同时操作同一 fold/seed。默认种子 42 与此前 HM 配置一致，多个种子不能代替生物学重复。

依赖在 `requirements.txt`。已有环境包齐全时无需安装；本项目使用 PyTorch 训练循环，不依赖 Lightning、Scanpy 或 PyVista。`scripts/run_python.sh` 可在服务器需要时优先加载当前环境已有的 nvJitLink：

```bash
bash RC_DeepSpatial/scripts/run_python.sh RC_DeepSpatial/examples/run_merfish.py --stage audit --holdouts 2 --arms control
```

## 实验臂与公平性

| 名称 | 训练与生成 |
|---|---|
| `baseline` | 原 GiT、全局原 UOT 成本/求解器、单向线性 velocity FM、原随机来源分布 |
| `control` | 同 GiT，局部同类物理支持 UOT、双向训练、自然丰度低方差来源；所有残差臂的共同预训练主干 |
| `R0` | control 冻结主干＋同容量零初始化残差；A/B/C 关闭 |
| `A` | R0＋运输边际校正及类别风险混合 |
| `B` | R0＋真实双端生态位条件 |
| `AB` | A＋B，C 关闭；也是完整模型去掉 C 的对照 |
| `ABC` | A＋B＋C |
| `ABC_zero` / `ABC_shuffle` | 同容量零条件 / 同类型同可见切片内置换条件 |
| `ABC_single` | h1 置零，只使用来源条件，保留全部参数 |
| `ABC_no_geometry` | 将邻域相对坐标的一、二阶矩置零，其余保持 |
| `ABC_no_anchor` | 保留门控及幅度预算，仅将锚点损失系数置零 |
| `AB_risk_only` | kappa=0，只保留类别风险混合 |
| `AB_marginal_only` | 类别风险混合=0，只校正运输边际 |

所有残差臂从同一 fold/seed 的 control EMA 主干开始，主干冻结；初始化、训练步数、残差参数量一致。A 只改变残差训练阶段的配对分布和损失风险，不改变生成类型预算。C 使用独立随机流采样锚点，避免额外随机数消费改变主任务的时间采样。

默认预训练 100 epochs，各残差臂再训练 100 epochs；两段成本分别记录。不能将两段模型与仅 100 epochs 的基线视为等训练开销；`R0` 是必要同容量、同追加训练量对照。

原 GiT 文件、原 UOT 文件和复用的 HM 工具逐字保存在 `rcspatial/vendor/`；来源 SHA-256 与上游提交在 `upstream_manifest.json`。原 MIT 许可和作者归属保留。这是原结构/目标的独立训练入口，不是原 Lightning 随机数消费和检查点格式的逐步复现。全局 UOT 输入提升到 float64，避免小正则下缺失类型导致 exp 下溢；成本公式与参数未更换。配对预算用最大余数法分完配置总数，避免原逐 gap 向下取整丢掉余数。

## 模块细节

**A — `rcspatial/pairing.py`**

前向与反向先决定方向，再分别计算：`q=(1-kappa)*pi/sum(pi)+kappa*uniform_supported_source*K`。只校正该方向的来源边际，不宣称双边同时守恒。无正质量边的端点不制造配对；生成时明确标记 `source_anchor`，保留在评价分母中。

配对 proposal 是 q，类别风险通过一次重要性权重实现：`(1-eta)+eta/(supported_class_count * q_class_mass)`；默认 eta=0.2、cap=10。裁剪数量写入诊断，裁剪后不宣称精确无偏。kappa=0.5 是预声明初值，不是验证所得最优值。

**B — `rcspatial/model.py`、`data.py`**

自身状态与邻域分开编码。邻域只使用可见实测切片、排除自身，包含邻居类型组成、每个实测基因的邻居标准化 log 表达、邻域观测量，以及相对 XY 的一/二阶矩。155 个实测基因分别作为测量特征，没有将其命名为生物功能通路。

双端条件来自采样的来源/对端候选，训练和推理定义一致。候选在每个生成 child 的共享调度中固定，同一来源可以对应多个对端条件。共享相同条件的确定性 ODE 仍只有一条路径，不把重复 parent 数量当作已恢复的新生物学多样性。置换控制使用独立 RNG，不改变配对和来源调度。

**C — `TrustController`、`training.py`**

修正经过按来源类别的固定门控和逐模态标准化 RMS 上限。空间使用训练归一化 XY，表达按训练基因标准差缩放，类别为连续 one-hot 状态单位。默认 common_gate=0.25，x/g/c RMS 上限分别 0.05/0.1/0.05；这些均是待验证初值。

锚点由自然配对分布独立采样，覆盖线性路径状态及模型向量场的一步邻近状态；不是完整 teacher ODE 轨迹。锚点惩罚是修正场偏移，没有使用隐藏测试误差做门控，也未将它称为经过校准的置信度。没有自动执行超参选择或宣称非劣。

## 数据隔离与评价

隐藏文件在读取前被排除。基因顺序、类别词表、表达尺度、邻域基因统计、稀有名单、probe 和 marker 候选均只由可见切片决定。X 沿用原输入尺度，不暗中 log/归一化训练表达；邻域/probe/评价所用 log1p 单独声明。

所有残差臂与 control 精确共享来源、类型、方向、目标深度和对端候选的 schedule 文件，不根据隐藏细胞数补预算。先完成该 fold/seed 的全部请求训练及生成，再首次读取隐藏表达/标签做评价。已有留出数据参与过方案开发，默认 role 是 `development`。

内层验证可以显式排除外层测试切片，例如：

```bash
# 内层验证 1，外层测试 2 完全排除；从剩余可见切片寻找两侧端点。
python RC_DeepSpatial/examples/run_merfish.py --holdouts 1 --excluded 2 --role validation \
  --arms control R0 ABC --output RC_DeepSpatial/outputs/inner_for_outer_2
```

它实现隔离的内层运行，不自动选择最优预算。常见/整体指标可接受退化量应在最终测试前确定；本代码不使用测试指标选择模型，不从同组织折间均值推断生物学显著性。

评价保留：原始 X 与共享 log1p 的基因均值 MSE/MAE/PCC；每类数量/比例/缺失类/虚假出现；25/50/100 µm 同类双向 coverage 与一对一最大匹配 F1；每类表达均值；固定空间分箱密度 TV；空间最近邻误差。缺失目标类的召回/F1 为 0，未见类与测试不存在类型单列。

逐类型同时报告 `model_class` 和训练切片拟合、冻结的表达 LogisticRegression probe 注释结果。probe 是表达一致性证据，训练面板与模型相同，不能代替独立生物学验证。marker 诊断也只使用训练选出的候选，报告 top-k 前后检出变化，不宣称独立 marker 真值。

## 输出与机制分析

```text
outputs/merfish_v1/holdout_i/seed_s/
  manifest.json
  shared/
    state.json                 train_counts.csv
    local_gap_*.npz            raw_gap_*.npz（请求 baseline 时）
    transport_local.csv        transport_raw.csv
    matching_support.csv       sampling_budget.csv
    schedule_control.npz       schedule_baseline.npz
    probe.joblib               markers.json
    environment.json           evaluation_truth.json（仅评价后）
  baseline/、control/、R0/、A/、B/、AB/、ABC/、各机制对照/
    manifest.json              last.pt
    training.csv               training_by_type.csv
    trajectory_indices.npz     trajectory_sampling.csv
    trained.json               predicted_gap.h5ad
    predicted_holdout.h5ad     generated_type_counts.csv
    generated.json             per_type.csv
    marker_sparsification.csv  metrics.json
    evaluated.json             failed.json（失败时）
```

`last.pt` 包含训练权重、EMA、AdamW、epoch、逐类日志和主任务/锚点/loader 随机状态；完成续跑以 epoch 为边界。`predicted_*` 保存独立来源字段、对端候选、支持状态、连续类别状态、`expression_before_sparsification` 与物理 XYZ。

汇总输出 `completion.csv`、`all_runs.csv`、`paired_deltas.csv`、`seed_averaged_folds.csv`、`descriptive_summary.csv`、`drift_to_control.csv`。存在未完成实验臂时不汇总为完整实验；独立 GPU 进程完成自己负责的折后可以正常结束，等待其他折完成再汇总。显式 `--stage compare` 会报告未完成组。按实际完成集合做比较，不自动混入旧目录其他配置。

机制证据对应：

| 要回答的问题 | 对应文件/对照 |
|---|---|
| 稀有类是否在 UOT 后额外受压？ | transport CSV 中 input/uot/corrected 比例；matching_support 中端点质量、熵和距离 |
| 是否只重复少量来源？ | trajectory_sampling 的 unique source/target、ESS、生成 parent 数 |
| 边际校正还是类别风险在起作用？ | AB、AB_risk_only、AB_marginal_only |
| 环境是否提供额外信息？ | ABC、zero、shuffle、single、no_geometry，固定同一 schedule |
| C 是否减少扰动？ | AB、ABC、ABC_no_anchor；training 的分模态 drift 和 drift_to_control |
| marker 是否在后处理丢失？ | marker_sparsification 与保存的截断前表达 |

precision、recall、丰度和表达结果需一起阅读，不能只用覆盖率上涨代表恢复更好。不同模块作用依靠这些干预与中间量判断，不能只列删模块后的总分。

## 工程检查

```bash
python RC_DeepSpatial/tests/test_pipeline.py
```

检查有向边际恒等式和禁边、隐藏/外层文件读取隔离、同类置换、零初始化等价、冻结主干与模态预算、线性 FM 目标一致性，以及合成数据上训练→保存→生成→独立评价→恢复。另模拟 epoch 检查点之后中断，验证恢复 AdamW/EMA/随机流后的结果与连续 CPU 训练相同。这些检查不是 MERFISH 效果实验。

2026-11-01：5 项工程检查通过。本机 MERFISH 真实数据入口已核对：留出 2 时读取 0/1/3/4，共 155 基因、15 类型、177 维邻域特征，特征均有限。另对这些可见数据的少量来源执行了 64 条轨迹、基础模型与 ABC 各单轮训练、90 个生成细胞的保存检查；没有读取隐藏切片，临时产物已清理。这是实际面板维度的接线验证，尚未执行默认 100 epochs 正式比较，不提供改进结论。
