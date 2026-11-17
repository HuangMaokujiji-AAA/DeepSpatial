# RA DeepSpatial：Zhuang-ABCA-2 稀有细胞三维重建

根据 `docs/稀有细胞三维重建_DeepSpatial改造方案.md` 的 A/B/C/D 定义实现的独立研究版本。保留联合位置/表达/连续类别状态的 GiT + UOT + velocity flow matching；原 `deepspatial`、HM、RC、ST、MM 目录不修改。只交付代码和工程验证，不包含服务器正式训练结果，不承诺模块一定改善恢复。

## 数据与固定划分

输入为同级 `data/Zhuang-ABCA-2/sec_*.h5ad` 的全部66张切片、1,227,408个细胞、1,122个基因。类型用 `obs['subclass']`；稀有定义为**58张训练切片中的细胞占比 ≤1%**，另报 ≤0.1% 子组；仅 seed=42。CCF CSV 不参与本版本训练或评价，不因缺失 CCF 删除细胞。

| 集合 | 切片 | 细胞数 | 用途 |
|---|---|---:|---|
| 训练 | 除下列8张外的58张 | 1,079,913 | 归一化、类型词表/频率、运输、生态位、FM、内部遮蔽 Marker 监督 |
| 验证 | 015、034、059 | 67,253 | 仅生成后评价；可用于后续方案/超参数开发 |
| 测试 | 008、022、040、053、066 | 80,242 | checkpoint 与协议冻结后，最终一次评价 |

完整逐片深度及上下文见 `docs/split_manifest.csv` / `.json`。切片编号缺24、29、38、43、64、68、69；不按文件序号伪造等间距。验证及测试的上下文只来自训练集合；不同测试任务不互相提供实测信息。本设计是一个58张训练的模型评价5个隐藏层，**不是五个各自使用其余65张的LOO模型**。没有把验证集重新并入训练，也没有按细胞随机拆分。

选择这8张只依据沿深度覆盖和可用训练端点，没有根据其细胞类型、表达或模型分数筛选。切片总数、行数、已知切面深度属于数据划分审计元数据；它们不用于推断隐藏细胞数。隐藏XY、表达、标签在评价前不打开。公开 atlas 的坐标已有预处理配准，本实验不重做配准，不声称模拟了完全未知坐标的现实缺片。

原 `obsm['spatial']` 和 `obs['z']` 保留方向，显式按毫米×1000转为µm，再用训练范围归一化供模型使用；输出为物理µm。该尺度与本地0–13.637的全脑前后轴、约0.2的常见切片间距相符。官方数据坐标来源说明见 [Allen 数据描述](https://github.com/AllenInstitute/abc_atlas_access/blob/main/descriptions/Zhuang-ABCA-2.md)；本地拆分H5AD没有保存单位属性，`coordinate_to_um=1000` 是明确声明的尺度假设，若本地制备流程改过单位必须修改配置并重建新缓存。

## 服务器运行

将整个 `RA_DeepSpatial` 目录上传至项目根目录旁的数据目录，保持如下结构：

```text
DeepSpatial/
  data/Zhuang-ABCA-2/sec_*.h5ad
  RA_DeepSpatial/
    configs/abca2.json
    examples/run_abca2.ipynb
    resources/markers.csv
```

使用已有 CUDA PyTorch 环境，按需安装缺失依赖；不要为了本项目盲目替换服务器 torch/CUDA。依赖列在 `requirements.txt`。

```bash
cd RA_DeepSpatial
python -m pip install -r requirements.txt
python tests/test_pipeline.py
python scripts/inspect_data.py
```

**外部 Marker 已在本地生成**：上传 `resources/markers.csv` 到服务器同名位置即可，不必在服务器重新读取XLSX或安装Excel转换依赖。来源与重新生成方法见 [resources/README.md](resources/README.md)。已核实实际 `subclass_annotation` 工作表；候选表含338个subclass、1,892条类型–基因记录，面板与训练表达筛选后覆盖327个类型、其中310/319个稀有类型，详见 `resources/marker_coverage.csv`。未覆盖类型仍保留在其余训练和评价中。缺失CSV时正式8配置训练会在准备大型缓存之前报错，不会用测试切片筛基因，也不会静默把C关闭。

GPU0、24GB 的默认入口是 `examples/run_abca2.ipynb`，每个阶段使用独立 Python 子进程，避免 Notebook 已初始化 CUDA 后设备变量不生效。也可顺序运行：

```bash
CUDA_VISIBLE_DEVICES=0 python examples/run_abca2.py --stage prepare
CUDA_VISIBLE_DEVICES=0 python examples/run_abca2.py --stage train --resume
CUDA_VISIBLE_DEVICES=0 python examples/run_abca2.py --stage validate --resume
CUDA_VISIBLE_DEVICES=0 python examples/run_abca2.py --stage freeze
CUDA_VISIBLE_DEVICES=0 python examples/run_abca2.py --stage test --resume
python examples/run_abca2.py --stage summary --device cpu
```

`--stage all --resume` 可依次完成 train→validate→freeze→test；推荐首次分阶段运行，查看验证表和覆盖率。已冻结时 `all --resume` 只检查冻结身份并补测试。所有开关和超参数由 `configs/abca2.json` 固定；本程序不会根据测试结果自动选参。训练使用预声明最后一个epoch的EMA checkpoint；验证集用于检视/开发，不是暗中从测试挑最优epoch。要基于验证结果改配置，使用新的 `--cache` 和 `--output`，新一轮验证完成后再冻结；测试已经看过就不能继续当全新独立测试。

中断后使用相同配置/源码/输入和 `--resume`。完整epoch恢复 AdamW、EMA、FM时间、Marker任务、锚点及随机数流；未保存的epoch重算。已完成模型和生成文件校验哈希后跳过。不要同时启动多个进程写同一 cache/output。

## 八种实验配置

| 配置 | 定义 |
|---|---|
| `baseline` | 原GiT与全局空间/余弦/类型成本UOT、单向FM、随机自然来源 |
| `control` | 同GiT，局部同类UOT、双向FM、自然丰度低方差来源；共同主干 |
| `R0` | 冻结control主干，同容量零初始化残差；A/B/C/D均关闭 |
| `full` | 冻结同一control主干，A+B+C+D |
| `no_A` | 全局原成本UOT、自然训练风险，保留B/C/D；移除主FM的局部同类限制、边际校正和风险加权 |
| `no_B` | 完整方法，双端生态位通道输入归零，保留相同参数量 |
| `no_C` | 完整方法，去掉Marker监督 |
| `no_D` | 完整方法，去掉门控、幅度上限与锚点损失 |

所有残差配置从同一个control EMA开始，冻结主干，追加相同100 epochs；R0是新增模块的同容量/同追加训练量参照。baseline和control各100 epochs。故残差方法总计含100主干+100追加epochs，不能称为与baseline等总开销。

control同时改变局部支持、训练方向和来源调度，其相对baseline的改善不能被归到某一项。`no_A` 在追加训练、C内部候选和推理中关闭A的运输/加权干预，但仍继承相同control预训练主干，不能解释为“从零训练且整个历史没有局部先验”。full对R0则隔离共同局部control上的额外模块。C的公式和内部遮蔽目标定义一致，其候选随A是否启用而切换局部/全局UOT。

除了原baseline随机来源外，其余配置共享自然丰度的方向/类型/parent数量与生成深度；no_A因B需要对端上下文，仅对端候选改用全局UOT。D不额外增加稀有类数量。无有效局部候选时保留来源位置/表达/类型，只设置目标深度，并标记 `source_anchor`；这些点始终保留在评价分母，不能称作可靠恢复。

## 模块如何实现

- **A**：完整切片均匀边际上的局部同类稀疏KL-UOT；按物理层间距缩放候选半径。来源边际混合校正独立于稀有类别风险混合；风险仅实施一次，cap=10的截断量和独立parent数写诊断。低训练质量不能靠权重制造新候选。
- **B**：训练切片内排除自身的类型组成、64个训练log表达变异较大的实测基因的邻域均值/标准差、几何矩、支持量。自身与双端邻域分开编码，最后残差层零初始化。对端候选只来自训练实测切片，同一生成child固定条件。64基因是测量特征，不称作GRN或功能程序。
- **C**：在58张训练中，逐次选择一个内部训练目标，从上下文中移除该目标，用两侧其他训练切片构造轨迹。隐藏训练目标仅提供监督，不提供其邻域给网络。采用4步可微midpoint ODE到目标深度，对外部候选中训练富集Marker的log表达均值和固定网格Marker质量空间图相关施加损失。只监督有候选的稀有类，记录实际目标/类型/可用Marker数。归一化由外层58训练拟合，允许包含这些监督目标，但不含外层验证/测试。no_A的内部对端也使用全局UOT，不保留局部禁边；其余C配置使用局部候选。Marker空间网格300µm/带宽200µm是待验证初值。
- **D**：来源类型门控与逐模态标准化RMS上限，独立自然风险锚点约束残差。初始等价不意味着训练后非劣。这里实现的是速度场附近的线性路径锚点，不宣称完整ODE输出指标有理论保证。

原GiT/commons以及低方差整数抽样函数复用记录在 `reuse_manifest.json`，保留MIT许可。全局baseline使用原成本与KL-UOT目标，分块磁盘kernel避免RAM中的N×M大矩阵；其空间/表达最大值归一化、reg=0.8、tau=0.05与源码默认保持一致。局部控制采用已界定的余弦距离直接作成本，不做全局两两余弦最大值扫描，这一差异写在方案中。不是原Lightning每一步随机数消费的逐位复现。生成按目标深度直接积分，原按来源非零数top-k稀疏化在所有配置相同，并保留截断前表达；原连续类别状态不是校准概率。

## 评价与输出

逐目标输出 H5AD、调度、原X/log1p整体MSE/MAE/PCC、逐类型数量与缺失/虚假出现、50/100µm双向coverage和一对一P/R/F1、精确截断Hungarian距离、类型内表达均值/余弦、Marker空间PCC/Spearman与富集、类型密度图、局部组成KL/JS和偏好邻居距离。邻居偏好只从训练估计。

截断Hungarian使用稀疏dummy节点，把未匹配点按100µm惩罚；它是带数量误差的分布距离，不是细胞身份追踪。另附普通匈牙利匹配的均值/中位数，只有N×M≤400万时计算；更大类该补充字段为空，主截断指标仍精确计算。数量为零的真值类型单列虚假生成，不除零。Marker图在共同物理分箱上比较；另报密度，避免把增细胞数当表达改善。局部组成使用同为二维切片、同半径、排除自身的邻域，仅在有效匹配且双方有邻居时计算，并报告可评价配对数；不冒充已验证真实三维邻域。

`loss` Marker结果是C直接监督过的诊断，`heldout_C`是未进入C的预留外部候选基因结果；这些基因仍参与主FM表达监督，因此不等于未见基因泛化或独立实验验证。无覆盖/常量图相关系数写空并报告覆盖数。类型表达比较是pseudobulk，不是未经证实的一一对应细胞误差。各项需要联合阅读，不能只挑召回上涨。

```text
cache/abca2/
  identity.json / state.json / split.csv
  sec_*/{g,x,y,h,physical,ids,labels}.npy
  transport/*/{local,raw}/
  banks/*_sampling.csv
outputs/abca2_seed42/
  identity.json / environment.json / marker_coverage.csv
  {baseline,control,R0,full,no_A,no_B,no_C,no_D}/
    last.pt / trained.json / training.csv / marker_training.csv
    {validation,test}/sec_*/
      prediction.h5ad / schedule.npz / generated.json
      metrics.json / per_type.csv / per_marker.csv
  frozen_test_protocol.json
  completion.csv / all_metrics.csv
  {validation,test}_summary.csv / *_paired_deltas.csv
```

`freeze`要求8配置均完成训练与3张验证，锁定checkpoint及验证证据；测试调用先核对冻结身份。某组缺失时只写完成状态，不生成“完整8配置”的平均表。汇总std是同一动物不同切片间的描述，不是独立动物置信区间，也不报告生物学显著性或“非劣性成立”。不自动设整体指标容忍阈值。

## 资源和完整体积

默认batch128、patch16、单卡GPU0，面向24GB显存；未在该服务器测量峰值，若确有OOM，降低batch/chunk并使用新配置/目录。全脑表达与邻域按切片缓存/内存映射，索引轨迹避免复制表达。建议服务器CPU内存至少32GB并留出数十GB磁盘空间，具体由数据、检查点、输出密度决定。

全局原方法仍有O(N0×N1×G)成本计算及磁盘I/O，不因分块变成线性算法。最大实测相邻训练切片约3–4万细胞，其单gap临时float64 kernel可达约10GB；临时文件逐gap删除，只保存采样边与Sinkhorn因子。no_A的C内部跳片任务还会首次计算额外的全局UOT并缓存因子，其开销也属于该消融。不要把24GB显存当作全部系统内存要求。基线不做隐蔽降采样，所有训练细胞均可进入配对分布。

完成最终评价后可生成任意深度虚拟切片，默认只导出完整方法，按平面写盘，不一次拼接百万细胞：

```bash
CUDA_VISIBLE_DEVICES=0 python examples/run_abca2.py --stage export --arms full --resume
```

输出 `full/volume/z_*um.h5ad` 和 `volume_manifest.csv`。模型仍是58张训练版本，验证/测试实测细胞没有偷偷并入。当前只做单个新数据集，不实现外部SpatialZ等复现或跨动物泛化。

## 工程验证

执行 `python tests/test_pipeline.py`。覆盖：KL-UOT与POT一致性、禁边/无支持、截断一对一数量惩罚、外层隔离、Marker预留、零残差初始等价和冻结主干、C的可微梯度、8配置端到端、冻结后篡改拒绝、epoch恢复随机流。使用小型合成数据，不支持任何MERFISH效果结论。实际数据检查清单和本机验证范围见 `engineering_verification.json`。
