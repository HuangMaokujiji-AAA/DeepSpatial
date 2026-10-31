# DeepSpatial 稀有细胞三维恢复：文献依据、修改方案与机制实验

检索日期：2026-11-01。本文为研究设计，未运行新模型、未产生效果结论。保留上游代码和已有 HM/MM 实验。文献事实、代码事实与拟议机制分别表述。

## 1. 任务边界与研究命题

任务是从已观测空间转录组切片重建切片间组织，改善训练中有支持的稀有细胞类型及其局部表达状态，同时控制原有整体和常见类型指标的退化。稀有类型的数量、空间分布和表达状态需要分别评价。

主要分析以本地 MERFISH 开发设置为起点：沿用训练丰度 ≤5% 的既有定义，预先增加 ≤1% 的极稀有子组；不按测试切片丰度重新选类型。训练可见切片和外部参考中都未出现的类型属于不可识别或开放集问题，不能承诺仅靠插值恢复。两端未出现、仅隐藏层出现的细胞岛要单列。

拟议命题：跨切片运输后的边际变化、局部环境信息不足以及修正时对已有向量场的扰动，可能分别限制稀有类恢复。研究应检验三个机制是否实际存在，以及对应干预是否改善它们。

## 2. 检索与证据范围

检索使用公开关键词与论文标题，覆盖 spatial transcriptomics / multi-slice alignment / missing-slice reconstruction / continuous 3D reconstruction / rare-cell recovery / long-tailed flow matching / population-conditioned flow matching。优先查官方期刊、NeurIPS/ICLR/PMLR、作者项目和原始论文记录，按题名和 DOI 去重。

下表是决策相关文献，不是穷尽目录或性能排名。大部分用于方法定位的是官方摘要或作者项目说明；PASTE2、GEASO、SINTER3D、STADiffuser 和 UOT-RFM 另有可访问正文。部分 Nature、bioRxiv 和 OpenReview 页面存在访问限制，不能把未读补充材料当作已经核实的证据。不拼接不同论文的性能数字。

### 2.1 空间组学三维重建与切片整合

| 工作、正式题名 | 年份/状态 | 类型与任务 | 已核实的方法事实 | 对本课题的作用及边界 |
|---|---|---|---|---|
| [PASTE — Alignment and integration of spatial transcriptomics data](https://www.nature.com/articles/s41592-022-01459-6)，[PubMed](https://pubmed.ncbi.nlm.nih.gov/35577957/) | Nature Methods，2022 | 方法；切片对齐、整合、堆叠 | 综合表达与空间结构做概率对齐 | 运输的经典基线；堆叠对齐与缺失层细胞生成应分开比较 |
| [PASTE2 — Partial alignment of multislice spatially resolved transcriptomics data](https://pmc.ncbi.nlm.nih.gov/articles/PMC10538490/) | Genome Research，2023；另见 RECOMB 2023 [官方目录](https://recomb.org/proceedings/proceedings/2025-2021/2023/) | 方法；部分对齐 | partial-FGW 允许部分重叠和切片特有类型 | 说明不应为了保留稀有类而强迫所有细胞跨层匹配；不是缺失层单细胞生成器 |
| [GPSA — Alignment of spatial genomics data using deep Gaussian processes](https://www.nature.com/articles/s41592-023-01972-2) | Nature Methods，2023 | 方法；非刚性坐标整合 | 两层 GP 将坐标映到共同空间并建模读出 | 作为坐标形变与表达场的相关工作；不直接等同于生成新细胞 |
| [STitch3D — Construction of a 3D whole organism spatial atlas by joint modelling of multiple slices with deep neural networks](https://www.nature.com/articles/s42256-023-00734-1)，[作者代码](https://github.com/YangLabHKUST/STitch3D) | Nature Machine Intelligence，2023 | 方法；联合整合与细胞类型分布 | 联合多切片和单细胞参考，推断三维空间区域与类型分布 | 支持跨层联合建模；其去卷积输出与单细胞级虚拟细胞需转换到共同评价尺度 |
| [STalign — Alignment of spatial transcriptomics data using diffeomorphic metric mapping](https://www.nature.com/articles/s41467-023-43915-7) | Nature Communications，2023 | 方法；微分同胚配准 | 支持空间组学数据在二维及三维共同坐标中的对齐 | 用于排查上游配准误差；不作为完整缺失层生成器 |
| [Spateo — Spatiotemporal modeling of molecular holograms](https://www.devo-evo.com/papers/spateo/) | Cell，2024 | 方法与综合工具；三维组织及发育分析 | 包括部分非刚性对齐、多切片细化和三维结构分析 | 借鉴结构与局部环境评价；跨发育时间的运动不等于沿组织深度的插值 |
| [SpatialZ — Bridging the dimensional gap from planar spatial transcriptomics to 3D cell atlases](https://www.nature.com/articles/s41592-025-02969-9)，[PubMed 摘要与出版日期](https://pubmed.ncbi.nlm.nih.gov/41476112/) | Nature Methods；在线 2025-12-31，卷期 2026 | 方法；虚拟切片和稠密三维细胞图谱 | 在实测切片之间生成虚拟切片，面向单细胞分辨率 | 最接近的正式发表比较对象之一；不能只比较配准方法 |
| [JADE — Joint Alignment and Deep Embedding for Multi-Slice Spatial Transcriptomics](https://proceedings.neurips.cc/paper_files/paper/2025/hash/24b7e2d0ef3781a363911b9f2b1ba833-Abstract-Conference.html) | NeurIPS 2025 主会 | 方法；多切片对齐与表示 | 交替细化对齐和共享嵌入，以注意力选择对齐相关维度 | 说明“换表征再做 OT”已有直接前作；未等同于缺失层稀有细胞恢复 |
| [GEASO — Network model for alignment, stitching and slice-to-volume 3D reconstruction of large-scale spatially resolved slices](https://www.nature.com/articles/s41467-026-71042-6.pdf) | Nature Communications，2026 | 方法；大规模切片配准与拼接 | 图特征、弹性配准、低秩及降采样加速 | 提醒需要控制形变和局部拓扑因素；其主要重建任务仍不同于生成未测细胞 |
| [SINTER3D — continuous 3D reconstruction of spatial transcriptomics via implicit neural representations](https://link.springer.com/article/10.1186/s13059-026-04160-5) | Genome Biology，2026-06-17 | 方法；连续多基因表达场 | 以三维坐标为输入联合插值多个基因 | 适合比较缺失位置表达场；不能把连续场结果直接解释成独立细胞身份恢复 |
| [STADiffuser — high-fidelity simulation and full-view 3D modeling of spatial transcriptomics](https://pmc.ncbi.nlm.nih.gov/articles/PMC13582971/) | Nature Communications，2026-08-19 | 方法；空间数据模拟和生成 | 图注意力 AE 后接潜在扩散与空间去噪，支持多样本及三维建模 | 最新正式发表的生成方向近邻；需核对位置是否给定、分辨率与条件输入后确定公平比较方式 |
| [DeepSpatial — Reconstructing True 3D Spatial Omics at Single-Cell Resolution](https://yyh030806.github.io/DeepSpatial/) | bioRxiv，2026；本次依据作者项目及本地代码，未核实到正式期刊发表 | 方法；OT 流匹配连续重建 | 联合空间、表达和类型的向量场，双向来源生成 | 本课题基础；论文宣传中的连续分辨率不等于任意深度均有实验保证 |
| [isoST — Three-dimensional spatial transcriptomics at isotropic resolution enabled by generative deep learning](https://www.biorxiv.org/content/10.1101/2025.08.15.670472v1)，[作者实现](https://github.com/deng-ai-lab/isoST) | 已核实 2025 预印本与作者代码；未确认正式发表状态 | 方法；沿深度的连续重建 | 作者代码说明使用 GNN 估计空间/表达梯度，以 SDE 建模组织连续性 | 适合补充连续重建比较；“各向同性”描述输出采样能力，仍需检查真实恢复误差 |

GPSA、STalign 的方法定位依据已检索到的期刊摘要/元数据；未据此推断它们对稀有类的具体效果。核心比较优先使用 SpatialZ、DeepSpatial、STADiffuser 及任务兼容的 isoST；表达场比较加入 SINTER3D。配准方法用于上游对齐诊断。

### 2.2 可迁移的深度学习机制

| 工作 | 年份/状态 | 公开方法依据 | 本方案中的迁移与限制 |
|---|---|---|---|
| [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) | ICLR 2023 | 回归条件向量场来学习生成流 | 作为基线原理；FM 不是本课题新增贡献 |
| [Wasserstein Flow Matching: Generative Modeling Over Families of Distributions](https://proceedings.mlr.press/v267/haviv25a.html) | ICML 2025 | 在分布空间建模，含空间转录组细胞微环境实验 | 支持环境应作为分布而非一个平均邻居；原文未证明本课题的稀有类提升 |
| [Meta Flow Matching: Integrating Vector Fields on the Wasserstein Manifold](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ebdb990471f653dffb425eff03c7c980-Abstract-Conference.html) | ICLR 2025 | 用 GNN 编码初始细胞群体，条件化流模型 | 支持 population-conditioned dynamics；把群体条件迁移到空间邻域是拟议操作 |
| [UOT-RFM — Reweighted Flow Matching via Unbalanced OT for Label-free Long-tailed Generation](https://arxiv.org/html/2509.25713v1) | arXiv，2025；本次仅确认预印本，未核实录用状态 | 根据数据分布与 UOT 边际的密度比重新加权，处理多数偏置 | 最近的机制重叠风险；“UOT+重加权”不能直接宣称新颖。其噪声到数据设置与跨组织切片、部分支持和几何恢复不同 |
| [WFR-FM: Simulation-Free Dynamic Unbalanced Optimal Transport](https://proceedings.iclr.cc/paper_files/paper/2026/hash/2030d35841f8617ab22cca913f55697e-Abstract-Conference.html) | ICLR 2026 | 同时学习位移场与质量变化率 | 若未来单独建模类型密度可借鉴；沿 z 的密度变化不是细胞增殖/死亡，首版不加此分支 |
| [Class-Balanced Loss Based on Effective Number of Samples](https://research.google/pubs/class-balanced-loss-based-on-effective-number-of-samples/) | CVPR 2019 | 根据有效样本数进行类别再加权 | 用作简单长尾对照；直接照搬不能解释三维结构改进 |
| [Delving into Deep Imbalanced Regression](https://proceedings.mlr.press/v139/yang21m.html) | ICML 2021 | 连续目标的不均衡学习与分布平滑 | 提醒表达/速度是回归问题；不能把 one-hot 速度头直接当分类概率套 focal loss |
| [ControlNet — Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543)，[正式 ICCV 页面](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html) | ICCV 2023 | 冻结主干，零初始化条件分支 | 借鉴保守增加条件信息；零初始化只保证初始等价，不保证训练后非劣 |
| [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html) | ICML 2019 | 对集合中元素的关系建模并进行置换不变聚合 | 邻域集合编码器的一种实现；需与简单均值/直方图比较 |
| [Gradient Surgery for Multi-Task Learning](https://papers.nips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html) | NeurIPS 2020 | 处理任务间梯度干扰 | 可作为观察到真实梯度冲突后的对照；不是首版必加模块 |

这些工作多数为方法论文；WFR-FM 同时有理论结果。本文未对未全面精读的论文打质量分，也不把作者实验结论外推成本方案结论。

## 3. 本地 DeepSpatial 的可核实结构与待检验瓶颈

以下定位针对根目录 `deepspatial/`，不是 HM/MM 副本。

1. `data_utils/uot_solver.py`：空间欧氏距离与表达余弦距离分别按最大值归一化，再加 `10 * class_mismatch`。标签已进入成本，不能说基线忽略类型。类别不同受到强惩罚，但不是数值上的严格禁止。默认 UOT 的 `reg=0.8`、`reg_m=0.05`；不同成本尺度下不可仅比较参数大小推断实际质量损失。
2. `data_utils/dataset.py:98–125`：按相邻切片细胞数乘积分配配对数，再以 `pi / pi.sum()` 抽取轨迹。UOT 行/列边际可以偏离输入边际；整体归一化不会恢复各类型原占比。稀有类是否被压低，需要直接计算，而不能由名称判断。
3. `module.py:63–98`：位置、表达和连续类别状态均做 flow-matching MSE。`c_head` 虽有 logits 注释，实际训练目标是类别状态速度。原 `ct` 条件已存在，应检查的是尾类有效监督量和梯度，而非再加一个类型 token。
4. `models/git.py:159–201`：注意力沿单个细胞的位置/基因 token 运行，batch 中没有显式的邻居细胞图或邻域集合。位置可能间接编码环境，因此需检验显式邻域是否增加条件信息。
5. `core.py:475–476`：生成来源使用随机有放回抽样。若某薄层预算为 N、某类来源比例为 p，该类没有来源的概率为 `(1-p)^N`。这是抽样性质，不等同于网络遗忘。
6. `core.py:482–515`：相同来源在一次生成中复用一条确定性 ODE 轨迹。在同一深度重复抽同一个 parent 不会自动产生新的表达状态。来源数变多、来源被重复利用、真实条件分布更丰富需要区分。
7. `core.py:528–548`：按来源细胞非零基因数保留生成表达的 top-k。低幅度 marker 有被截掉的可能；需比较截断前后，而不能把 marker 丢失都归因于流模型。

代码事实只支持上述结构；“UOT 吞掉稀有类”“邻域一定有用”“marker 实际已被损坏”目前都是待测假设。

已有 `HM_DeepSpital/README.md` 记录同类局部 UOT、混合抽样、低方差来源、生态位 real/zero/shuffle、双向训练、来源锚定和空间插值对照；`MMDeepSpatial/README.md` 记录欧氏/双曲/联合表示及加权、图编码对照。它们是开发基础，不重复当成新贡献，也不把工程通过记录当效果证据。本次未重新审核这些实验的完整结果。

## 4. 首先做四个有停止条件的定位实验

| 定位问题 | 固定因素 | 读取量 | 继续条件 |
|---|---|---|---|
| 原始类型丰度是否在运输后进一步下降？ | 固定切片、成本和 UOT 参数 | 每类原始比例、UOT 行/列质量、归一化后配对比例、unique parent、配对 ESS | 有明显且跨折重复的额外压低，才以运输校正作为主要贡献 |
| 改变来源调度是否足以解释恢复？ | 同一权重、同一总数与深度分布 | 多抽样 seed 的稀有类缺失概率、来源数、重复率、空间精确率/召回率 | 来源控制已解释主要收益时，先解决采样，不继续归因于表征或生成学习 |
| 显式环境是否提供额外信息？ | 相同来源、配对、模型容量及训练预算 | real/同类同片 shuffle/zero 的留出空间误差和表达差异 | 真实环境持续优于对照时，才升级成复杂邻域编码器 |
| 稀疏化是否造成 marker 损失？ | 同一个生成张量 | top-k 前后 marker 检出率、表达分布与整体误差 | 若损伤主要发生在后处理，先独立比较输出算子；不再声称这是 FM 表达学习缺陷 |

来源复制、同类 OT 直接插值是必要低成本对照。若模型没有超过它们，优先研究空间运动或表达变化机制，不扩大模型。

## 5. 推荐架构：三个独立模块

首版保留 DeepSpatial 主干，建立 A 运输后边际校正、B 双端生态位条件残差、C 分模态信赖控制。来源调度和输出稀疏化先作为所有组共享且单独报告的控制因素。

### A：有支持的运输边际校正

目标：区分合理的不匹配与训练采样中的额外多数偏置。保留原成本或复用已经定义的局部同类支持图；不先更换到双曲空间，也不强制每类等质量。

在支持集合上的耦合归一化为 `bar_pi`，令 `r_i=sum_j bar_pi_ij`。对于存在可信候选的来源行，定义：

\[
K_{ij}=\bar\pi_{ij}/r_i,\qquad
q^{\rm cal}_{ij}=\widetilde a_i K_{ij},\qquad
q^{(\kappa)}=(1-\kappa)\bar\pi+\kappa q^{\rm cal}.
\]

`tilde_a` 是可匹配来源细胞的自然经验分布，归一化总和为 1。对反向训练使用对应的目标列条件分布。常数 `kappa` 下，前向来源边际恰为 `(1-kappa) r_i + kappa tilde_a_i`；这是代数性质，不是改善效果的定理。前向校正不同时保证目标边际，反向校正也不同时保证来源边际，不能把二者说成精确双边守恒。

若使用支持掩码，必须拆开比较“支持图改变”与“边际校正”。无支持来源不适用除法校正：不制造远距离或跨类匹配，沿用明确标记的来源锚定策略并计入总评价。支持阈值、物理半径和 `kappa` 由内部验证固定。熵和最近邻距离只是诊断量，未经验证不能叫校准置信度。

稀有类的训练重视程度由独立损失混合控制：

\[
\mathcal L_{\rm fit}=(1-\eta)\mathbb E_{q^{(\kappa)}}\ell
 +\eta\frac1{|\mathcal C_S|}\sum_{c\in\mathcal C_S}
 \mathbb E_{q^{(\kappa)}\mid c}\ell.
\]

类别按有向轨迹来源定义。若通过 proposal 重采样实现，则记录 proposal，使用目标分布/proposal 的重要性权重，避免重复加权。极少样本的权重应有上限；裁剪后不再宣称精确无偏。

关键区分：`kappa` 校正 UOT 引入的训练边际变化；`eta` 改变模型对尾类误差的重视。两者都不直接把推理类型比例设为均匀。A 的新颖性单独较弱，必须对照 UOT-RFM 和普通 class-balanced 方法。

### B：自身信息与双端生态位分离的条件残差

目标：在同一种稀有细胞的不同空间位置/微环境之间分辨恢复路径，并避免邻域均值覆盖该细胞自身信息。

以每张可见真实切片建立物理邻域，排除自身，分别保留三类输入：

- 自身：基因表达与已知类型信息；
- 环境：邻居类型组成、邻居表达/程序、相对 XY 和邻域有效观测数；
- 对端：从 A 的条件运输分布抽取的可见对端候选及其邻域摘要。

先比较简单直方图/均值与同容量集合注意力，再决定是否需要更复杂的图编码。邻域输入使用可见实测数据，不在 ODE 中用越来越多的生成细胞循环更新。

基线向量场冻结，新分支输出 `delta_v=(delta_v_x,delta_v_g,delta_v_c)`，其末层零初始化。实际场写为：

\[
v_{\rm new}(s,t)=v_{\rm base}(s,t)
 +G(s_0,h_0,h_1,\Delta z,t)\odot\Delta v_\theta(s,t,h_0,h_1).
\]

这里 `s=(x,g,c)`，`h0/h1` 都来自可见端点。训练和推理使用同样的候选抽取与条件定义；训练不得看隐藏中间切片。一个来源可以对应多个抽取的对端条件，因此不必把所有候选先平均成一个重心。条件相同且 ODE 确定时仍只有一条轨迹，不能因此宣称已经恢复完整随机分布。

双端条件是首选候选；如果它不能超过单端条件，不保留其复杂度。配对仅表示跨深度的统计对应，不解释为同一细胞的真实谱系。

### C：分模态信赖控制与常见类型保持

目标：使“基本不影响原有指标”成为模型约束和验证标准。C 控制 B 的修正幅度，分别对空间、表达和类别状态设置预算，而非使用一个总尺度。

训练目标包括自然分布上的基线保持项：

\[
\min_\theta\mathcal L_{\rm fit}
 +\sum_{m\in\{x,g,c\}}\lambda_m
 \mathbb E_{s\sim\mathcal D_{\rm anchor}}
 \|v_{{\rm new},m}(s)-v_{{\rm base},m}(s)\|^2.
\]

锚点覆盖自然分布中的常见类型、基线与新模型轨迹状态。另在标准化单位中限制 `G_m delta_v_m` 的范数，门控依赖来源类型和可见支持信息；不使用测试误差或隐藏标签。每个模态的单位、尺度、预算都记录清楚。冻结主干和零初始化在 B/C 相关对照中保持一致，以分离“冻结”与“信赖约束”的效果。

最终模型只能在内层验证中选择满足各项退化预算的候选。若所有候选都不能满足，保留原模型或进一步缩小命题，不能靠总体加权分数掩盖某一项退化。

理论解释：在同初始状态、有限区间 `[0,T]`、基线场 L-Lipschitz 且残差统一有界 `||G delta_v||≤delta` 的条件下，连续状态偏移满足 Gronwall 型界 `delta*(exp(L*T)-1)/L`，L=0 时为 `delta*T`。它只解释小修正为什么有望减小轨迹漂移；实际 L、状态尺度、数值积分误差和 argmax 不连续性都会影响结果。它不保证 PCC、F1 或稀有类恢复指标非劣。

### 首版明确不加入的复杂度

双曲几何、NB/ZINB 输出、独立 birth/death 分支、全量三维动态图和额外扩散模型均不作为默认模块。已有表达表示实验可以决定是否需要改 OT 距离；计数似然需要明确原始 counts 与归一化协议；深度密度变化没有时间上的增殖解释。若某项定位实验提供直接证据，再扩展对应分支。

## 6. 机制消融：从替代解释到可证伪预测

模块删除表保留，但主要证据由以下实验组成。所有“预计”均为待检验预测。

| 机制命题 | 必须控制/干预 | 中间证据与结果 | 什么结果会否定或缩小解释 |
|---|---|---|---|
| A 修复运输后额外多数偏置 | 固定原成本/支持图；比较原 pi、仅 support、仅 marginal correction、普通类别均衡；统一训练步数和生成预算 | 每类原始/UOT/校正/实采样占比、unique parent、ESS、运输距离和同类局部相容性，随后看稀有类空间与表达 | 校正前边际本就没有压低，或只增加配对重复数而不改善结果，不能把 A 解释为主要瓶颈修复 |
| A 改善真实结构而非“采更多” | 固定每类最终生成数量、来源和目标深度；另外报告自然预算完整生成 | 同类空间 precision/recall、截断距离一对一匹配、marker 条件分布 | 优势在固定数量后消失且 precision 降低，说明主要是预算效应 |
| B 的环境含有额外可用信息 | 相同参数量、输入维度、初始化、配对和 parent；real/zero/同类型同切片 shuffle；再对距离层和密度做匹配置换 | 同类型细胞在不同 niche 下的位移/表达误差；半径剥离曲线；真实相对坐标与打乱坐标对比 | real 不优于 shuffle；或只在训练组织好、跨组织失效，分别指向无条件信息或位置记忆 |
| B 保留自身特征并利用环境 | 自身分支、邻域分支、二者合并、均值聚合、集合注意力同容量对照 | 稀有 marker 可分性、与多数邻居表达的接近程度、边界区误差 | 邻域加入后 marker 被削弱，说明环境污染；注意力无额外收益时使用简单聚合 |
| B 双端条件解决单源路径歧义 | 固定来源与候选预算；单端、双端、对端打乱；采样候选与重心候选分开 | 每类路径分散度、conditional coverage、niche 模式数、空间 precision | 多样性增长但 precision/表达质量下降，属于不受约束扩散；双端不优于单端则删掉双端 |
| C 减少改进时的常见类型漂移 | 相同冻结主干和残差容量；无锚点、常数门控、完整 C；扫预算不扫测试集 | 稀有类收益—常见类/原始指标退化的 Pareto 曲线、分模态速度/终点漂移 | C 只把所有增益压成零，说明没有形成有效权衡；不能仅凭 common 保持就说成功 |
| 上游匹配与环境互补 | A/B 的 2×2 因子设计，在相同 C 和预算下训练 | 交互差分 `Delta_AB=metric(AB)-metric(A)-metric(B)+metric(base)`，以及对应中间量 | 只有正交互数值而没有配对/条件信息证据，仍不足以说明机制 |

另做稀有度、空间结构和噪声的区分实验：

1. 在训练侧将若干原本足量的类型逐级下采样为长尾，完整测试真值固定；各臂共享保留下来的细胞。展示相对保留率的曲线。这是可控缺失模拟，不能代替天然稀有类结果。
2. 同时分析孤立稀有细胞、稀有细胞小岛、组织边界和重复但远隔的同类区域。它们区别“缺样本”与“配错局部结构”。
3. 用同一切片的已知变换/可控表达扰动构造有真实对应的配准诊断。真实相邻物理切片无一对一真值，那里只能称类型/邻域相容性，不能报“真实配对准确率”。
4. 稀有标签噪声对加权机制尤其关键：只在训练标签上注入预定噪声，检查是否放大错误。保留原始无噪声主实验。
5. 稀疏化算子单独比较；若改成 marker 保护，需要公开 marker 选择来源，不能由测试 marker 结果反向选择。

## 7. 数据、基线与评价

### 7.1 数据拆分

- 现有 MERFISH 留中间切片可做开发和机制定位；它已经参与多轮方案开发，不能继续称为完全独立最终测试。
- 所有归一化、基因选择、稀有类型名单、程序、图编码器、支持半径、门控及损失预算只在外层训练数据内部确定。留出切片的表达、标签、细胞数和空间密度不进入这些步骤。
- 最终验证优先使用独立动物/组织；另考虑 STARmap/RIBOmap 原生三维数据，按 z 遮蔽薄层或连续块，区分单层插值与更大间距缺失。作者 DeepSpatial 项目已列有 STARmap/RIBOmap 验证，但具体可下载样本、基因面板、标签质量和许可仍须在执行前核对。
- 再扩展到不同组织或不同技术的单细胞分辨率数据。Visium 的 spot 恢复先在 spot/去卷积尺度评价，不能直接视为稀有单细胞真值。

### 7.2 基线层次

1. 原版 DeepSpatial 完整配置作为主基线，记录 commit、输入、配对、超参和总预算。
2. 共用采样与输出算子的 DeepSpatial 对照，与原配置单列，避免把工程调度收益算给新模块。
3. 来源复制、最近切片复制、同类 OT 直接路径/表达插值，以及仅普通重采样或 class-balanced 的 DeepSpatial。
4. SpatialZ；任务与输入兼容时加入 STADiffuser 和 isoST；SINTER3D 用于坐标条件下的表达场结果。
5. UOT-RFM 风格的边际比重加权移植对照，明确“适配版”而非冒称完整原论文复现；原论文完整实现若能匹配任务则另列。
6. HM 已有 G/S/zero/shuffle 和 MM 的表征实验作为内部机制基线。是否纳入最终表取决于完整实验协议核对，不能凭目录名判断配置。

各方法使用相同可见输入和评价区域；有额外 scRNA 参考或组织图像的方法单列额外信息。模型容量、训练步数、调参预算及 CPU/GPU 开销均记录。对齐方法不强行承担其没有的生成输出。

### 7.3 评价指标

| 层面 | 指标与用途 | 陷阱控制 |
|---|---|---|
| 原有指标保持 | 沿用已定义 MSE/MAE/PCC，并补齐相同尺度上的分层表达结果 | 基因均值 PCC 不能代表逐细胞/局部空间恢复；禁止混用 raw 与 log 尺度 |
| 类型丰度 | 每类 count/fraction error、缺失类率、TV/JS、自然预算总细胞数误差 | 完全缺失的目标类纳入惩罚；测试真值为零的类型单独报虚假生成 |
| 稀有类位置 | 固定物理尺度的同类型 precision/recall/F1、截断半径的一对一最大匹配，必要时按类型的分布距离 | nearest-neighbor coverage 可多对一且可被增样本数抬高；必须同时报 precision、固定预算结果及自然预算结果 |
| 表达与身份一致 | 每类 pseudobulk、marker 检出和分布、空间分箱表达、独立冻结注释器下的类型恢复 | 生成 cell_class、来源 cell_class 不是生物学验证。独立注释器只用训练/外部参考拟合，尽可能使用独立 marker/留出基因并报告面板限制 |
| 局部组织结构 | 类型邻接、局部类型组成、pair-correlation、稀有岛边界与组件尺度 | 先按密度/面积控制；不把网络连得更平滑直接当真实三维拓扑更好 |
| 机制量 | 运输边际、unique parent、ESS、候选支持、表达截断损失、速度/终点漂移 | 诊断量本身不能替代最终恢复效果 |
| 资源 | 训练和生成时间、显存、参数量、配对开销 | 所有模块开销进入比较；多个 seed 不是独立生物学重复 |

自由生成点云与真值细胞通常无严格一一对应。表达逐点误差要明确是空间关联或模型输出位置上的场误差；真实逐细胞重建准确率仅在对应真实可知的设计中报告。

### 7.4 原指标“基本不变”的统计定义

为每个需保持的指标预先定义可接受退化 `epsilon_j`，由业务/生物意义和内层基线波动确定，不能看测试后改。统一转换为“越大越好”的差值 `Delta_j`，目标是相应置信区间下界高于 `-epsilon_j`。主要保持指标采用事先规定的联合判定，不能挑一项通过来代表全部通过。

按独立动物/样本做配对统计或分层 bootstrap；多切片属于同一组织时保留层级依赖。只有一个组织时只能报告该组织的折间与随机性描述，不能用大量细胞伪装生物学样本量。未发现显著下降不等于证明非劣；样本量不足时报告不确定。

## 8. 理论预期及其失效条件

| 模块 | 有条件的理论预期 | 失效条件 |
|---|---|---|
| A | 若 UOT 已在可靠支持区域额外压低尾类边际，校正会增加自然分布下被忽视细胞的监督机会 | 小质量主要代表错误候选/不重叠；校正放大坏配对，或少量 parent 重复过多 |
| B | 若给定自身状态后邻域仍包含目标状态信息，显式条件有望降低条件回归歧义；双端候选可保留不同路径 | 相关性来自绝对位置/批次记忆；邻域与类型混合污染自身 marker；端点候选错误 |
| C | 冻结主干、锚点及小残差可限制向量场漂移，并提供可选择的保持—改善权衡 | 基线尾类本就严重错误而预算过紧；速度保持未转化为输出指标保持；门控抑制全部学习 |
| 组合 | 对应瓶颈均存在时，A 提供可用监督，B 使用额外条件，C 控制影响范围，有望在尾类恢复与原指标间形成更好的权衡 | 若定位实验不支持某个瓶颈，删除该模块，不为了消融表凑组件 |

平方损失下，加入额外条件后的理想 Bayes 风险不高于不使用该条件的风险，因为模型可选择忽略它；这依赖足够表达能力、正确优化和真实分布。有限样本泛化可以更差，故不能用这个性质宣称实际模型必然提升。

## 9. 执行顺序与可保留的贡献

1. 冻结原版完整基线和数据协议，完成运输边际、来源抽样、生态位条件与稀疏化四项定位。
2. 先做 A，并对照简单重加权、支持图和来源复制；无机制证据则停止扩大运输设计。
3. 在固定配对/预算下做 B 的 simple/real/zero/shuffle 和单端/双端比较；不把所有变化同时混入。
4. 加入 C，画稀有类收益与各原指标退化的曲线，只用内层验证选择参数。
5. 在独立样本与原生三维遮蔽设置做完整比较，并冻结最终主张。

工程分工可对应：A 修改 `data_utils/uot_solver.py` 的诊断输出及 `dataset.py` 的有向配对采样；B 在独立邻域特征模块和 `models/git.py` 残差分支中实现；C 在独立控制器、`module.py` 损失和 `core.py` 推理门控中实现。每个模块单独开关、记录配置，并保存来源 ID、候选 ID、类型、方向、深度、支持状态、连续类别和稀疏化前表达以便机制复查。本次仅制定方案，没有修改这些实现。

最值得验证的贡献表述是：在不完整跨切片支持下，区分训练边际校正与真实生成丰度，利用自身—生态位条件改善稀有细胞的空间及表达恢复，并通过受约束残差控制对已有组织结构的扰动。各零件都有明确前作，组合是否形成新的实证认识取决于机制证据；目前不宣称首创、领先或已优于基线。
