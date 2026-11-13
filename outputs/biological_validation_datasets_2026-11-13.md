# 稀有群体三维恢复：数据集筛选与获取清单

## Material Passport

- 日期：2026-11-13。
- 工作流：academic-research-suite / deep-research，定向数据检索与来源核验。
- 状态：核验了论文、官方数据说明、公开文件列表和作者代码中的注释字段；未下载大型表达矩阵，未验证候选群体的实际稀有度。
- 目标：脑稀有神经元亚区富集；肿瘤界面低丰度免疫群体；区域/微环境划分的增量价值。
- 原则：每个数据集按可承担的证据角色选择，不把跨平台相邻切片自动当作密集同平台三维序列。

## 1. 优先组合

| 优先级 | 数据集 | 建议角色 | 主要条件/限制 |
|---|---|---|---|
| A1 | Zhuang-ABCA-1，配合现有 ABCA-2 | 脑主实验和跨动物验证 | 同 1122 基因面板、同类型层级；需独立划分动物与开发集 |
| A2 | 3D lung CosMx，Pentimalli et al. 2025 | 单细胞肿瘤恢复与生态位案例 | 真正的六平面序列、公开处理对象；单患者，需要外部复现 |
| A3 | HTAN WUSTL / Mo et al. 2024，优先 HT206B1 | Visium 连续切片重建主候选 | 原发乳腺癌；需获取切片清单、精确 Z 间距及处理数据 |
| B1 | CRC Visium HD + Xenium，GSE280318 | 界面免疫群体及局部组成的正交验证 | 配套参考、边缘注释；目前不能据来源认定有充分同平台 Z 序列 |
| B2 | Breast Visium + Xenium + scFFPE-seq，GSE243280 | Visium 解卷积与身份/空间模式验证 | 有匹配多模态信息，但仅一个 Visium 样本条目，不作为 3D 主数据 |
| C1 | Zhuang-ABCA-3 | 脑跨方向验证 | 23 张矢状切片，1122 基因；不能照搬冠状 Z 轴含义 |
| C2 | MERFISH-C57BL6J-638850 | 脑跨研究/跨面板验证 | 500 基因、较大；需要核验目标 marker 的实测面板交集 |

这是基于资料完整性和实验适配性的筛选。没有任何一个候选已被证明满足目标亚群在训练端点中“低丰度但有足够支持”的条件，正式使用前都需要元数据计数。

## 2. 脑：Zhuang-ABCA-1 与 ABCA-3

### Zhuang-ABCA-1：新增脑数据的第一优先级

官方说明为 147 张冠状切片、1122 基因，与本地 ABCA-2 同体系。表达矩阵与高置信度注释细胞范围不同：说明页分别约 420 万和 280 万，必须按 cell_label 对齐，不能仅按矩阵行数拼接。

- 官方数据页：[Zhuang-ABCA-1](https://alleninstitute.github.io/abc_atlas_access/descriptions/Zhuang-ABCA-1.html)。
- 获取流程：[官方 MERFISH 教程](https://alleninstitute.github.io/abc_atlas_access/notebooks/zhuang_merfish_tutorial.html)。
- 下载组件：实测表达矩阵、cell metadata、Zhuang-ABCA-1-CCF 坐标、共享 WMB-taxonomy、Allen-CCF-2020 分区对应表。
- 说明页体积：表达 3.09 GB、metadata 1.33 GB、CCF 0.21 GB；实际文件和压缩状态以 manifest 为准。
- 使用建议：ABCA-2 开发、ABCA-1 验证，或相反；固定后不得再根据测试动物挑选最有利亚型。共享分类体系有利于对照，但共享标签转移来源不等于独立技术注释。
- 首批筛选类型：092 TMv-PMv Tbx3 Hist-Gaba、128 VMH Fezf1 Glut；本地 ABCA-2 已确认可行性，ABCA-1 中的数量及覆盖尚待检查。
- 已知坐标问题：官方提示部分前端切片 Z 被截为 0；该部分应核实真实切片位置，不直接纳入严格递增深度的训练序列。

### Zhuang-ABCA-3：跨方向验证

官方为 23 张矢状切片、1122 基因，具备相同层级标签及 CCF 资料。

- [官方说明与下载入口](https://alleninstitute.github.io/abc_atlas_access/descriptions/Zhuang-ABCA-3.html)。
- 获取同样的表达、cell metadata、CCF、分类和解剖对应表。
- 说明页体积：表达 1.69 GB、metadata 0.75 GB、CCF 0.12 GB。
- 优先用于检验空间富集是否跨动物、跨切片方向保持，不直接混合冠状和矢状切片训练一个“相同 Z 轴”序列。
- ABCA-4 官方仅 3 张矢状切片，因此不优先用于主要缺失切片实验。[ABCA-4 说明](https://alleninstitute.github.io/abc_atlas_access/descriptions/Zhuang-ABCA-4.html)

### 跨研究备选：MERFISH-C57BL6J-638850

该数据是独立研究来源的整脑单细胞 MERFISH，500 基因，提供类型与解剖配套资源。适合检验在不同面板中的目标亚群富集与脑分区，但不必首批全量下载。

- [官方数据页](https://alleninstitute.github.io/abc_atlas_access/descriptions/MERFISH-C57BL6J-638850.html)。
- 获取实测表达和配套重建/CCF 坐标；先检查 Hdc、Tbx3、Fezf1 等候选标记是否实测。
- 官方列出的表达组件约 14.2 GB。不要把另一个 imputed 数据包中的推断基因当作独立实测真值。

## 3. 肿瘤单细胞主候选：3D lung CosMx

**论文：** Pentimalli et al., *Combining spatial transcriptomics and ECM imaging in 3D for mapping cellular interactions in the tumor microenvironment*, Cell Systems, 2025。该研究是一个肺癌标本的多模态三维案例。[论文](https://doi.org/10.1016/j.cels.2025.101261)

来源核实：CosMx 测量第 4、10、16、22、28、34 张组织切片，测量平面间距 30 μm；共 340,644 个分割细胞，1000-plex 测量。另有 H&E、SHG 和 IF；不同模态的采集切片不同，IF 不能默认验证所有免疫亚群。[作者机构全文](https://edoc.mdc-berlin.de/id/eprint/25290/1/25290oa.pdf)

作者 R 笔记本明确使用 `celltypes`、`section`、`x_3D_mn`、`y_3D_mn`、`z_3D_mn`，并含 `Regulatory T cells`、`Dendritic cells`、`Mast cells`、`Cytotoxic T cells` 等类型，以及 Tumor surface、T cell niches、Dendritic cell niches 等生态位名称。[注释与分析代码](https://github.com/rajewsky-lab/3D_lung/blob/main/code_availability.Rmd)

**公开下载：** [Zenodo v3，记录 15240431](https://zenodo.org/records/15240431)。注意早期版本 7899174 的检索片段仍提示 peer-review reserved；使用已公开 v3。

| 文件 | 体积 | 首批是否需要 |
|---|---:|---|
| `preprocessed_objects.zip` | 4.2 GB | 是；包含作者建议使用的 Seurat 对象，代码中主对象为 `cosmx.rds` |
| `analysis_notebooks.zip` | 53.2 MB | 是；检查处理步骤和数据字段 |
| `stimwrap_files.zip` | 8.3 MB | 建议；核验配准相关输入/输出 |
| `SHG.zip` | 2.8 MB | 做 ECM 背景验证时需要 |
| `cosmx_flat_files.zip` | 4.7 GB | 第二批；用于原始表达、坐标和处理步骤核验 |

**适用设计：** 用 section_4、16 等可见平面预测被隐藏的 section_10；或用 10、22 预测 16。先核验共同有效组织覆盖、配准及候选群体数量，后固定正式折。可在真实单细胞位置上测量恢复，也可聚合成模拟 spot，独立检查低丰度解卷积/恢复链条，但模拟 spot 不是实测 Visium。

**限制：** 只有一个患者；作者生态位来自其数据分析，不能作为完全独立的金标准。肿瘤表面也不自动等同于病理确认的侵袭前沿。原文 IF 针对的验证内容不等于已经获得 FOXP3/CD4 或 cDC1 的独立蛋白真值。Treg 和 DC 是否足够稀有、是否可细分 cDC1，都需下载后检查。

## 4. 肿瘤 Visium 主候选：HTAN WUSTL

**论文：** Mo et al., *Tumour evolution and microenvironment interactions in 2D and 3D space*, Nature, 2024。[论文](https://doi.org/10.1038/s41586-024-08087-4)

该研究中有实际连续组织块，而不是所有多患者切片都属于 Z 序列。论文描述了 15 个连续组织块的 48 张可用于三维分析切片；正式选样必须按组织块和采集顺序筛选。

| 组织/案例 | 文献明确的信息 | 本项目角色 |
|---|---|---|
| `HT206B1` | 原发乳腺癌，文中描述 5 张连续切片 | 优先核验的原发乳腺癌 Visium 序列候选；补精确 assay-ID、Z、覆盖 |
| `HT268B1` | 乳腺癌肝转移案例；三维模型使用 4 张 Visium | 可做中间平面恢复；生物学描述为转移灶–肝组织/间质界面 |
| `HT397B1` | 三维多模态实例使用 6 H&E、4 CODEX、2 Visium | 蛋白/组成空间验证；仅这 2 张 Visium 不足以建立中间 Visium 留出实验 |

样本更细的模态配置可能随组织块、公开批次和分析子集不同，以上不替代逐文件 manifest。

**获取路径：**

1. [作者 Data_access 目录](https://github.com/ding-lab/ST_subclone_publication/tree/main/Data_access) 中的 `Sample_ID_Lookup_table_v1.xlsx`，将上述样本名转换为 HTAN Biospecimen/Participant ID。
2. 进入 [HTAN Phase 1 数据入口](https://humantumoratlas.org/explore/phase1)，筛选 HTAN WUSTL、目标组织块和 Visium/多重成像。
3. 优先获取 processed Level 3/4：表达矩阵、spot 坐标、H&E、组织掩膜、可获得的注释和图像配准信息。另取论文 Supplementary Table 1 的切片/临床信息，以及相应 CODEX panel 表。
4. [作者主仓库说明](https://github.com/ding-lab/ST_subclone_publication) 和 [Mushroom 历史分析版本](https://github.com/ding-lab/mushroom/tree/subclone-resubmission) 可帮助追踪原分析子集。

**权限：** HTAN 处理数据和多重图像属于开放数据范畴，Synapse 路线通常需免费账户；原始测序 FASTQ/BAM 属受控访问。本文不把“归入 dbGaP”误写成“所有处理矩阵都不可公开获取”。具体目标文件是否已经公开及其路径尚未逐个下载测试。[官方权限说明](https://docs.humantumoratlas.org/data_access/introduction/)、[下载说明](https://docs.humantumoratlas.org/data_access/portal/)

**重要边界：** CODEX panel 必须实查；有 CD3/CD8 不等于能验证 Treg 或 cDC1。也不能拿 HT397B1 的染色当作 HT206B1 同一切片的真值；它是独立案例复现。

## 5. 肿瘤正交验证优先：CRC Visium HD + Xenium

**论文：** Oliveira et al., *High-definition spatial transcriptomic profiling of immune cell populations in colorectal cancer*, Nature Genetics, 2025。[论文](https://doi.org/10.1038/s41588-025-02193-3)

**入口：** [10x 数据集](https://www.10xgenomics.com/products/visium-hd-spatial-gene-expression/dataset-human-crc)、[GEO GSE280318](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE280318)、[作者代码与注释](https://github.com/10XGenomics/HumanColonCancer_VisiumHD)。

优先关注 P1CRC、P2CRC、P5CRC：论文对这些病例有相邻/近邻切片的 Visium HD 与 Xenium，并有单细胞参考资源。它非常适合验证免疫亚群的区域分布和局部组成。

作者 README 明确提供：

- `SingleCell_MetaData.csv.gz`：`Patient`、`QCFilter`、`Level1`、`Level2` 等。
- `P1CRC_Metadata.parquet` 等：`X`、`Y`、`DeconvolutionLabel1/2`、`Periphery`、`MacrophageSubtype` 等。
- `Periphery` 描述原研究 50 μm 肿瘤周边区；`MacrophageSubtype` 区分相关的 SPP1/SELENOP 状态。它们是作者分析注释，独立验证要用未输入模型的 Xenium/组织证据。[字段说明](https://github.com/10XGenomics/HumanColonCancer_VisiumHD#metadata)

**用途：** 检验前沿附近免疫/髓系组成，比较真实高分辨率数据与降采样结果；验证相邻 Xenium 中的定位。可以重点评估小范围免疫群体，但 SPP1 群是否稀有必须实测，不能预设。

**限制：** 文献中的“serial sections”用于跨技术对照，不足以认定每个患者都有至少三个独立实测 Visium HD 深度。也不能把 2 μm 的 bin 当成自然独立细胞。跨平台数据主要提供区域尺度正交验证，不作为密集 3D 主基准，除非后续 manifest 证明存在足够 Z 序列。

## 6. 易获取的第二验证集：乳腺癌 GSE243280

**论文：** Janesick et al., *High resolution mapping of the tumor microenvironment using integrated single-cell, spatial and in situ analysis*, Nature Communications, 2023。[论文](https://doi.org/10.1038/s41467-023-43458-x)

**入口：** [GEO GSE243280](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE243280)、[10x 配套数据](https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast)。

优先核对组合：Visium `GSM7782699`、scFFPE/Flex `GSM7782698`、Xenium `GSM7780153` 与 `GSM7780154`。按原文样本匹配关系再选择；SuperSeries 中其余样本不能默认同一组织块。

GEO 当前列 7 个样本条目，其中仅 1 个 Visium、3 个 Xenium，其余为单细胞。整包 supplementary archive 约 31.8 GB，可按子系列/样本文件挑选。[GEO 列表](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE243280)

**用途：** 检验 Visium 亚群组成估计；使用相邻 Xenium 作为未输入模型的空间测量，比较肿瘤/间质界面与局部组成。适合补充乳腺癌应用链条。

**限制：** 不是至少三张同平台 Visium 深度序列，不能仅凭这套数据完成中间 Visium 切片重建。313 基因 Xenium 面板对稀有免疫亚群的分辨能力需单独核验；不承诺一定能分 Treg/cDC1。

## 7. 暂不放入核心组合的数据

- HER2ST：有 36 张 ST 切片、H&E 与病理/组成资料，适合肿瘤区域和免疫浸润的补充验证。但当前核验材料没有确认可直接使用的精确 Z 序列，不以文件名 A1/A2 自动推定物理相邻。[作者资料](https://github.com/almaan/her2st)
- 两张相邻 Visium 的 CRC 技术重复队列：可做重复性验证，但不足以留出中间同平台切片。
- 普通 scRNA-seq 肿瘤图谱：适合作为类型签名参考，不能提供空间真值。
- Imputed 表达、作者输出的生态位、模型预测的 IHC：可以作辅助，不能替代独立原始测量。

## 8. 建议下载顺序与资格检查

**第一批：** ABCA-1 的 metadata/CCF 与必要表达；肺癌 CosMx 的处理对象、分析笔记本和配准资料。它们最接近当前单细胞恢复任务。

**第二批：** HTAN 样本对照表与 HT206B1 processed Visium、H&E、Z 清单；同时按可用权限挑选 HT397B1 CODEX。若坚持只做 Visium 肿瘤主实验，则把该批移到第一批。

**第三批：** CRC 多模态验证资料；若主癌种锁定乳腺癌则改为 GSE243280。ABCA-3、独立 500 基因脑数据放在扩展验证阶段。

下载后统一检查：

1. 分清患者、组织块、切片、ROI、技术重复，建立显式关系表。
2. 核对实际深度、测量平面间距、方向和共同组织覆盖。
3. 统计每个候选类型在各切片的数量、母谱系占比和置信度；冻结正式目标。
4. 核对原始/归一化/推断表达、marker 的实测面板交集。
5. 将训练先验与正交评价证据隔离；作者全切片配准或生态位分析若使用隐藏表达，需要重新计算或明确为转导设定。
6. 先做一个三平面试点，再决定全量下载和跨患者扩展。

## 9. 检索与核验记录

- 范围：Allen 官方数据说明；原始论文与 Data availability；GEO 样本列表；Zenodo 已公开版本文件清单；作者 GitHub 注释与字段说明；HTAN 官方访问文档。
- 检索词覆盖 `serial/consecutive sections + Visium + tumor`、`3D CosMx lung ECM`、`Visium HD Xenium CRC`、`Zhuang-ABCA coronal sagittal`。
- 同一论文的 DOI、Nature、PMC、机构全文去重；未将综述/数据汇编当作最终规模或模态配置依据。
- 这是一份按实验资格定向筛选的 shortlist，不是穷尽式系统综述；没有编造 PRISMA 命中数。
- 10x 动态网页在本次工具中常无可读正文，相关配置由 GEO、原论文和作者仓库交叉支持；没有冒充逐个大文件下载成功。
- 最值得优先补齐的未决项：HT206B1 逐切片 manifest/Z；HT397B1 CODEX panel 是否包含所选亚群必要标记；CosMx Treg/DC 的真实丰度；CRC 配对文件与切片顺序。
