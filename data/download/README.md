# 数据集下载 notebooks

每个 notebook 只负责一套数据集；不依赖先运行另一个 notebook。所有产物均在项目 `data/` 下。

| Notebook | 数据目录 |
|---|---|
| [download-ABCA-1.ipynb](download-ABCA-1.ipynb) | `data/Zhuang-ABCA-1/` |
| [download-ABCA-3.ipynb](download-ABCA-3.ipynb) | `data/Zhuang-ABCA-3/` |
| [download-MERFISH-C57BL6J-638850.ipynb](download-MERFISH-C57BL6J-638850.ipynb) | `data/MERFISH-C57BL6J-638850/` |
| [download-3D-lung-CosMx.ipynb](download-3D-lung-CosMx.ipynb) | `data/3D-lung-CosMx/` |
| [download-HTAN-WUSTL.ipynb](download-HTAN-WUSTL.ipynb) | `data/HTAN-WUSTL/` |
| [download-CRC-VisiumHD-GSE280318.ipynb](download-CRC-VisiumHD-GSE280318.ipynb) | `data/CRC-VisiumHD-GSE280318/` |
| [download-Breast-GSE243280.ipynb](download-Breast-GSE243280.ipynb) | `data/Breast-GSE243280/` |

## 使用

1. 在项目目录或 `data/download/` 打开 notebook，使用 Python ≥3.11 内核。
2. 运行环境单元格，查看/修改参数，再依次运行文件清单和下载单元格。
3. 默认保留官方表达、元数据、图像和归档；GEO 整系列 `_RAW.tar` 与逐个附件重复，因此选择逐附件下载。原始 FASTQ/BAM 不在本任务的处理数据下载范围。
4. `EXPORT_H5AD=False` 默认只下载/解压。需要额外 H5AD 时改为 `True`，运行最后的转换单元格。

每套数据使用 `official/` 保存官方文件、`metadata/` 保存来源清单、`extracted/` 保存解压副本、`h5ad/` 保存可选导出。Allen 按表达格式区分 `h5ad/raw/` 和 `h5ad/log2/`；不会改动现有 `data/Zhuang-ABCA-2/`。

HTTP 下载支持 `.part` 续传、服务器不支持 Range 时重新下载、官方 MD5 校验（Allen/Zenodo）以及长度校验。已存在的文件若不符合清单会报错并保留，不静默覆盖。下载完成后有 `.download.json` 收据；这些收据和清单不含认证 token。

## 数据集差异

- **Allen**：默认 raw 表达；`EXPRESSION_FORMATS` 可增加 log2。按细胞/基因标识符连接，按原切片导出。原切片坐标和 CCF 坐标分开保存，CCF 缺失不删除细胞。
- **CosMx**：默认获取 Zenodo 已公开 v3 的全部五个包。可选 RDS 转换需要可运行的本机 Rscript，以及 `SeuratObject`（Seurat v5）和 `Matrix`；Python 不直接伪读取 RDS。转换保留原 metadata，额外提供 `cell_type` 与 `z_coord_um`。本机现有 Rscript 在检查中返回架构错误，运行此可选步骤前需使用匹配架构的 R。
- **HTAN**：默认获取 HT268B1 的 Th1H3 组织块四张 Visium 切片（U2、U12、U22、U32）的 Level 3 文件，当前共 24 个；`TARGET_SECTIONS` 明确指定切片顺序，筛选清单保存在 `metadata/htan_3d_files.json`。根据门户快照解析 Synapse 文件 ID；运行下载 cell 后通过隐藏输入或已有环境变量 `SYNAPSE_AUTH_TOKEN` 登录，不把 token 写进代码。逐文件下载失败会写入清单并最终报错。切片间物理 Z 需另行核实。
- **CRC**：超系列 GSE280318 没有列全 Xenium 表达包，因此另解析子系列 GSE280314 的 Series-level ZIP，并按病例放回对应 GSM 目录；默认还保留参考和作者组成元数据。
- **GEO 转换**：Visium/HD 保留 full-resolution pixel XY，Xenium 使用实测微米 centroid。没有可靠物理 Z 时保持缺失，不以切片编号造 Z；`Z_UM` 可填写已核实的深度。非空间参考和无法唯一连接的矩阵会记入 `metadata/h5ad_skipped.json`，不产生伪空间坐标。

全系列 GEO 图像、Xenium 输出和解压副本可能很大。清单会在数据传输前显示，`SAMPLE_IDS` 可选择研究样本；`FILES` 可选择 CosMx 包，`INCLUDE_IMAGES` 控制 HTAN 图像。默认配置没有执行下载；只有运行对应单元格才开始传输。

## 验证

已完成 notebook JSON/schema 与所有 Python 单元格语法检查；测试覆盖 Range 续传/忽略 Range、MD5 失败保留、归档路径、GEO 系列级附件、Allen 条码/基因精确对齐、CCF 缺失保留、Visium/HD/Xenium 空间转换和 HTAN 前缀 Matrix Market 文件。

```bash
python data/download/test_download_workflow.py
```

另用官方小型 gene.csv 实际验证了 Allen 下载和 MD5，读取实际细胞/CCF 表头，核验 CRC Xenium ZIP 的 HTTP 状态，并用公开 HTAN 快照解析出所选案例的文件 ID。未执行大型表达矩阵的全量下载；未使用用户 Synapse token 登录；CosMx 完整 RDS 转换未在本机执行。
