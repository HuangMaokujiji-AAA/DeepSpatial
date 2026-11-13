# 外部 Marker 文件

下载 Yao et al. 2023, *A high-resolution transcriptomic and spatial atlas of cell types in the whole mouse brain* 的 **Supplementary Table 7**。它包含 cluster/subclass 归属及 overall marker 等字段；Table 5 只是基因总表，不能替代“类型→基因”的映射。

- 论文与表格说明：https://pmc.ncbi.nlm.nih.gov/articles/PMC10719114/
- PMC 下载：https://pmc.ncbi.nlm.nih.gov/articles/instance/10719114/bin/41586_2023_6812_MOESM8_ESM.xlsx
- 出版社备用：https://static-content.springer-cdn.com/esm/art%3A10.1038%2Fs41586-023-06812-z/MediaObjects/41586_2023_6812_MOESM8_ESM.xlsx

请把 XLSX 放到本目录，保留文件名 `41586_2023_6812_MOESM8_ESM.xlsx`。本次未将该外部文件打包，也未验证实际工作表表头；先检查列名，不要把任意说明列当作 Marker：

```bash
cd RA_DeepSpatial
python scripts/import_markers.py resources/41586_2023_6812_MOESM8_ESM.xlsx --inspect
python scripts/import_markers.py resources/41586_2023_6812_MOESM8_ESM.xlsx
```

自动识别失败时，按 `--inspect` 输出指定真实工作表和列名，例如：

```bash
python scripts/import_markers.py resources/41586_2023_6812_MOESM8_ESM.xlsx \
  --sheet '实际工作表名' --header 0 \
  --subclass-column '实际subclass列名' --marker-columns '实际正向marker列名'
```

输出 `markers.csv` 的格式：`subclass,gene,source,source_sha256,source_column`。同一 subclass 的 cluster Marker 合并为**外部候选池**，不宣称每个候选都对整个 subclass 特异。导入时不读本地隐藏切片，也不根据测试分数选择基因。正式程序按 Ensembl ID 或精确 gene_symbol 匹配本地面板，记录覆盖率；未覆盖类型不伪造 Marker，其空间/数量评价仍保留。

每类候选先按固定哈希次序划分，候选≥5 时每5个留出1个，不参与 C；剩余候选仅用训练切片的类型内/类型外 log1p 表达差排序，取正富集的至多8个进入损失。该筛选是对外部候选的训练期面板适配，不能称为完全独立的生物学验证。预留基因仍参与主FM的表达监督，只是未进入C，不能宣称未见基因泛化。`heldout_C`结果只针对确实有预留基因的类型，同时报告覆盖数。`marker_coverage.csv` 记录实际映射；不能把未覆盖的类型解释为 C 已经验证有效。

CSV 也可由你提供，但必须有 `subclass,gene,source` 三列。程序不会静默切换为训练差异基因或从名称猜 Marker。下载缺失不妨碍 metadata 检查和数据缓存；包含 C 的正式8配置实验必须先提供文件。
