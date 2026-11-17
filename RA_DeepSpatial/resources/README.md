# 外部 Marker 文件

来源为 Yao et al. 2023, *A high-resolution transcriptomic and spatial atlas of cell types in the whole mouse brain* 的 **Supplementary Table 7**。已在本地核实并转换实际工作表。`subclass_annotation` 正好有338个subclass，使用 `subclass_id_label`、`subclass.markers.combo`、`subclass.tf.markers.combo` 三列，直接提取subclass级外部Marker候选，不再合并cluster级列表。Table 5只是基因总表，不能替代类型映射。

**本目录已生成可直接上传服务器的 `markers.csv`（1,892条唯一subclass–gene记录，覆盖338个subclass）。服务器只需这个CSV，正式训练不需要openpyxl或原始XLSX。** POT的`ot`模块只用于工程测试中的求解器对照，正式训练使用项目自己的UOT实现。

- 论文与表格说明：https://pmc.ncbi.nlm.nih.gov/articles/PMC10719114/
- PMC 下载：https://pmc.ncbi.nlm.nih.gov/articles/instance/10719114/bin/41586_2023_6812_MOESM8_ESM.xlsx
- 出版社备用：https://static-content.springer-cdn.com/esm/art%3A10.1038%2Fs41586-023-06812-z/MediaObjects/41586_2023_6812_MOESM8_ESM.xlsx

需要重新生成时，把 XLSX 放到本目录，保留文件名 `41586_2023_6812_MOESM8_ESM.xlsx`；脚本现在默认识别上述实际工作表和列：

```bash
cd RA_DeepSpatial
python scripts/import_markers.py resources/41586_2023_6812_MOESM8_ESM.xlsx --inspect
python scripts/import_markers.py resources/41586_2023_6812_MOESM8_ESM.xlsx
```

也可以显式指定已核实的参数（输出已存在时会拒绝覆盖）：

```bash
python scripts/import_markers.py resources/41586_2023_6812_MOESM8_ESM.xlsx \
  --sheet subclass_annotation --header 0 \
  --subclass-column subclass_id_label \
  --marker-columns subclass.markers.combo subclass.tf.markers.combo
```

输出 `markers.csv` 的格式：`subclass,gene,source,source_sha256,source_sheet,source_column`。已保留以数字开头的合法Rik基因名。CSV是外部候选池，不宣称其中所有候选都在本地面板中或对该subclass独占特异。导入时不读本地隐藏切片，也不根据测试分数选择基因。正式程序按 Ensembl ID 或精确 gene_symbol 匹配本地面板，记录覆盖率；未覆盖类型不伪造 Marker，其空间/数量评价仍保留。

每类候选先按固定哈希次序划分，候选≥5 时每5个留出1个，不参与 C；剩余候选仅用训练切片的类型内/类型外 log1p 表达差排序，取正富集的至多8个进入损失。该筛选是对外部候选的训练期面板适配，不能称为完全独立的生物学验证。预留基因仍参与主FM的表达监督，只是未进入C，不能宣称未见基因泛化。`heldout_C`结果只针对确实有预留基因的类型，同时报告覆盖数。`marker_coverage.csv` 记录实际映射；不能把未覆盖的类型解释为 C 已经验证有效。

CSV 也可由你提供，但必须有 `subclass,gene,source` 三列。程序不会静默切换为训练差异基因或从名称猜 Marker。下载缺失不妨碍 metadata 检查和数据缓存；包含 C 的正式8配置实验必须先提供文件。

本地还执行了 `python scripts/check_markers.py`：只用58张训练切片的表达统计执行与正式代码相同的Marker筛选，生成 `marker_coverage.csv` 和 `marker_audit.json`；不读取验证/测试表达、不训练模型。
