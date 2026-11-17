# 原版 DeepSpatial：Open-ST 转移性淋巴结

脚本：`scripts/train_openst.py`。使用根目录的 `deepspatial/`，不使用 HM、RA、RC、ST 或 MM 变体，也不修改原版模型、损失、UOT 或采样代码。此任务提供训练脚本；完整训练需在资源足够的机器上执行。

## 划分与输入

- 输入：`data/Open-ST/GSE251926_metastatic_lymph_node_3d.h5ad`。
- 训练：#2、#3、#5、#7、#11、#17、#19、#23、#25、#26、#28、#33、#36。
- 验证：#6、#34。
- 测试：#4、#18、#24、#9。

按 `obs['n_section']` 选择真实物理切片编号。使用 `obsm['spatial_3d_aligned']` 的已配准 XY 和真实 Z（µm）；文件中的 `obsm['spatial']` 不是这里要用的已配准 XY。不会将切片重编号后假定等距。

输入表达采用文件现有 `X`，保留全部 28,943 个基因和全部细胞，不重复 normalize/log1p，不选 HVG、不下采样；标签采用 `obs['annotation']`。只加载需要的 X 行和元数据，不加载 `layers['raw']`。

训练、标签编码、UOT 配对和坐标归一化仅使用 13 张训练切片；验证/测试切片不作为重建参考。原版没有 validation_step，因此保留固定轮数训练和原版训练损失 checkpoint 标准；验证集用于最终模型的留出重建评估，不做 early stopping 或 checkpoint 选择。测试和验证的表达真值在各自重建后才读取。

## 参数

以 `deepspatial/core.py` 的 API 默认值为准（教程中的 10 epochs、512/2048 batch size 是示例覆盖值）：

| 项目 | 参数 |
|---|---|
| 训练 | 100 epochs，batch size 128，workers 4 |
| 配对 | n_samples_base 50000，alpha_spatial 0.5，uot_reg 0.8，uot_tau 0.05 |
| GiT | patch_size 8，hidden_size 256，depth 6，heads 8，mlp_ratio 4 |
| 优化 | AdamW，lr 2e-4，weight_decay 1e-5 |
| 损失 | 原版 Linear flow matching，lambda_g 0.1，lambda_c 10 |
| 内部默认 | velocity，train_eps/sample_eps 0.02，EMA decay 0.999 |
| ODE | dopri5，atol/rtol 1e-5，20 个输出时间点 |

固定种子 42。重建使用最终 `last.ckpt` 中的 EMA 权重；另外保留原版按训练 `loss` 排名的最佳 checkpoint。保存 Lightning checkpoint 的优化器状态及模型配置、基因列表和训练日志。

## 运行

在已安装项目依赖的 Python 环境、项目根目录执行。依赖按仓库 `pyproject.toml` 安装；可参考已有 `constraints-deepspatial.txt` 锁定版本。

```bash
# 只检查实际文件、划分和内存组件大小；只需 numpy、h5py。
python scripts/train_openst.py --inspect

# 单 GPU：训练一次，同时导出两张验证切片和四张测试切片。
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_openst.py \
  --accelerator gpu --device cuda:0 \
  --output-dir output/openst_original
```

输出目录必须不存在，避免覆盖既有训练结果。不传 `--output-dir` 时创建 `output/openst/<时间戳>/`。终端显示进度；逐步/逐轮损失写入 CSV。

如果模型训练完成但重建中断，无需重新训练：

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_openst.py \
  --reconstruct-only output/openst_original --device cuda:0 --chunk-size 256
```

此命令重新加载最终模型，沿用原实验的种子、厚度、薄层宽度和 ODE 步数；重新生成六张预测文件并覆盖相应重建结果，可调整推理设备和 chunk size。脚本在本地恢复配置与 state_dict，以避开原版 `load_checkpoint()` 将内部 transport 参数误传给 `build_model()` 的问题，库代码保持原样。

## 结果

```text
output/openst_original/
  run_parameters.json                 # 输入身份、环境版本、完整参数、划分
  split_manifest.csv                  # 19 张切片编号、细胞数、物理 Z、所属集合
  training_result.json
  checkpoints/
    config.json                       # 原版模型/训练配置、归一化统计、类别
    genes.csv                         # 完整基因列表和顺序
    last.ckpt                         # 最终模型、EMA、优化器等状态
    deepspatial-*.ckpt                 # 最佳训练损失 checkpoint
  logs/version_0/metrics.csv           # loss、loss_x、loss_g、loss_c
  test/
    reconstructed_section_04.h5ad
    reconstructed_section_18.h5ad
    reconstructed_section_24.h5ad
    reconstructed_section_09.h5ad
    section_*_metrics.json
    section_*_class_counts.csv
  validation/
    reconstructed_section_06.h5ad
    reconstructed_section_34.h5ad
    section_*_metrics.json
    section_*_class_counts.csv
  reconstruction_metrics.csv
  completed.json                      # 六张重建及评估均完成才创建
```

重建参考对：#4 → (#3,#5)，#18 → (#17,#19)，#24 → (#23,#25)，#9 → (#7,#11)，#6 → (#5,#7)，#34 → (#33,#36)。

为沿用原版生成行为，先调用 `reconstruct_between_slices()` 在两张训练切片间生成连续体积，再截取目标真实 Z ±5 µm 的薄层，分别写入 h5ad。默认体积密度参数 `thickness=10 µm`，不使用目标真值细胞数决定生成数量。因此每个 h5ad 是对应切片附近的三维薄层，Z 不强制压到同一个平面，细胞也不是与真值逐一对应的。

每个 h5ad 的 `X` 为原版生成表达，`var` 保留基因顺序，`obs` 含预测 annotation、生成物理 z_coord、目标 n_section、split，`obsm['spatial']` 为物理 XY，`obsm['spatial_3d_aligned']` 为物理 XYZ。删除原版组装时继承的源细胞注释列，以免被误当成预测。`uns['openst_protocol']` 记录参考切片、薄层范围和权重来源。

指标包括：现有 X 尺度的基因均值 MSE/MAE/PCC、双向 XY 最近邻均距、50 µm XY 覆盖率，以及真值/预测类别计数。基因均值指标不表示逐细胞配准准确率；空间指标在固定薄层上计算。

## 资源与验证

实际文件为 1,097,769 个细胞 × 28,943 个基因。最大训练相邻对为 #26/#28；**仅一张 float64 全局 UOT 矩阵约 39.17 GiB**，原版同时创建多张距离、代价和 Sinkhorn 中间矩阵，实际主存需求明显高于此值。50,000 条轨迹的两端基因张量本身约 10.78 GiB，拼接时还会复制。GiT 每个细胞有 3,618 个基因 token，训练显存需求也较高。

原版推理会将两张完整参考切片的表达转为设备上的稠密矩阵，再分 chunk 做 ODE；减小 `--chunk-size` 只能降低 ODE 批次开销，不能消除完整端点的显存占用。脚本忠实保留这些行为；普通内存/显存配置可能无法跑完全量实验。

可运行的数据读取、划分、原版默认参数对齐、薄层导出和指标检查：

```bash
python tests/test_train_openst.py
```

检查需要 numpy、h5py、anndata、scipy、pandas，不需要 GPU。导出检查使用确定性替身生成器验证流程，不代表已验证原版网络训练收敛或完成全量训练。
