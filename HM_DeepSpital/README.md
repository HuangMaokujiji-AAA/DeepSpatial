# HM DeepSpatial：独立研究副本

源代码由上游 `deepspatial/` 复制到本目录，原目录不参与修改。`upstream_manifest.json` 记录复制前的提交与源码 SHA-256；保留原作者和 MIT 许可。

当前实现了第一阶段 V1：可观测生态位条件、同类 UOT 配对混合抽样、低方差 parent 抽样、双向训练和生成追溯。所有新增开关默认关闭，可以使用同一份副本运行基线。新增及原先缺少说明的函数均补充了简洁注释。

## 直接运行

推荐打开 `examples/run_v1.ipynb`，选择服务器原有 DeepSpatial Python 内核后运行全部单元格。Notebook 默认按顺序独立留出 1、2、3，每轮用其余四张切片重新训练，并沿用 `configs/merfish_v1.json` 中的全部参数。结果分别保存到 `outputs/v1_holdouts_1_2_3/holdout_1/`、`holdout_2/`、`holdout_3/`。每轮输出格式与原脚本一致；已有目录不会被覆盖，重跑时在配置单元格更换 `OUTPUT_ROOT`。Notebook 默认使用 GPU 0；切换 GPU 后应重启内核。

原 Python 命令行入口继续可用：

在包含上游运行依赖的环境中执行：

```powershell
cd D:\Code\DeepSpatial\HM_DeepSpital
python examples/run_v1.py --config configs/merfish_v1.json --output outputs/v1_holdout_2 --holdout 2
```

脚本强制从本目录导入代码，输出目录也限定在副本内。`--holdout` 在读取 H5AD 前排除指定编号；没有实现自动评价隐藏真值。已有留出结果属于开发资料，独立测试集应另行固定。

默认 `training.devices=1`，在有多张 GPU 的服务器上也先使用一张卡。可用 `--devices 1` 显式覆盖旧配置；需要多卡训练时使用 `--devices 2`。DDP 子进程重新执行入口时不创建输出目录、不写运行配置，训练结束后仅主进程保存最终模型和重建结果。上一次失败留下的目录仍保留，重跑时请指定新输出目录。

配置中的切片 Z 为显式指定的 0/50/100/150/200 µm，与此前开发设置一致；使用其他数据时应改成实际测量元数据。半径 50 µm、混合比例 0.2 以及四基因程序均为可检查的初始设置，不是已经验证的最佳参数。

输出包含 `run_config.json`、`checkpoints/config.json`、`checkpoints/last.ckpt`、CSV 训练日志、`model/model.ckpt` 和 `generated.h5ad`。脚本不覆盖已有输出目录。配置里的输入路径以本项目目录为基准，输入数据只读。

## 实验 S：复用三轮权重，只修改抽样

打开 `examples/run_s.ipynb`，选择原 DeepSpatial 内核运行全部单元格。默认加载 `outputs/v1_holdouts_1_2_3/holdout_{1,2,3}/model/model.ckpt` 的已有 EMA 权重，不调用训练或重新计算 UOT。固定 10 µm 薄层中心为 0、10、20…，按薄层×类型×方向分配低方差整数预算；类型的两个方向联合舍入，保持原深度、方向和来源比例的联合期望。层内深度按原条件分布抽样，期望小于 1 时允许零计数。

结果分别写入 `outputs/s_holdouts_1_2_3/holdout_i/`，包含 `generated.h5ad`、逐组期望/整数预算 `sampling_budget.csv`、实际来源/生成类别计数 `sampling_counts.csv`、100 个纯抽样种子的 `sampling_audit.csv`，以及记录原权重 SHA-256 的 `run_config.json`。完整来源调度保存在 H5AD 的 obs 中。已有结果不会被覆盖。

服务器项目根目录也可直接运行：

```bash
CUDA_VISIBLE_DEVICES=0 python HM_DeepSpital/examples/run_s.py

# 只检查抽样，不做 ODE；使用独立的新输出目录。
python HM_DeepSpital/examples/run_s.py --audit-only \
    --output-root HM_DeepSpital/outputs/s_sampling_audit
```

可用 `--holdouts 2` 只推理一轮、`--audit-seeds 0` 跳过重复抽样检查、`--output-root` 指定新的输出目录。原训练配置和推理分支 `random/stratified` 不变。工程检查：`python HM_DeepSpital/tests/test_experiment_s.py`。

## 核心开关

```python
model.setup_data(
    observed_slices,
    num_workers=0,
    niche_config={
        "radius": 50,
        "sigma": 25,
        "programs": {"myelin_panel_proxy": ["Mbp", "Opalin", "Ermn", "Gjc3"]},
        "expression_transform": "log1p",
        "mode": "real",
    },
    balanced_fraction=0.2,
    source_sampling="stratified",
)
model.build_model(bidirectional_training=True)
```

| 对照内容 | 配置方式 |
|---|---|
| 上游基线计算路径 | `niche_config=None`、`balanced_fraction=0`、`source_sampling="random"`、`bidirectional_training=False` |
| 只做训练/来源保障 | 仅打开 `balanced_fraction` 与 `source_sampling`，不提供生态位配置 |
| 只有邻域类型组成 | 提供 `niche_config`，其中 `programs={}` |
| 类型与功能生态位 | `mode="real"` 并提供明确的程序权重 |
| 同容量打乱对照 | 同一特征维度，`mode="shuffle"`，在同类型、同切片内打乱 |
| 同容量零条件对照 | 同一特征维度，`mode="zero"` |

需要区分双向增强的收益时，各组使用相同 `bidirectional_training` 配置；上游原配置单列。程序字典也接受带符号权重，例如 `{"program": {"geneA": 1, "geneB": -1}}`。未测量的程序基因会报错，不静默缩减基因集。`log1p` 用于非负表达；已经变换过的数据可显式选择 `identity`。

程序基因的均值/尺度仅在 `setup_data()` 的可见训练切片上拟合。邻域使用原始物理 XY、排除自身，记录类型组成、邻居程序均值、加权观测量、有效邻居数和缺失标志。源条件在一条 ODE 轨迹中固定，反向训练同步交换源条件与深度。

## 模型恢复与结果追溯

```python
model.save_model("outputs/run/model")
restored = ds.DeepSpatial()
restored.load_checkpoint("outputs/run/model/model.ckpt")
result = restored.reconstruct_full_volume(
    observed_slices, thickness=10, save_intermediates=True,
)
```

恢复时重用训练期的基因顺序、类型顺序、程序统计量、物理归一化和采样配置；新切片不重新拟合这些参数。新版本仍可读取上游的权重和配置。

生成 `obs` 含 `parent_id`、`source_cell_id`、`source_z`、`source_cell_index`、`source_direction`、`source_cell_class`、`is_observed` 和 `annotation_origin`。其中来源注释不能作为隐藏组织的预测真值。物理 XYZ 保存在 `obsm["spatial_3d"]`。

打开 `save_intermediates` 后，额外保存 `layers["expression_before_sparsification"]` 及 `obsm["cell_class_continuous"]`。后者是连续类别状态，不是经校准的概率；原离散类别取值规则保持不变。

`source_sampling="stratified"` 保留原深度/方向抽样规则，按该方向的实测类型比例分配来源数量，再均衡同类 parent。它不强制每个薄层都有稀有类，也不增加稀有类的期望占比。`balanced_fraction` 只改变训练分布，推理仍使用实测来源比例。

## V2 的可运行细化接口

```python
from deepspatial.bio_prior import refine_program_expression

refined = refine_program_expression(
    generated,
    fitted_program_featurizer,
    frozen_lower_bounds,
    frozen_upper_bounds,
)
```

该函数仅修正 `is_observed=False` 的细胞和程序涉及的基因，位置、类型、其他基因和真实观测保持不动。区间内无 NPC 收缩，区间外使用软惩罚，基础表达提供锚点。输入区间必须与细胞和程序顺序一致。

这里已经实现表达优化算子；谱系共享分位数预测器的训练、三维邻域先验拟合与独立校准尚未实现，也未自动启用。调用前应在与应用邻域算子一致的真实训练/校准数据上提供冻结区间，不能用隐藏测试表达计算边界。DIALOGUE 的公开程序也尚未导入，示例四基因集合明确标为面板代理程序。

## 验证

```powershell
python tests/test_pipeline.py
```

检查内容：排除自身、邻域缺失、来源预算、同类配对混合、双向条件一致性、短训练和 ODE、保存/加载、H5AD 写回、DDP 子进程不创建或覆盖输出目录、V2 仅修改指定生成表达，以及关闭开关时与上游计算路径的数值一致性。这些是工程检查，不代表生物学效果已提升。

2026-10-26 验证记录：8 项回归检查通过；另用现有 MERFISH 的 155 基因小子集执行了示例入口的单轮训练、留出文件排除、模型保存和重建，结果可正常写入/读取 H5AD。修复多卡入口后，额外用微型模拟切片完成本机两进程 CPU DDP 单轮训练、主进程模型保存和重建检查，生成 12 个细胞；未在服务器验证双 GPU。这些小规模检查的临时数据与模型已清理，未替代完整训练和独立效果评价。

本机验证所缺的现有项目依赖安装在本目录 `.runtime_deps/`，未修改全局环境；仅 Windows 测试和示例入口会使用它。Linux 服务器使用自己的 Conda 依赖，上传项目时无需上传该目录。项目的分发名称是 `deepspatial-hm`，导入名称仍是 `deepspatial`；需要安装时请使用单独环境，避免与上游分发覆盖同一导入名。

## Linux 服务器的 CUDA 动态库路径

如果在 `import torch` 阶段出现 `libcusparse.so.12: undefined symbol: __nvJitLinkComplete_12_4`，当前加载的 nvJitLink 未提供 cuSPARSE 所需符号。项目原有 Notebook 曾通过优先加载当前环境中的 nvJitLink 解决相同错误；终端进程也需要在 Python 启动前设置该路径。

在项目根目录、已激活服务器 Conda 环境时运行：

```bash
# 查看 PyTorch 是否可以正常导入。
bash HM_DeepSpital/scripts/run_python.sh -c 'import torch; print(torch.__version__, torch.version.cuda)'

# 无参数默认执行测试。
bash HM_DeepSpital/scripts/run_python.sh

# 同一入口可启动训练。
bash HM_DeepSpital/scripts/run_python.sh HM_DeepSpital/examples/run_v1.py \
    --output HM_DeepSpital/outputs/v1_holdout_2 --holdout 2
```

脚本通过当前 Python 的 `sysconfig` 查找库目录，在启动 Python 前调整子进程的 `LD_LIBRARY_PATH`，不重装 PyTorch、不修改系统 CUDA。可用 `PYTHON=/path/to/env/bin/python bash ...` 指定解释器。

也可在当前终端手动设置：

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nvjitlink/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python HM_DeepSpital/tests/test_pipeline.py
```

上面的手动路径对应已报告的 Python 3.10 环境，通用启动脚本会自动适配 Python 版本。若依旧失败，先检查环境中的实际库文件与包版本：

```bash
python -m pip show torch nvidia-nvjitlink-cu12 nvidia-cusparse-cu12
find "$CONDA_PREFIX" -name 'libnvJitLink.so*' -o -name 'libcusparse.so*'
```

路径优先级调整只能解决兼容库已存在但没有优先加载的情况；库缺失或本身版本不匹配时，需要根据检查结果处理依赖。此问题发生于测试启动前，切换测试到 CPU 也仍然需要先成功导入 PyTorch。

---

以下保留上游原始说明和引用信息：

<p align="center">
  <img src="docs/source/_static/text_logo.png" width="200" alt="DeepSpatial">
</p>

<p align="center">
  <a href="https://pypi.org/project/deepspatial/"><img src="https://img.shields.io/badge/Pypi-1.0.0-317EC2.svg" alt="PyPI"></a>
  <a href="https://yyh030806.github.io/DeepSpatial/"><img src="https://img.shields.io/badge/Homepage-deepspatial-f773a8.svg" alt="Homepage"></a>
  <a href="https://yyh030806.github.io/DeepSpatial/docs/"><img src="https://img.shields.io/badge/Documentation-latest-4CAF50.svg" alt="Docs"></a>
  <a href="https://doi.org/10.64898/2026.04.28.721395"><img src="https://img.shields.io/badge/Paper-bioRxiv-00AAB5.svg" alt="BioRxiv"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-FF9800.svg" alt="License"></a>
</p>


**DeepSpatial** is a package for true 3D reconstruction of spatial omics tissues from serial 2D slices, built with `PyTorch` and designed to work smoothly with `AnnData`/`Scanpy` workflows.

## 3D spatial omics reconstruction

DeepSpatial provides an end-to-end framework for learning continuous 3D tissue representations:

- Reconstructs missing biological structure between adjacent sections.
- Jointly models spatial coordinates, gene expression, and cell identities.
- Supports large-scale training and sampling with GPU acceleration.
- Produces outputs that can be directly used in downstream single-cell/spatial analysis.

The package exposes a high-level API (`DeepSpatial`) for data setup, model training, and 3D reconstruction with minimal boilerplate.

## Installation

Recommended (PyPI):

```bash
pip install deepspatial
```

From source (development):

```bash
git clone https://github.com/yyh030806/DeepSpatial.git
cd DeepSpatial
pip install -e .
```

If you use GPU, install a PyTorch build matching your CUDA version.

## Quick start

1. Download an example dataset first: [Google Drive dataset folder](https://drive.google.com/drive/folders/11MICO4KmGRDlKfFf6LQ_aZC3mtF9NrNG?usp=sharing)

   Then place files under `data/merfish_mouse_hypothalamus/`.

2. Run DeepSpatial:

```python
import glob
import scanpy as sc
import deepspatial as ds

adatas = [
    sc.read_h5ad(p)
    for p in sorted(glob.glob("data/merfish_mouse_hypothalamus/merfish_*.h5ad"))
]

model = ds.DeepSpatial()
model.setup_data(adatas)
model.build_model()
model.fit(max_epochs=100)

adata_3d = model.reconstruct_full_volume(adatas, thickness=10.0)
```

## Resources

- Documentation, tutorials, and API reference: https://yyh030806.github.io/DeepSpatial/docs/
- Homepage: https://yyh030806.github.io/DeepSpatial/
- Bug reports and feature requests: https://github.com/yyh030806/DeepSpatial/issues

## Citation

If you use DeepSpatial in your research, please cite:

```bibtex
@article {yang2026deepspatial,
	author = {Yang, Yuhang and Luo, Yiming and Zhang, Kai and Bu, Yonggan and Xia, Zheng and Peng, Haoxin and Yan, Rui and Liu, Qi and Chen, Yang and Shen, Lin and Chen, Enhong},
	title = {Reconstructing True 3D Spatial Omics at Single-Cell Resolution},
	year = {2026},
	doi = {10.64898/2026.04.28.721395},
	journal = {bioRxiv}
}
```

## License

DeepSpatial is released under the MIT License.
