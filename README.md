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

## Visualize a saved reconstruction

The standalone viewer reads coordinates and cell classes from an existing H5AD;
it does not load a model or require a GPU. Use the project's Python environment
(the script uses NumPy, h5py, Matplotlib, and Plotly):

```bash
python visualize_reconstruction.py --open
```

By default it reads `output/deepspatial_3d.h5ad` and writes an offline interactive
`reconstruction_viewer.html`, `reconstruction_overview.png`, `virtual_slices.png`,
and full cell-class counts to `output/reconstruction_visualization/`. The HTML
supports rotating the 3D cloud, toggling classes, moving through XY slabs along Z,
and viewing all rare-class cells separately. It embeds Plotly for offline use.

```bash
python visualize_reconstruction.py \
  --input output/deepspatial_3d.h5ad \
  --output-dir output/reconstruction_visualization \
  --max-points 60000 --slab-width 10 --z-stretch 5
```

`--max-points 0` displays the full global point cloud. Slabs, rare-class views,
and counts always use all relevant cells. `--z-stretch` changes the displayed
aspect ratio only; click **Physical aspect** to restore physical proportions.
The rare threshold defaults to <1% of this generated file's cells, which differs
from defining rare classes using training-set counts during holdout evaluation.

## Compare saved SpatialZ and DeepSpatial results

```bash
python compare_spatialz.py --spatialz-root ../SpatialZ --include-variants
python tests/test_compare_spatialz.py
```

This compares SpatialZ's saved `merfish_holdout_009/generated_009.h5ad` with
DeepSpatial's `holdout_1/predicted_holdout.h5ad` against the identical held-out
slice. It checks the SpatialZ input hashes, truth identity, gene alignment, and
thin-slab bounds before computing the metrics. Default DeepSpatial run:
`output/merfish_holdout/20261018_021947_653667` (override with
`--deepspatial-run`). Results go to `output/spatialz_comparison_20261018`.

The main MSE, MAE, and Pearson PCC compare gene-mean expression profiles on both
the provided X scale and a shared log1p scale. Rare-class XY coverage recall at
50/100 µm uses a fixed training-defined class set and includes completely missing
classes as zero. A separate file reports expression error after XY nearest-point
association; it does not imply cell identity. `--include-variants` adds the saved
joint gene/cell sampling variants to a separate table, preserving the original
SpatialZ result as the main comparison. No models are retrained.

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
