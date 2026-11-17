"""Optional conversions; preserve original files, labels and coordinate units."""

from __future__ import annotations

import gzip
import json
import re
import shutil
import subprocess
from decimal import Decimal, InvalidOperation
from pathlib import Path


def _write(adata, target, provenance):
    import anndata as ad
    import numpy as np

    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError("导出索引必须唯一。")
    if "spatial" in adata.obsm:
        xy = np.asarray(adata.obsm["spatial"])
        if xy.shape != (adata.n_obs, 2) or not np.isfinite(xy).all():
            raise ValueError("空间坐标必须是完整、有限的 N×2 数组。")
    if "z_coord_um" in adata.obs and not np.isfinite(adata.obs["z_coord_um"].to_numpy(dtype=float)).all():
        raise ValueError("已提供的物理 Z 含非有限值。")

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    source = json.dumps(provenance, ensure_ascii=False, sort_keys=True)
    if target.exists():
        existing = ad.read_h5ad(target, backed="r")
        try:
            if (existing.shape != adata.shape or not existing.obs_names.equals(adata.obs_names)
                    or not existing.var_names.equals(adata.var_names)
                    or existing.uns.get("download_provenance") != source):
                raise ValueError(f"已有 H5AD 与当前导出配置不同：{target}；请另选输出目录。")
        finally:
            existing.file.close()
        print("已验证，跳过", target.name)
        return target
    adata.uns["download_provenance"] = source
    # Plain Python strings/categories avoid pandas extension-string H5AD issues.
    for frame in (adata.obs, adata.var):
        for column in frame.columns:
            if str(frame[column].dtype).startswith("string"):
                frame[column] = frame[column].astype(object)
    temporary = target.with_name(target.stem + ".part.h5ad")
    adata.write_h5ad(temporary, compression="gzip")
    temporary.replace(target)
    print("导出", target.name, adata.shape)
    return target


def _csv_index(path, index):
    import pandas as pd

    result = pd.read_csv(path, dtype={index: str}, low_memory=False).set_index(index)
    if not result.index.is_unique:
        raise ValueError(f"重复标识符：{path}")
    return result


def export_allen(root, records, expression_format="raw"):
    import anndata as ad
    import numpy as np
    import pandas as pd

    root = Path(root)
    files = {r["role"]: root / r["path"] for r in records}
    cells = _csv_index(files["cells"], "cell_label")
    ccf = _csv_index(files["ccf"], "cell_label").rename(columns={k: k + "_ccf" for k in ("x", "y", "z")})
    cells = cells.join(ccf[[k for k in ccf if k not in cells]], how="left")
    cells["ccf_matched"] = cells.index.isin(ccf.index)
    if "reconstructed_coordinates" in files:
        coordinates = _csv_index(files["reconstructed_coordinates"], "cell_label")
        # The C57BL6J matrix metadata uses x/y/z_section rather than ABCA x/y/z.
        for key in coordinates:
            if key not in cells:
                cells[key] = coordinates[key].reindex(cells.index)
    xy_keys = next((p for p in [("x", "y"), ("x_reconstructed", "y_reconstructed"), ("x_section", "y_section")]
                    if set(p).issubset(cells)), None)
    if xy_keys is None:
        raise KeyError(f"没有原切片 XY 坐标，实际列：{list(cells)}；不以 CCF 坐标替代。")
    if "brain_section_label" not in cells:
        raise KeyError("缺少 brain_section_label，不能按切片导出。")
    for column in cells.select_dtypes(include="object"):
        if column != "cell_label":
            cells[column] = cells[column].astype("category")
    genes = pd.read_csv(files["genes"], low_memory=False)
    source = ad.read_h5ad(files["expression_" + expression_format], backed="r")
    outputs, summary = [], []
    try:
        if not source.obs_names.is_unique or not source.var_names.is_unique:
            raise ValueError("源表达矩阵含重复索引。")
        gene_key = next((key for key in ["gene_identifier", "gene_symbol", "ensembl", "gene_id"]
                         if key in genes and genes[key].astype(str).is_unique
                         and source.var_names.isin(genes[key].astype(str)).all()), None)
        if gene_key is None:
            raise ValueError("gene.csv 与表达矩阵 var_names 不匹配，拒绝按行位置硬拼。")
        annotations = genes.assign(**{gene_key: genes[gene_key].astype(str)}).set_index(gene_key).reindex(source.var_names)
        available = cells.index.isin(source.obs_names)
        print(f"高置信度元数据 {len(cells):,}；表达可匹配 {available.sum():,}；无 CCF 的细胞保留。")
        if not available.all():
            raise ValueError("部分元数据细胞缺少表达，先核对 manifest 和版本。")
        for section, metadata in cells.groupby("brain_section_label", observed=True, sort=True):
            rows = source.obs_names.get_indexer(metadata.index)
            order = np.argsort(rows)  # h5py dense backed indexing requires ascending positions.
            rows, metadata = rows[order], metadata.iloc[order]
            piece = source[rows, :].to_memory()
            piece.obs = metadata.copy()
            piece.var = source.var.copy()
            piece.var[gene_key] = source.var_names.to_numpy()
            for column in annotations:
                piece.var[column] = annotations[column].to_numpy()
            if "gene_symbol" in piece.var and piece.var["gene_symbol"].notna().all():
                piece.var_names = piece.var["gene_symbol"].astype(str)
                piece.var_names_make_unique()
            piece.obsm["spatial"] = metadata[list(xy_keys)].to_numpy(dtype=float)
            if not np.isfinite(piece.obsm["spatial"]).all():
                raise ValueError(f"{section} 含缺失空间坐标。")
            piece.uns["coordinate_system"] = "aligned_section_native"
            piece.uns["coordinate_units"] = "source_native; not rescaled"
            piece.uns["expression_format"] = expression_format
            suffix = re.search(r"\.(\d+)$", str(section))
            filename = "sec_" + (suffix.group(1).zfill(3) if suffix else re.sub(r"[^\w.-]", "_", str(section))) + ".h5ad"
            target = root / "h5ad" / expression_format / filename
            outputs.append(_write(piece, target, {"matrix": files["expression_" + expression_format].name,
                                                 "format": expression_format, "section": str(section)}))
            summary.append({"section": str(section), "file": str(target.relative_to(root)), "n_obs": piece.n_obs,
                            "n_vars": piece.n_vars, "ccf_matched": int(piece.obs["ccf_matched"].sum())})
            del piece
    finally:
        source.file.close()
    pd.DataFrame(summary).to_csv(root / "metadata" / "h5ad_sections.csv", index=False)
    return outputs


def read_10x_h5(path):
    import anndata as ad
    import h5py
    import pandas as pd
    from scipy.sparse import csc_matrix

    with h5py.File(path) as file:
        matrix = file["matrix"]
        values = csc_matrix((matrix["data"][:], matrix["indices"][:], matrix["indptr"][:]), shape=tuple(matrix["shape"][:])).T.tocsr()
        barcodes = matrix["barcodes"].asstr()[:]
        features = matrix["features"]
        var = pd.DataFrame({name: features[name].asstr()[:] for name in ("id", "name", "feature_type", "genome") if name in features})
    # Keep the index distinct from the original symbol column after making it unique.
    var.index = pd.Index(var["name"].astype(str), name="gene")
    var["gene_symbol"] = var["name"]
    result = ad.AnnData(values, obs=pd.DataFrame(index=pd.Index(barcodes, name="cell_label")), var=var)
    if "feature_type" in result.var:
        result = result[:, result.var["feature_type"].eq("Gene Expression")].copy()
    result.var_names_make_unique()
    return result


def read_10x_mtx(matrix_path):
    import anndata as ad
    import pandas as pd
    from scipy.io import mmread

    matrix_path = Path(matrix_path)
    prefix = matrix_path.name.removesuffix("matrix.mtx.gz").removesuffix("matrix.mtx")
    def companion(suffix):
        matches = [p for p in matrix_path.parent.glob(prefix + suffix + "*") if p.name.endswith((".tsv", ".tsv.gz"))]
        if len(matches) != 1:
            raise ValueError(f"预期一个 {suffix} 文件：{matrix_path}")
        return matches[0]
    features = pd.read_csv(companion("features.tsv"), sep="\t", header=None, dtype=str)
    barcodes = pd.read_csv(companion("barcodes.tsv"), sep="\t", header=None, dtype=str)[0]
    var = pd.DataFrame({"gene_id": features[0].to_numpy(), "gene_symbol": features[1].to_numpy()}, index=features[1].to_numpy())
    if features.shape[1] > 2:
        var["feature_type"] = features[2].to_numpy()
    stream = gzip.open(matrix_path, "rb") if matrix_path.suffix == ".gz" else matrix_path.open("rb")
    with stream:
        values = mmread(stream).T.tocsr()
    result = ad.AnnData(values, obs=pd.DataFrame(index=pd.Index(barcodes, name="cell_label")), var=var)
    if "feature_type" in result.var:
        result = result[:, result.var["feature_type"].eq("Gene Expression")].copy()
    result.var_names_make_unique()
    return result


def _read_table(path):
    import pandas as pd

    path = Path(path)
    if path.name.endswith(".parquet.gz"):
        with gzip.open(path, "rb") as stream:
            return pd.read_parquet(stream)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    columns = pd.read_csv(path, nrows=0).columns
    return pd.read_csv(path, dtype={k: str for k in ["barcode", "cell_id"] if k in columns})


def attach_positions(adata, path):
    import pandas as pd

    path = Path(path)
    table = _read_table(path)
    if "barcode" not in table:
        # Space Ranger legacy tissue_positions_list.csv has no header.
        table = pd.read_csv(path, header=None, names=["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"], dtype={"barcode": str})
    table["barcode"] = table["barcode"].astype(str)
    table = table.set_index("barcode")
    if not table.index.is_unique or not adata.obs_names.isin(table.index).all():
        raise ValueError(f"spot 坐标与表达条码不匹配：{path}")
    table = table.reindex(adata.obs_names)
    if not {"pxl_col_in_fullres", "pxl_row_in_fullres"}.issubset(table):
        raise ValueError(f"缺少 full-resolution pixel XY：{path}")
    adata.obs = adata.obs.join(table)
    adata.obsm["spatial"] = table[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(dtype=float)
    adata.uns["coordinate_units"] = "full-resolution image pixels"
    return adata


def export_spatial_downloads(root, *, z_um=None):
    """Read downloaded 10x H5/MTX and real coordinates; never invent Z or cell types."""
    import pandas as pd

    root = Path(root)
    outputs, skipped = [], []
    candidates = list((root / "official").rglob("*.h5")) + list((root / "extracted").rglob("*.h5"))
    candidates += list((root / "official").rglob("*matrix.mtx.gz"))
    for matrix in sorted(candidates):
        if not any(t in matrix.name for t in ["feature_bc_matrix", "cell_feature_matrix", "matrix.mtx"]):
            continue
        # GEO isolates samples into GSM folders; HTAN matrices use sample-prefixed names.
        sample = next((p for p in matrix.parts if re.fullmatch(r"GSM\d+", p)), None)
        sample = sample or re.sub(r"[-_]?(?:filtered_)?(?:feature_bc_matrix|matrix\.mtx).*", "", matrix.name)
        if not sample:
            sample = matrix.parent.name
        search_roots = [root / "official" / sample, root / "extracted" / sample] if sample.startswith("GSM") else [matrix.parent]
        positions, cells = [], []
        for directory in search_roots:
            positions.extend(p for p in directory.rglob("*tissue_positions*") if p.is_file() and not p.name.endswith(".download.json"))
            cells.extend(p for p in directory.rglob("*cells.*") if p.name.endswith((".csv", ".csv.gz", ".parquet")))
        if not sample.startswith("GSM"):
            positions = [p for p in positions if p.name.startswith(sample + "-") or p.name.startswith(sample + "_")]
        if len(positions) == 1:
            adata = read_10x_mtx(matrix) if ".mtx" in matrix.name else read_10x_h5(matrix)
            attach_positions(adata, positions[0])
            adata.uns["observation_unit"] = "spot_or_bin"
            for directory in search_roots:
                for metadata_path in directory.rglob("*Metadata.parquet*"):
                    if metadata_path.name.endswith(".download.json"):
                        continue
                    metadata = _read_table(metadata_path)
                    if "barcode" in metadata:
                        metadata = metadata.assign(barcode=metadata["barcode"].astype(str)).set_index("barcode")
                        if not metadata.index.is_unique or not adata.obs_names.isin(metadata.index).all():
                            raise ValueError(f"元数据条码不匹配：{metadata_path}")
                        for column in metadata:
                            if column not in adata.obs:
                                adata.obs[column] = metadata[column].reindex(adata.obs_names)
        elif cells and "cell_feature_matrix" in matrix.name:
            # New Xenium bundles can provide the same cells table in CSV and Parquet.
            parents = {p.parent.resolve() for p in cells}
            if len(parents) != 1:
                raise ValueError(f"发现多个 Xenium cells 目录，拒绝模糊连接：{sample}")
            cell_table = sorted(cells, key=lambda p: (p.suffix != ".parquet", len(p.name)))[0]
            adata = read_10x_h5(matrix)
            metadata = _read_table(cell_table)
            if "cell_id" not in metadata:
                raise KeyError(f"Xenium cell metadata 缺少 cell_id：{cells[0]}")
            metadata["cell_id"] = metadata["cell_id"].astype(str)
            metadata = metadata.set_index("cell_id")
            if not metadata.index.is_unique or not adata.obs_names.isin(metadata.index).all():
                raise ValueError("Xenium cells 与矩阵 cell_id 不匹配。")
            adata.obs = metadata.reindex(adata.obs_names)
            adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
            adata.uns["coordinate_units"] = "micrometers"
            adata.uns["observation_unit"] = "segmented_cell"
            for labels_path in sorted((root / "official" / sample).glob("*supervised*.csv.gz")):
                labels = pd.read_csv(labels_path, index_col=0, dtype=str)
                # Early Xenium CSVs mix integer IDs with scientific notation (1e+05).
                # Only normalize numbers when the matrix itself uses canonical integer IDs.
                if adata.obs_names.str.fullmatch(r"(?:0|[1-9][0-9]*)").all():
                    try:
                        numbers = [Decimal(value) for value in labels.index]
                        if all(n.is_finite() and n == n.to_integral_value() for n in numbers):
                            labels.index = pd.Index([str(int(n)) for n in numbers])
                    except InvalidOperation:
                        pass
                if not labels.index.is_unique:
                    raise ValueError(f"Xenium 注释含重复 cell_id：{labels_path}")
                extra = int((~labels.index.isin(adata.obs_names)).sum())
                if extra:
                    # Numeric overlap alone cannot establish identity across segmentations.
                    reason = f"注释含 {extra} 个矩阵之外的 cell_id，无可靠映射，未合并此注释；表达和坐标保留。"
                    skipped.append({"annotation": str(labels_path.relative_to(root)), "reason": reason})
                    print(f"跳过注释：{labels_path.name}；{reason}")
                    continue
                for column in labels:
                    key = "annotation_" + str(column)
                    if key in adata.obs:
                        key = labels_path.name.removesuffix(".csv.gz") + "_" + str(column)
                    # A valid annotation subset leaves other cells unlabeled, not deleted.
                    adata.obs[key] = labels[column].reindex(adata.obs_names)
        else:
            skipped.append({"matrix": str(matrix.relative_to(root)), "reason": "没有唯一匹配的真实空间坐标；可能是非空间参考或尚未解压，未生成伪坐标。"})
            continue
        adata.obs["sample_id"] = sample
        if z_um and sample in z_um:
            adata.obs["z_coord_um"] = float(z_um[sample])
        target = root / "h5ad" / (sample + ".h5ad")
        outputs.append(_write(adata, target, {"matrix": str(matrix.relative_to(root)), "size": matrix.stat().st_size,
                                             "z_um": (z_um or {}).get(sample)}))
    from download_utils import save_json
    save_json(root / "metadata" / "h5ad_skipped.json", skipped)
    if not outputs:
        raise ValueError("未找到可转换的带真实空间坐标矩阵，见 metadata/h5ad_skipped.json。")
    return outputs


def export_cosmx(root):
    """Use R only to read the native Seurat object; export counts section by section."""
    import anndata as ad
    import pandas as pd
    from scipy.io import mmread

    root = Path(root)
    source = list((root / "extracted").rglob("cosmx.rds"))
    if len(source) != 1:
        raise ValueError("请先下载并解压 preprocessed_objects.zip，预期一个 cosmx.rds。")
    if not shutil.which("Rscript"):
        raise RuntimeError("可选 CosMx 导出需要 Rscript、R 包 SeuratObject 和 Matrix；官方 RDS 已保留，可先完成下载。")
    staging = root / "conversion"
    staging.mkdir(parents=True, exist_ok=True)
    script = staging / "export_cosmx.R"
    script.write_text('''args <- commandArgs(trailingOnly=TRUE)
suppressPackageStartupMessages(library(SeuratObject))
suppressPackageStartupMessages(library(Matrix))
obj <- readRDS(args[1])
meta <- obj@meta.data
required <- c("section", "celltypes", "x_3D_mn", "y_3D_mn", "z_3D_mn")
if (!all(required %in% colnames(meta))) stop("CosMx metadata lacks required spatial fields")
counts <- GetAssayData(obj, assay="Spatial", layer="counts")
for (section in sort(unique(as.character(meta$section)))) {
  if (!grepl("^section_[0-9]+$", section)) stop("Unexpected section identifier")
  dest <- file.path(args[2], section)
  dir.create(dest, recursive=TRUE, showWarnings=FALSE)
  ids <- rownames(meta)[as.character(meta$section)==section]
  writeMM(counts[, ids, drop=FALSE], file.path(dest, "counts.mtx"))
  write.table(meta[ids,,drop=FALSE], file.path(dest, "obs.tsv"), sep="\\t", quote=TRUE, col.names=NA)
  write.table(data.frame(gene_symbol=rownames(counts)), file.path(dest, "var.tsv"), sep="\\t", quote=TRUE, row.names=FALSE)
}
''')
    try:
        subprocess.run(["Rscript", "--vanilla", str(script), str(source[0]), str(staging)], check=True)
    except OSError as error:
        raise RuntimeError("Rscript 无法执行；请安装与当前系统架构匹配的 R，再运行可选 CosMx 转换。") from error
    outputs = []
    for directory in sorted(staging.glob("section_*")):
        obs = pd.read_csv(directory / "obs.tsv", sep="\t", index_col=0)
        var = pd.read_csv(directory / "var.tsv", sep="\t").set_index("gene_symbol", drop=False)
        piece = ad.AnnData(mmread(directory / "counts.mtx").T.tocsr(), obs=obs, var=var)
        piece.obs_names = piece.obs_names.astype(str)
        piece.var_names_make_unique()
        piece.obsm["spatial"] = obs[["x_3D_mn", "y_3D_mn"]].to_numpy(dtype=float)
        piece.obs["z_coord_um"] = obs["z_3D_mn"].to_numpy(dtype=float)
        piece.obs["cell_type"] = obs["celltypes"].astype("category")
        piece.uns["coordinate_units"] = "micrometers"
        piece.uns["observation_unit"] = "segmented_cell"
        outputs.append(_write(piece, root / "h5ad" / (directory.name + ".h5ad"),
                              {"matrix": source[0].name, "size": source[0].stat().st_size, "section": directory.name}))
    return outputs
