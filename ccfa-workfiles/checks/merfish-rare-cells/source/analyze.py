"""Audit the saved MERFISH run without changing its inputs or checkpoint files."""
from pathlib import Path
import io
import json
import os
import re
import zipfile

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[4]
DEST = ROOT / "output" / "analysis"
DEST.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / "build" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def as_markdown(frame):
    # Keep the report independent of optional Markdown-table packages.
    rows = [[str(v) for v in row] for row in frame.itertuples(index=False, name=None)]
    return "\n".join([
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ])


def main():
    params = json.loads((ROOT / "output/run_parameters.json").read_text(encoding="utf-8-sig"))
    archive = next((ROOT / "data/merfish_mouse_hypothalamus").glob("merfish*.zip"))
    with zipfile.ZipFile(archive) as zf:
        names = sorted(
            [n for n in zf.namelist() if re.search(r"merfish_\d+\.h5ad$", n)],
            key=lambda n: int(re.search(r"merfish_(\d+)", n).group(1)),
        )
        refs = [ad.read_h5ad(io.BytesIO(zf.read(n))) for n in names]
    result = ad.read_h5ad(ROOT / "output/deepspatial_3d.h5ad")
    assert len(refs) == 5 and all(r.var_names.equals(result.var_names) for r in refs)
    assert not result.obs.cell_class.isna().any()
    ref_obs = pd.concat([r.obs for r in refs], ignore_index=True)
    classes = list(result.obs.cell_class.cat.categories)
    sizes = np.array([r.n_obs for r in refs])
    ref_counts = np.array([r.obs.cell_class.value_counts().reindex(classes, fill_value=0).values for r in refs])
    labels = result.obs.cell_class.astype(str).to_numpy()
    codes = pd.Categorical(labels, categories=classes).codes
    assert np.all(codes >= 0)
    z = result.obs.z_coord.to_numpy()
    gap_width = float(params["slice_gap"])
    gap_edges = np.arange(len(refs), dtype=float) * gap_width
    assert np.isfinite(z).all() and z.min() >= -1e-5 and z.max() <= gap_edges[-1] + 1e-5
    gaps = np.clip(np.searchsorted(gap_edges, z, side="right") - 1, 0, 3)
    obs_gap = np.zeros((4, len(classes)), dtype=int)
    np.add.at(obs_gap, (gaps, codes), 1)
    z_edges = np.arange(0, gap_edges[-1] + 1, 10.0)
    obs_bins = np.zeros((len(z_edges) - 1, len(classes)), dtype=int)
    bins = np.clip(np.searchsorted(z_edges, z, side="right") - 1, 0, len(z_edges) - 2)
    np.add.at(obs_bins, (bins, codes), 1)
    expected_gap = np.zeros_like(obs_gap, dtype=float)
    expected_bins = np.zeros_like(obs_bins, dtype=float)
    source_weights = np.zeros(len(refs))
    for i in range(4):
        n0, n1 = sizes[i:i + 2]
        p0, p1 = ref_counts[i] / n0, ref_counts[i + 1] / n1
        dn, dp = n1 - n0, p1 - p0
        n_generated = obs_gap[i].sum()
        # Null reference: follow the actual t/direction/source sampling and preserve source labels.
        def integrate(lo, hi):
            return 2 / (n0 + n1) * (
                n0 * p0 * (hi - lo)
                + (n0 * dp + dn * p0) * (hi ** 2 - lo ** 2) / 2
                + dn * dp * (hi ** 3 - lo ** 3) / 3
            )
        expected_gap[i] = n_generated * integrate(0, 1)
        mean_t = (n0 + 2 * n1) / (3 * (n0 + n1))
        source_weights[i] += n_generated * (1 - mean_t) / n0
        source_weights[i + 1] += n_generated * mean_t / n1
        for j, (lo, hi) in enumerate(zip(z_edges[:-1], z_edges[1:])):
            a = max(lo, gap_edges[i])
            b = min(hi, gap_edges[i + 1])
            if b > a:
                expected_bins[j] += n_generated * integrate((a - gap_edges[i]) / gap_width, (b - gap_edges[i]) / gap_width)
    assert np.allclose(expected_gap.sum(axis=1), obs_gap.sum(axis=1))
    assert np.allclose(expected_bins.sum(axis=0), expected_gap.sum(axis=0))

    stats = pd.DataFrame({
        "cell_class": classes, "input_count": ref_counts.sum(axis=0),
        "output_count": obs_gap.sum(axis=0), "sampling_reference_count": expected_gap.sum(axis=0),
    }).set_index("cell_class")
    stats["input_pct"] = stats.input_count / sizes.sum() * 100
    stats["output_pct"] = stats.output_count / result.n_obs * 100
    stats["output_to_input_share_ratio"] = stats.output_pct / stats.input_pct
    stats["output_to_sampling_reference_ratio"] = stats.output_count / stats.sampling_reference_count
    stats["rare_input_below_1pct"] = stats.input_pct < 1
    stats["empty_z10_bins"] = (obs_bins == 0).sum(axis=0)
    stats["empty_50um_gaps"] = (obs_gap == 0).sum(axis=0)
    stats = stats.sort_values("input_count")
    rare = stats.index[stats.rare_input_below_1pct].tolist()
    all_xy = np.vstack([r.obsm["spatial"] for r in refs])
    ref_labels = ref_obs.cell_class.astype(str).to_numpy()
    ref_z = np.concatenate([np.full(r.n_obs, i * gap_width) for i, r in enumerate(refs)])
    ref_ids = np.concatenate([r.obs_names.to_numpy() for r in refs])
    spatial_distances = {}
    anchor_rows = []
    input_groups = ref_obs[["slice_id", "cell_class", "Region"]].astype(str).groupby(["slice_id", "cell_class", "Region"]).size().rename("input_count")
    output_groups = result.obs[["slice_id", "cell_class", "Region"]].astype(str).groupby(["slice_id", "cell_class", "Region"]).size().rename("output_inherited_origin_count")
    region_groups = input_groups.to_frame().join(output_groups).fillna(0).astype(int)
    missing_origin_groups = region_groups.loc[(region_groups.output_inherited_origin_count == 0) & region_groups.index.get_level_values("cell_class").isin(rare)]
    for c in classes:
        rmask, omask = ref_labels == c, labels == c
        distance, neighbor = cKDTree(result.obsm["spatial"][omask]).query(all_xy[rmask])
        spatial_distances[c] = distance
        stats.loc[c, "median_input_to_output_xy_nn_um"] = np.median(distance)
        stats.loc[c, "max_input_to_output_xy_nn_um"] = distance.max()
        stats.loc[c, "input_anchors_without_same_class_output_within_50um_xy"] = int((distance > 50).sum())
        stats.loc[c, "input_anchors_without_same_class_output_within_100um_xy"] = int((distance > 100).sum())
        if c in rare:
            source_rows = np.flatnonzero(rmask)
            for k, ref_row in enumerate(source_rows):
                origin = ref_obs.iloc[ref_row]
                group_count = region_groups.loc[(str(origin.slice_id), c, str(origin.Region)), "output_inherited_origin_count"]
                anchor_rows.append({"cell_class": c, "input_cell_id": str(ref_ids[ref_row]),
                    "input_z_um": ref_z[ref_row], "input_x_um": all_xy[ref_row, 0], "input_y_um": all_xy[ref_row, 1],
                    "input_Region": str(origin.Region), "nearest_same_class_output_xy_um": distance[k],
                    "nearest_same_class_output_z_um": z[omask][neighbor[k]],
                    "output_in_same_inherited_slice_class_region_group": group_count})
    anchors = pd.DataFrame(anchor_rows)
    anchors.to_csv(DEST / "rare_cell_anchor_distances.csv", index=False, encoding="utf-8-sig")
    last_source_id = str(refs[-1].obs.slice_id.iloc[0])
    last_source_draws = int((result.obs.slice_id.astype(str) == last_source_id).sum())
    last_source_zero_probability = (1 - 1 / sizes[-1]) ** last_source_draws

    raw_ref = np.vstack([r.X.toarray() if sparse.issparse(r.X) else np.asarray(r.X) for r in refs])
    ref_log = np.log1p(raw_ref.astype(np.float64))
    out_log = result.X.astype(np.float64)
    out_log.data = np.log1p(out_log.data)
    weights = np.concatenate([np.full(r.n_obs, source_weights[i]) for i, r in enumerate(refs)])
    for c in classes:
        mask_r, mask_o = ref_labels == c, labels == c
        ref_mean = np.average(ref_log[mask_r], axis=0, weights=weights[mask_r])
        ref_var = np.average((ref_log[mask_r] - ref_mean) ** 2, axis=0, weights=weights[mask_r]).mean()
        out_type = out_log[mask_o]
        out_mean = np.asarray(out_type.mean(axis=0)).ravel()
        out_var = (np.asarray(out_type.power(2).mean(axis=0)).ravel() - out_mean ** 2).mean()
        stats.loc[c, "weighted_mean_log_profile_PCC"] = np.corrcoef(ref_mean, out_mean)[0, 1]
        stats.loc[c, "mean_log_gene_variance_ratio"] = out_var / ref_var
    assert int(stats.input_count.sum()) == int(sizes.sum())
    assert int(stats.output_count.sum()) == result.n_obs
    stats.to_csv(DEST / "rare_cell_counts.csv", encoding="utf-8-sig")
    by_z = []
    for j, lo in enumerate(z_edges[:-1]):
        for c in rare:
            k = classes.index(c)
            by_z.append({"z_start_um": lo, "z_end_um": z_edges[j + 1], "cell_class": c,
                         "output_count": obs_bins[j, k], "sampling_reference_count": expected_bins[j, k]})
    pd.DataFrame(by_z).to_csv(DEST / "rare_cells_by_z.csv", index=False, encoding="utf-8-sig")

    subtypes = []
    for subtype, count in ref_obs.neuron_class.dropna().astype(str).value_counts().items():
        mask_r = ref_obs.neuron_class.astype(object).to_numpy() == subtype
        mask_o = result.obs.neuron_class.astype(object).to_numpy() == subtype
        corr = np.corrcoef(ref_log[mask_r].mean(axis=0), np.asarray(out_log[mask_o].mean(axis=0)).ravel())[0, 1] if mask_o.any() else np.nan
        subtypes.append({"neuron_subtype": subtype, "input_count": count, "output_inherited_annotation_count": int(mask_o.sum()), "mean_log_profile_PCC": corr, "annotation_origin": "copied_from_source_cell_not_model_predicted"})
    subtype_table = pd.DataFrame(subtypes).sort_values("input_count")
    subtype_table.to_csv(DEST / "neuron_subtypes.csv", index=False, encoding="utf-8-sig")
    log = (ROOT / "output/train.log").read_text(encoding="utf-8")
    losses = {}
    for epoch, value in re.findall(r"Epoch\s+(\d+):[^\r\n]*loss_epoch=([\d.eE+-]+)", log):
        losses[int(epoch)] = float(value)
    loss_table = pd.DataFrame(sorted(losses.items()), columns=["epoch", "rounded_console_training_loss"])
    loss_table.to_csv(DEST / "training_loss.csv", index=False, encoding="utf-8-sig")

    font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(14, 9), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1.1, 1])
    ax = fig.add_subplot(grid[:, 0])
    y = np.arange(len(stats))
    for j, row in enumerate(stats.itertuples()):
        ax.plot([row.input_pct, row.output_pct], [j, j], color="#b5b9c4", lw=1.5)
    ax.scatter(stats.input_pct, y, color="#667085", s=36, label="Input slices", zorder=3)
    ax.scatter(stats.output_pct, y, color="#078c88", marker="D", s=31, label="Reconstruction", zorder=4)
    ax.set_yticks(y, stats.index)
    ax.invert_yaxis(); ax.set_xscale("log"); ax.set_xlim(.03, 60)
    ax.axvline(1, color="#d68c3d", linestyle=":", lw=1)
    ax.set_xlabel("Cell-class proportion (%) [log scale]")
    ax.set_title("A  All 15 classes remain present", loc="left", fontweight="bold")
    ax.legend(loc="lower left", frameon=False)
    ax.grid(axis="x", color="#e5e7eb", lw=.6)
    for tick, c in zip(ax.get_yticklabels(), stats.index):
        if c in rare: tick.set_fontweight("bold")
    ax = fig.add_subplot(grid[0, 1])
    ratio = stats.loc[rare, "output_to_sampling_reference_ratio"]
    ax.barh(np.arange(len(rare)), ratio, color="#078c88", height=.55)
    ax.axvline(1, color="#535b6c", linestyle="--", lw=1)
    ax.set_yticks(np.arange(len(rare)), rare); ax.invert_yaxis(); ax.set_xlim(0, 1.35)
    for i, value in enumerate(ratio): ax.text(value + .025, i, f"{value:.3f}", va="center", fontsize=10)
    ax.set_xlabel("Observed / source-sampling reference count")
    ax.set_title("B  Rare-class totals follow sampling counts", loc="left", fontweight="bold")
    ax = fig.add_subplot(grid[1, 1])
    ax.plot(loss_table.epoch, loss_table.rounded_console_training_loss, color="#536cba", lw=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Training loss (rounded log values)")
    ax.set_title("C  100 training epochs completed", loc="left", fontweight="bold")
    ax.grid(color="#e5e7eb", lw=.6)
    fig.savefig(DEST / "overview.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 4.2), layout="constrained")
    matrix = obs_bins[:, [classes.index(c) for c in rare]].T
    im = ax.imshow(np.log1p(matrix), cmap="YlGnBu", aspect="auto", vmin=0)
    ax.set_yticks(np.arange(len(rare)), rare)
    ax.set_xticks(np.arange(len(z_edges) - 1), [f"{int(lo)}" for lo in z_edges[:-1]])
    for i in range(len(rare)):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=10,
                    color="white" if np.log1p(matrix[i, j]) > np.log1p(matrix.max()) * .65 else "#182234")
    ax.set_xlabel("Z-bin start (um); each bin spans 10 um")
    ax.set_title("Rare-cell counts: all six classes occupy all 20 Z bins", loc="left", fontweight="bold")
    cb = fig.colorbar(im, ax=ax, shrink=.85, pad=.015)
    cb.set_label("log(1 + cell count)")
    fig.savefig(DEST / "rare_cells_z_counts.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(12, 13), sharex=True, sharey=True, layout="constrained")
    lo = np.minimum(all_xy.min(axis=0), result.obsm["spatial"].min(axis=0)) - 40
    hi = np.maximum(all_xy.max(axis=0), result.obsm["spatial"].max(axis=0)) + 40
    for ax, c in zip(axes.flat, rare):
        rxy, oxy = all_xy[ref_labels == c], result.obsm["spatial"][labels == c]
        ax.scatter(oxy[:, 0], oxy[:, 1], s=9, alpha=.55, color="#078c88", label="Generated")
        ax.scatter(rxy[:, 0], rxy[:, 1], s=38, marker="x", lw=1.1, color="#19223a", label="Input anchors", zorder=4)
        distance = spatial_distances[c]
        far = distance > 100
        ax.scatter(rxy[far, 0], rxy[far, 1], s=140, facecolors="none", edgecolors="#c43c39", lw=1.4, label="No same class within 100 um", zorder=5)
        if distance.max() > 200:
            j = int(np.argmax(distance))
            ax.annotate(f"Nearest same class: {distance[j]:.0f} um", rxy[j], xytext=(8, -18),
                        textcoords="offset points", fontsize=10, color="#b12626",
                        arrowprops={"arrowstyle": "->", "color": "#b12626"})
        ax.set_title(f"{c}: {len(rxy)} input / {len(oxy)} generated\n{int(far.sum())} input anchor(s) farther than 100 um", fontsize=11)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
        ax.set_xlabel("X (um)"); ax.set_ylabel("Y (um)")
    axes.flat[0].legend(loc="upper right", frameon=False, fontsize=9)
    fig.suptitle("Rare-class totals survive, but some input locations lack local coverage", fontsize=15, fontweight="bold")
    fig.savefig(DEST / "rare_cells_spatial.png", dpi=170)
    plt.close(fig)

    rare_display = pd.DataFrame({
        "细胞类型": rare,
        "输入数量": stats.loc[rare, "input_count"].astype(int).values,
        "生成数量": stats.loc[rare, "output_count"].astype(int).values,
        "输入占比": [f"{v:.4f}%" for v in stats.loc[rare, "input_pct"]],
        "生成占比": [f"{v:.4f}%" for v in stats.loc[rare, "output_pct"]],
        "采样参照数量": [f"{v:.1f}" for v in stats.loc[rare, "sampling_reference_count"]],
        "生成/参照": [f"{v:.3f}" for v in stats.loc[rare, "output_to_sampling_reference_ratio"]],
    })
    gap_display = pd.DataFrame(obs_gap[:, [classes.index(c) for c in rare]], columns=rare)
    gap_display.insert(0, "Z 区间 (um)", [f"{int(gap_edges[i])}–{int(gap_edges[i + 1])}" for i in range(4)])
    expression_display = pd.DataFrame({
        "细胞类型": rare,
        "平均 log1p 表达谱 PCC": [f"{v:.4f}" for v in stats.loc[rare, "weighted_mean_log_profile_PCC"]],
        "平均基因方差比": [f"{v:.3f}" for v in stats.loc[rare, "mean_log_gene_variance_ratio"]],
    })
    coverage_display = pd.DataFrame({
        "细胞类型": rare,
        "输入数量": stats.loc[rare, "input_count"].astype(int).values,
        "100 um 内无同类输出的输入点": stats.loc[rare, "input_anchors_without_same_class_output_within_100um_xy"].astype(int).values,
        "最远最近邻距离 (um)": [f"{v:.1f}" for v in stats.loc[rare, "max_input_to_output_xy_nn_um"]],
    })
    missing_groups_display = missing_origin_groups.reset_index().rename(columns={
        "slice_id": "来源切片标识", "cell_class": "细胞类型", "Region": "输入来源区域",
        "input_count": "输入数量", "output_inherited_origin_count": "该来源分组输出数量"})
    missing_classes = stats.index[(stats.input_count > 0) & (stats.output_count == 0)].tolist()
    missing_subtypes = subtype_table.loc[subtype_table.output_inherited_annotation_count == 0, "neuron_subtype"].tolist()
    checkpoint_names = ", ".join(p.name for p in sorted((ROOT / "output/checkpoints").glob("deepspatial-*.ckpt")))
    values = result.X.data
    report = f"""# MERFISH 重建结果：稀有细胞检查

分析对象：`output/deepspatial_3d.h5ad`。输入来源：`{archive.relative_to(ROOT).as_posix()}` 内的 5 个切片。

## 结论

本次重建保留了 `cell_class` 大类和稀有大类的总量，但发现部分稀有输入细胞的位置缺少同类生成细胞。输入共 {sizes.sum():,} 个细胞，输出 {result.n_obs:,} 个生成细胞，均包含 {len(classes)} 个细胞大类。完全缺失的大类：{missing_classes}。

这里将输入占比小于 1% 的类别作为描述性的“稀有类”分组，共 {len(rare)} 类；此阈值不是统一的生物学定义。数量保留不等于真实缺失组织中细胞的逐个恢复，输入标签本身参与了条件生成。

最需要关注的两个局部例子均在末端切片 Z=200：`OD Mature 3` 输入细胞 `-3688x-3493`（MPA，XY≈-828.6,-452.6），距最近同类输出的 XY 距离约 549.4 um；`OD Immature 2` 输入细胞 `-3464x-3103`（BST，XY≈-605.2,-62.2），对应距离约 377.2 um。这里搜索了整块输出体、忽略 Z 差异，是较宽松的覆盖检查；这些点在三维距离下也不会更近。

## 数量与比例

{as_markdown(rare_display)}

输出数量增多由体积采样密度和 `thickness=10` 等配置共同决定，不代表获得了更多独立生物学观测。输入内部切片参与前后两个间隔，端点切片只参与一个，因此不宜仅用全局输入占比判断低估。

“采样参照”依据仓库 `_generate_and_prune_optimized` 的随机深度、方向和均匀来源细胞采样机制推算，并假设来源标签保持不变。对某间隔，N0/N1 为两侧细胞数，E[t]=(N0+2*N1)/(3*(N0+N1))，类别参照数量为 Ngen*((1-E[t])*p0+E[t]*p1)。它是采样诊断基线，不是缺失组织真值。

## 沿 Z 的局部占据

{as_markdown(gap_display)}

按 10 um 划分 20 个 Z 区间，六个稀有大类在每个区间均非零。该结论仅针对沿 Z 的分箱，不能证明每个 XY 微环境均无缺失。最稀有类别单个区间只有少量细胞，数量波动需结合参照和抽样规模解释。

## 输入稀有细胞附近的局部覆盖不足

{as_markdown(coverage_display)}

对每个输入细胞，在全部同类生成点中搜索 XY 最近邻。100 um 是用于筛查明显覆盖空缺的描述性半径，不是细胞一一对应或生物学消失的判定标准；稀有细胞本身更稀疏，因此不能直接把此表与丰富类型的覆盖率当作公平的准确率比较。完整逐点结果保存在 `rare_cell_anchor_distances.csv`。

沿来源切片、来源区域和细胞大类细分，共有 {len(missing_origin_groups)} 个原先非空的稀有分组在输出中为零：

{as_markdown(missing_groups_display)}

这些分组中的 Region 和 slice_id 是来源注释，仅用于追溯生成来源，不作为新细胞真实区域的预测指标。当前结果支持“局部输入稀有细胞覆盖不足”，尚不能证明真实缺失组织中的生物学细胞丢失。

代码使用有放回的均匀来源细胞采样，不保证每个输入细胞都被选择；返回的 AnnData 也只包含生成点，没有自动拼回原始观测细胞。本次末端切片来源抽样共 {last_source_draws:,} 次，对 {sizes[-1]:,} 个输入细胞平均每个 {last_source_draws / sizes[-1]:.2f} 次。均匀独立采样下，一个给定细胞一次未被抽中的概率约 {last_source_zero_probability * 100:.1f}%。这提供了局部漏覆盖的一种合理机制，但由于没有保存 parent_id 和连续类别分数，仍无法把采样遗漏、类别变化或位置变化逐一分离。

## 表达一致性与多样性

{as_markdown(expression_display)}

表达检查先对每细胞表达做 log1p，再比较各类别的平均基因表达谱。输入参照按相同采样机制对各切片加权。方差比为输出平均逐基因方差/输入加权平均逐基因方差。OD Immature 2、Pericytes、OD Mature 3 的方差略低，提示可能有一定平滑；这不是稀有类别消失的证据，也不是独立准确率。

这 5 个切片全部参与训练；上述相关性和方差都是内部一致性诊断，没有独立留出真值，也没有生物学重复或多随机种子，因此不将其作为恢复准确率或显著性检验。

## 细分神经元亚型的解释

输入 `neuron_class` 有 {len(subtype_table)} 个非空亚型，输出继承字段中也有 {len(subtype_table) - len(missing_subtypes)} 个，缺失项：{missing_subtypes}。但模型训练的 `label_key` 是 `cell_class`，并未独立预测这 {len(subtype_table)} 个细分标签。`neuron_class`、`Region`、`annotation` 等由来源细胞复制，不能直接作为细分类型恢复成功的证据。

细分亚型的数量和平均表达谱检查保存在 `neuron_subtypes.csv`；其中标签来源已明确标记。输入仅 2–5 个细胞的亚型尤其缺少独立验证。

## 训练与数据质量

- 日志完成 {len(losses)} 个 epoch（编号 {min(losses)}–{max(losses)}）；日志四舍五入后的训练 loss 约从 {losses[min(losses)]:.2f} 降至 {losses[max(losses)]:.2f}。它仍在下降，不能据此认定已充分收敛或验证集表现良好。
- 按训练 loss 筛选保存的 checkpoint：`{checkpoint_names}`；运行参数指定输出使用 `checkpoints/last.ckpt`。
- 输出表达矩阵有限值检查：{bool(np.isfinite(values).all())}；负数元素：{int((values < 0).sum())}；全零细胞：{int((np.asarray(result.X.sum(axis=1)).ravel() == 0).sum())}。
- 输出标签缺失值：{int(result.obs.cell_class.isna().sum())}；三维坐标有限值检查：{bool(np.isfinite(result.obsm['spatial_3d']).all())}；155 个基因的名称及顺序与输入一致。

## 下一步验证

若研究问题是“模型会不会漏掉真实稀有细胞”，下一步应采用留出中间切片或 DeepSTARmap 真实 3D 数据的未观测区间，独立评价稀有类数量、空间匹配和分子身份，并检查细分亚型。若下游图谱需要保留已观测的真实稀有细胞，应保留原始观测数据并明确区分 observed/generated；这不能替代模型自身的独立评估，也不能将拼回观测点后的覆盖提升当作模型性能。当前结果支持的表述是：本次条件生成运行保留了输入已有的 15 个大类和其中 6 个稀有大类的数量，但存在部分局部输入位置未获同类覆盖的问题。

## 文件

- `rare_cell_counts.csv`：15 类的数量、占比、采样参照、表达一致性和方差比。
- `rare_cells_by_z.csv`：稀有类 10 um 分箱数量与参照。
- `rare_cell_anchor_distances.csv`：680 个输入稀有细胞到同类生成点的最近 XY 距离及来源分组统计。
- `neuron_subtypes.csv`：继承的细分亚型统计（非独立预测标签）。
- `training_loss.csv`：从控制台日志提取的每 epoch 最后一个 loss_epoch；数值有日志舍入。
- `overview.png`、`rare_cells_z_counts.png`、`rare_cells_spatial.png`：数量、Z 占据和 XY 投影图。
- 可复现分析源码：`ccfa-workfiles/checks/merfish-rare-cells/source/analyze.py`。
"""
    (DEST / "report.md").write_text(report, encoding="utf-8")
    print("Analysis written to", DEST)
    print(rare_display.to_string(index=False))


if __name__ == "__main__":
    main()
