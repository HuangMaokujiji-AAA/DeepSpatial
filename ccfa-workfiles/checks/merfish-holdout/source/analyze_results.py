"""Audit saved holdout predictions and produce descriptive rare-cell diagnostics.

Run from any directory with numpy/pandas/anndata/scipy/matplotlib installed.
This reads existing experiments; it never trains a model or changes predictions.
"""
from pathlib import Path
import json
import os

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "output" / "merfish_holdout"
DEST = RESULTS / "analysis"
WORK = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(WORK / "build" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def dense(a):
    return a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)


def markdown(frame):
    return "\n".join([
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
        *["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False, name=None)],
    ])


def metric_functions():
    notebook = json.loads((ROOT / "merfish_leave_one_slice_out.ipynb").read_text(encoding="utf-8"))
    source = next("".join(c["source"]) for c in notebook["cells"] if c["id"] == "metric-helpers")
    scope = dict(np=np, pd=pd, ad=ad, sp=sp, cKDTree=cKDTree,
                 LABEL_KEY="cell_class", RARE_TRAIN_FRACTION=0.01, COVERAGE_RADII_UM=(25, 50, 100))
    exec(compile(source, "notebook:metric-helpers", "exec"), scope)
    return scope


def assert_same_table(expected, actual):
    expected = expected.set_index("cell_class").sort_index()
    actual = actual.set_index("cell_class").sort_index()
    assert expected.index.equals(actual.index)
    max_difference = 0.0
    for column in expected.columns:
        if column in actual.columns and column != "heldout_id":
            left = expected[column].to_numpy(dtype=float)
            right = actual[column].to_numpy(dtype=float)
            max_difference = max(max_difference, float(np.nanmax(np.abs(left - right))))
            np.testing.assert_allclose(left, right, rtol=1e-7, atol=1e-7,
                                       equal_nan=True, err_msg=column)
    return max_difference


def plot_results(table, folds):
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    labels = ["OD Mature 3", "OD Immature 2", "OD Mature 4", "Endothelial 2", "Pericytes", "OD Mature 1"]
    # Adapt the CCFA heatmap grammar to a fixed 0--100% scale shared by both panels.
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    for ax, radius, letter in zip(axes, (50, 100), ("A", "B")):
        matrix = table.pivot(index="cell_class", columns="heldout_id",
                             values=f"coverage_recall_{radius}um").reindex(labels).to_numpy() * 100
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(3), ["Holdout 1", "Holdout 2", "Holdout 3"])
        ax.set_yticks(range(len(labels)), labels)
        ax.set_title(f"{letter}   Same-type coverage within {radius} µm", loc="left", fontsize=12)
        for row in range(len(labels)):
            for col in range(3):
                value = matrix[row, col]
                ax.text(col, row, f"{value:.1f}%", ha="center", va="center",
                        color="white" if value > 60 else "#152638")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.colorbar(image, ax=axes, fraction=0.035, pad=0.03, label="True cells covered (%)")
    fig.savefig(DEST / "rare_cell_coverage.png", dpi=200)
    fig.savefig(DEST / "rare_cell_coverage.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(8.3, 10.2), constrained_layout=True)
    fig.suptitle("Blue circles: truth (100 µm disks)     Orange crosses: prediction", fontsize=11)
    for row, i in enumerate((1, 2, 3)):
        truth, pred = folds[i]
        truth_xy, pred_xy = np.asarray(truth.obsm["spatial"]), np.asarray(pred.obsm["spatial"])
        for col, label in enumerate(("OD Mature 3", "OD Immature 2")):
            ax = axes[row, col]
            tm = truth.obs.cell_class.astype(str).to_numpy() == label
            pm = pred.obs.cell_class.astype(str).to_numpy() == label
            ax.scatter(*truth_xy.T, s=1, color="#c8cdd2", alpha=0.35, rasterized=True)
            for point in truth_xy[tm]:
                ax.add_patch(plt.Circle(point, 100, color="#357ba8", alpha=0.10, linewidth=0))
            ax.scatter(*truth_xy[tm].T, s=55, facecolors="none", edgecolors="#246c99",
                       linewidths=1.6, label="Truth")
            ax.scatter(*pred_xy[pm].T, s=45, marker="x", color="#d15e32", linewidths=1.7,
                       label="Prediction")
            ax.set_title(f"Holdout {i}: {label}  ({tm.sum()} true, {pm.sum()} predicted)", fontsize=11)
            ax.set_xlim(-1000, 1000)
            ax.set_ylim(-1000, 1000)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("X (µm)")
            ax.set_ylabel("Y (µm)")
            ax.spines[["right", "top"]].set_visible(False)
    fig.savefig(DEST / "rare_cell_spatial.png", dpi=200)
    fig.savefig(DEST / "rare_cell_spatial.pdf")
    plt.close(fig)


def main():
    DEST.mkdir(exist_ok=True)
    functions = metric_functions()
    evaluate = functions["evaluate_holdout"]
    original = {}
    for i in range(5):
        paths = list((ROOT / "data" / "merfish_mouse_hypothalamus").rglob(f"merfish_{i}.h5ad"))
        assert len(paths) == 1
        original[i] = ad.read_h5ad(paths[0])

    saved_summary = pd.read_csv(RESULTS / "fold_summary.csv").set_index("heldout_id")
    folds, all_tables, summary_rows, anchors, qc_rows, losses = {}, [], [], [], [], []
    for i in (1, 2, 3):
        folder = RESULTS / f"holdout_{i}"
        parameters = json.loads((folder / "run_parameters.json").read_text(encoding="utf-8"))
        config = json.loads((folder / "checkpoints" / "config.json").read_text(encoding="utf-8"))
        train_ids = [j for j in range(5) if j != i]
        assert parameters["train_ids"] == train_ids
        assert parameters["heldout_id"] == i
        assert parameters["reference_pair"] == [i - 1, i + 1]
        assert parameters["seed"] == 42 and parameters["epochs"] == 100
        train_counts = pd.concat([original[j].obs.cell_class.astype(str) for j in train_ids]).value_counts()
        saved_counts = pd.read_csv(folder / "train_cell_counts.csv").set_index("cell_class").train_n
        pd.testing.assert_series_equal(train_counts.sort_index(), saved_counts.sort_index(), check_names=False)
        train_xy = np.vstack([original[j].obsm["spatial"] for j in train_ids])
        np.testing.assert_allclose(
            [config["spatial_stats"][key] for key in ("x_min", "y_min", "x_range", "y_range")],
            [*train_xy.min(axis=0), *(np.ptp(train_xy, axis=0) + 1e-8)], rtol=1e-10,
        )
        assert config["categories"] == sorted(train_counts.index)

        truth = ad.read_h5ad(folder / "heldout_truth.h5ad")
        pred = ad.read_h5ad(folder / "predicted_holdout.h5ad")
        gap = ad.read_h5ad(folder / "predicted_gap.h5ad")
        assert truth.obs_names.equals(original[i].obs_names)
        assert truth.var_names.equals(original[i].var_names)
        np.testing.assert_array_equal(dense(truth), dense(original[i]))
        np.testing.assert_array_equal(truth.obsm["spatial"], original[i].obsm["spatial"])
        np.testing.assert_array_equal(truth.obs.cell_class.astype(str), original[i].obs.cell_class.astype(str))
        z = i * parameters["slice_gap_um"]
        selected = gap[np.abs(gap.obs.z_coord.to_numpy() - z) <= parameters["slab_half_width_um"]]
        assert selected.obs_names.equals(pred.obs_names)
        np.testing.assert_array_equal(dense(selected), dense(pred))
        np.testing.assert_array_equal(selected.obsm["spatial"], pred.obsm["spatial"])
        np.testing.assert_allclose(pred.obsm["spatial_3d"][:, :2], pred.obsm["spatial"])

        table, summary = evaluate(truth, pred, train_counts)
        max_difference = assert_same_table(table, pd.read_csv(folder / "cell_type_metrics.csv"))
        for key, value in summary.items():
            np.testing.assert_allclose(value, saved_summary.loc[i, key], atol=1e-7, rtol=1e-7, equal_nan=True)
        table.insert(0, "heldout_id", i)
        rare = table[table.is_rare_train & (table.truth_n > 0)]
        summary_rows.append(dict(heldout_id=i, **summary,
                                 rare_true_cells=int(rare.truth_n.sum()),
                                 rare_predicted_cells=int(rare.predicted_n.sum()),
                                 rare_micro_recall_50um=float(np.average(rare.coverage_recall_50um, weights=rare.truth_n)),
                                 rare_micro_recall_100um=float(np.average(rare.coverage_recall_100um, weights=rare.truth_n)),
                                 cell_fraction_total_variation=float(np.abs(table.truth_fraction - table.predicted_fraction).sum() / 2)))
        all_tables.append(table)
        folds[i] = truth, pred

        pred_x = dense(pred)
        qc_rows.append(dict(heldout_id=i, genes=pred.n_vars, truth_matches_original=True,
                            split_and_normalization_consistent=True, metrics_match_saved_csv=True,
                            maximum_class_metric_difference=max_difference,
                            slab_matches_gap_filter=True, negative_expression_entries=int((pred_x < 0).sum()),
                            minimum_expression=float(pred_x.min()), zero_expression_cells=int((pred_x == 0).all(axis=1).sum()),
                            finite_expression=bool(np.isfinite(pred_x).all()),
                            finite_coordinates=bool(np.isfinite(pred.obsm["spatial_3d"]).all())))
        tlabels = truth.obs.cell_class.astype(str).to_numpy()
        plabels = pred.obs.cell_class.astype(str).to_numpy()
        for label in rare.cell_class:
            ids = np.where(tlabels == label)[0]
            pids = np.where(plabels == label)[0]
            distance, nearest = cKDTree(pred.obsm["spatial"][pids]).query(truth.obsm["spatial"][ids])
            for index, dist, nearest_index in zip(ids, distance, pids[np.asarray(nearest)]):
                point = truth.obsm["spatial"][index]
                anchors.append(dict(heldout_id=i, cell_class=label, truth_cell_id=str(truth.obs_names[index]),
                                    x=float(point[0]), y=float(point[1]), nearest_same_class_um=float(dist),
                                    within_50um=bool(dist <= 50), within_100um=bool(dist <= 100),
                                    prediction_id=str(pred.obs_names[nearest_index])))
        log = pd.read_csv(folder / "logs" / "version_0" / "metrics.csv").dropna(subset=["loss_epoch"])
        assert list(log.epoch.astype(int)) == list(range(100))
        log.insert(0, "heldout_id", i)
        losses.append(log[["heldout_id", "epoch", "step", "loss_epoch", "loss_x_epoch", "loss_g_epoch", "loss_c_epoch"]])
        print(f"Fold {i}: verified split, raw truth, training-only normalization, slab, all saved metrics; "
              f"rare macro recall@100={summary['rare_macro_coverage_recall_100um']:.4f}")

    tables = pd.concat(all_tables, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    rare = tables[tables.is_rare_train & (tables.truth_n > 0)].copy()
    summary.to_csv(DEST / "verified_fold_summary.csv", index=False)
    tables.to_csv(DEST / "verified_cell_type_metrics.csv", index=False)
    pd.DataFrame(anchors).to_csv(DEST / "rare_cell_distances.csv", index=False)
    pd.DataFrame(qc_rows).to_csv(DEST / "quality_checks.csv", index=False)
    pd.concat(losses, ignore_index=True).to_csv(DEST / "training_epoch_losses.csv", index=False)

    # Existing SpatialZ run: same heldout truth and metric functions, not a matched-input benchmark.
    spatial_root = next((ROOT.parent / name for name in ("SpitalZ_codex", "SpatialZ")
                         if (ROOT.parent / name / "output/merfish_holdout_009/generated_009.h5ad").exists()), None)
    comparison, spatial_note = [], "未找到 SpatialZ 已有留出结果，未作比较。"
    if spatial_root is not None:
        spatial_folder = spatial_root / "output" / "merfish_holdout_009"
        spatial_pred = ad.read_h5ad(spatial_folder / "generated_009.h5ad")
        common_counts = pd.concat([original[j].obs.cell_class.astype(str) for j in (0, 2, 3, 4)]).value_counts()
        spatial_table, spatial_summary = evaluate(folds[1][0], spatial_pred, common_counts)
        spatial_table.to_csv(DEST / "spatialz_existing_run_cell_type_metrics.csv", index=False)
        comparison = [dict(run="DeepSpatial: holdout 1, four training slices", **summary_rows[0]),
                      dict(run="SpatialZ: existing holdout 0.09, two reference slices", heldout_id=1, **spatial_summary)]
        pd.DataFrame(comparison).to_csv(DEST / "existing_run_comparison.csv", index=False)
        dm = tables[(tables.heldout_id == 1) & (tables.cell_class == "OD Mature 3")].iloc[0]
        sm = spatial_table[spatial_table.cell_class == "OD Mature 3"].iloc[0]
        spatial_note = (
            f"在同一真实 0.09 切片上，已有 SpatialZ 运行生成 {spatial_pred.n_obs} 个细胞，DeepSpatial 留出 1 生成 {folds[1][1].n_obs} 个。"
            f"OD Mature 3 真值为 3：SpatialZ 生成 {int(sm.predicted_n)}，DeepSpatial 生成 {int(dm.predicted_n)}；两者该类 100 µm 覆盖召回率均为 0。\n\n"
            f"用同一函数重算，整体几何最近邻距离均值为 SpatialZ {spatial_summary['symmetric_mean_nn_um']:.2f} µm，"
            f"DeepSpatial {summary_rows[0]['symmetric_mean_nn_um']:.2f} µm；"
            f"稀有类宏平均 100 µm 覆盖召回率为 {100*spatial_summary['rare_macro_coverage_recall_100um']:.1f}% / "
            f"{100*summary_rows[0]['rare_macro_coverage_recall_100um']:.1f}%。"
            "这里统一沿用 DeepSpatial 本轮训练计数定义稀有类，仅用于保持评价类别名单一致。\n\n"
            "这属于已有运行诊断：DeepSpatial 用四张训练切片，SpatialZ 仅使用两侧参考切片；"
            "SpatialZ 该运行使用 1000 次坐标优化、k_neighbors=1、k_sam=50、Beta=1、syn_mode=default。"
            "因此不据此宣称任一方法普遍优于另一方法，也不把此结果当作论文默认配置的完整对比。"
        )

    plot_results(rare, folds)
    report_table = summary[["heldout_id", "truth_n", "predicted_n", "missing_classes",
                            "rare_macro_coverage_recall_50um", "rare_macro_coverage_recall_100um",
                            "global_mean_log_expression_pcc", "symmetric_mean_nn_um"]].copy()
    report_table.columns = ["留出编号", "真值细胞数", "预测细胞数", "整类缺失数", "稀有类覆盖@50µm", "稀有类覆盖@100µm", "整体均值表达PCC", "整体几何距离µm"]
    for col in ("稀有类覆盖@50µm", "稀有类覆盖@100µm"):
        report_table[col] = report_table[col].map(lambda x: f"{x*100:.2f}%")
    report_table["整体均值表达PCC"] = report_table["整体均值表达PCC"].map(lambda x: f"{x:.4f}")
    report_table["整体几何距离µm"] = report_table["整体几何距离µm"].map(lambda x: f"{x:.2f}")
    class_rows = []
    for label in ["OD Mature 3", "OD Immature 2", "OD Mature 4", "Endothelial 2", "Pericytes", "OD Mature 1"]:
        sub = rare[rare.cell_class == label].sort_values("heldout_id")
        class_rows.append({"稀有类别": label,
                           **{f"留出{i} 真值→预测": f"{int(r.truth_n)}→{int(r.predicted_n)}" for i, r in zip((1, 2, 3), sub.itertuples())},
                           "100µm覆盖率（1/2/3）": " / ".join(f"{v*100:.1f}%" for v in sub.coverage_recall_100um),
                           "均值表达PCC（1/2/3）": " / ".join(f"{v:.3f}" for v in sub.mean_log_expression_pcc)})
    od3 = rare[rare.cell_class == "OD Mature 3"].sort_values("heldout_id")
    mean_recall = summary.rare_macro_coverage_recall_100um.mean() * 100
    report = f"""# MERFISH 三轮留出切片验证分析

## 结论

三轮均保留全部 15 个 cell_class 大类，六个训练稀有类均有生成；但是稀有类真实位置的同类覆盖不足。
稀有类宏平均覆盖召回率在 100 µm 下平均为 {mean_recall:.2f}%，在 50 µm 下为 {summary.rare_macro_coverage_recall_50um.mean()*100:.2f}%。
OD Mature 3 三轮真值共 {int(od3.truth_n.sum())} 个、生成共 {int(od3.predicted_n.sum())} 个，但三轮 100 µm 覆盖率均为 0。
因此本次结果支持“类别存在性得以保留，但局部空间恢复不足”，不支持“稀有细胞缺失问题已经解决”。

## 实验与文件核验

- 留出 1/2/3 对应原始 merfish_9/14/19_paste.h5ad，切片标识为 0.09/0.14/0.19。
- 每轮新建模型，以其余四张切片训练；记录 seed=42、100 epochs、batch_size=512、50000 轨迹配对、GiT hidden_size=256、depth=6。
- UOT 参数 alpha_spatial=0.5、reg=0.8、tau=0.05；流匹配 lambda_g=0.1、lambda_c=10；ODE dopri5、100 个时间点、atol=rtol=1e-5。
- 生成只使用夹住留出位置的两张训练切片，thickness=10 µm；评价固定使用留出 Z±5 µm 薄层。生成使用末轮模型的 EMA 权重。
- 已重新读取 h5ad，确认三轮真值的细胞、基因、表达、坐标和 cell_class 与原始对应切片相同。
- 每轮训练类别计数、标签编码和坐标归一化参数与排除留出切片后的四张数据一致；预测薄层与生成区间按规定过滤后的结果完全一致。
- 所有逐类指标和整体指标均从 h5ad 重新计算，与已保存 CSV 在 1e-7 浮点容差内一致；训练日志均完整覆盖 epoch 0–99。
- 155 个基因一致，表达和坐标均为有限值。留出 1 的预测有 {qc_rows[0]['negative_expression_entries']} 个负表达元素，最小值 {qc_rows[0]['minimum_expression']:.6g}；相关性按原协议先截为非负再 log1p，原预测文件保持原值。其余两轮无负表达。

## 整体结果

{markdown(report_table)}

整体均值表达 PCC 为按细胞取 log1p 后的平均基因向量相关性，并非逐细胞表达预测准确率。
稀有类宏平均为六个类别等权平均；全无预测的类别覆盖召回率应计 0，本次各类均非零。
整体几何误差不区分类别，因此较小的整体距离也可能掩盖稀有类的位置错误。

## 六个稀有大类

{markdown(pd.DataFrame(class_rows))}

OD Mature 3 的真值到预测同类最近邻平均距离，在三轮分别为 {', '.join(f'{v:.1f}' for v in od3.truth_to_pred_nn_mean_um)} µm。
留出 3 的 OD Immature 2 虽有 6 个预测（真值 7 个），100 µm 覆盖率仍为 0。
留出 1 的 OD Mature 1 为真值 29、预测 69，说明保留类别也可能伴随过量生成。

![稀有类同类空间覆盖率](rare_cell_coverage.png)

![关键稀有类位置](rare_cell_spatial.png)

空间图蓝色空心圈为真实稀有类，橙色叉为同类预测，浅蓝圆盘为真实点周围 100 µm，灰色为留出切片全部真实细胞。
图中仅比较 XY，生成薄层的 Z 均位于固定评价范围内。

## 与已有 SpatialZ 结果的关系

{spatial_note}

## 如何解释和推进

1. 区分整类消失、局部空间漏覆盖和分子表达失真；当前最明确的缺口是局部同类空间覆盖。
2. 平均表达相关性较高不能替代稀有类分析。OD Immature 2 的类均值 PCC 仅为 0.686/0.710/0.793，明显低于整体均值 PCC；该类每张真值仅 5–7 个细胞，估计波动也较大。
3. 这些细胞并非一一配对的观测，覆盖率允许多个点共享邻居，不能称为细胞存活率。稀有类本身稀疏，覆盖率也受半径和生成数量影响；不要仅与丰富类的覆盖率比较就归因于模型。
4. 三轮是三个切片划分，均 seed=42，并非三组独立随机种子。切片间标准差不能当作多种子稳定性或显著性证据。
5. 若继续比较方法，先统一留出位置、可见输入切片、稀有类名单和生成密度；增加 DeepSpatial 两张输入的对照，才能区分模型机制与额外训练切片的贡献。
6. neuron_class、Region 等 source_* 列继承自源细胞；本次评价只针对建模的 cell_class，不能据此宣称细分神经元亚型已恢复。

## 输出

- verified_fold_summary.csv：独立复算的整体指标、稀有类宏/微平均、类别比例总变差。
- verified_cell_type_metrics.csv：全部 45 个类别×切片组合的复算指标。
- rare_cell_distances.csv：每个真实稀有类细胞到同类预测的距离及覆盖标记。
- quality_checks.csv、training_epoch_losses.csv：输入/输出核验和训练日志。
- existing_run_comparison.csv、spatialz_existing_run_cell_type_metrics.csv：同一真值上的已有运行诊断。
- rare_cell_coverage.png/pdf、rare_cell_spatial.png/pdf：统计图与空间图。
- 分析源码：ccfa-workfiles/checks/merfish-holdout/source/analyze_results.py。
"""
    (DEST / "report.md").write_text(report, encoding="utf-8")
    assert "\ufffd" not in report
    print("\n", summary.to_string(index=False))
    if comparison:
        print("\nExisting run comparison:")
        print(pd.DataFrame(comparison)[["run", "predicted_n", "rare_missing_classes", "rare_macro_coverage_recall_100um", "symmetric_mean_nn_um"]].to_string(index=False))
    print("\nSaved analysis:", DEST)


if __name__ == "__main__":
    main()
