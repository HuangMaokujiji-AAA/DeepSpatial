"""Evaluate downloaded G volumes against the canonical <=5% protocol and matched controls."""
from pathlib import Path
import gc
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[4]
PROJECT = ROOT / 'HM_DeepSpital'
DEST = PROJECT / 'outputs/g_holdouts_1_2_3/analysis'
DEST.mkdir(exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(DEST / '.matplotlib')
sys.path[:0] = [str(PROJECT), str(PROJECT / '.runtime_deps')]

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

from examples.resume_g import inspect_results, verify_schedule
from examples.run_g import file_sha256
from recompute_rare_metrics import evaluation_functions
from analyze_three_holdouts import parent_copy, extra_metrics
from analyze_v1 import dense, md_table, niche_composition, spatial_expression

RUNS = ['DeepSpatial', 'V1', 'S', 'Source copy', 'G real', 'G zero', 'G shuffle']


def main():
    evaluate, threshold = evaluation_functions()
    assert threshold == .05
    status = inspect_results(PROJECT / 'outputs/g_holdouts_1_2_3')
    status.to_csv(DEST / 'integrity.csv', index=False)
    original = {i: ad.read_h5ad(next((ROOT / 'data/merfish_mouse_hypothalamus').rglob(f'merfish_{i}.h5ad'))) for i in range(5)}
    summaries, tables, support_rows, displacement_rows, anchor_rows, spatial_rows, check_rows = [], [], [], [], [], [], []
    plots = {}
    for fold in (1, 2, 3):
        truth = original[fold]
        training = pd.concat([a.obs.cell_class.astype(str) for i, a in original.items() if i != fold]).value_counts()
        rare_classes = set(training[(training / training.sum()) <= .05].index)
        folder = PROJECT / f'outputs/g_holdouts_1_2_3/holdout_{fold}'
        shared = json.loads((folder / 'shared/run_config.json').read_text(encoding='utf-8'))
        expected = shared['experiment_g']['source_schedule_sha256']
        pair_hash = file_sha256(folder / 'shared/pair_indices.npz')
        assert pair_hash == shared['experiment_g']['shared_pair_sha256']
        runs = {'DeepSpatial': ad.read_h5ad(ROOT / f'output/merfish_holdout/holdout_{fold}/predicted_holdout.h5ad')}
        reference_obs = None
        for name, path in [('V1', PROJECT / f'outputs/v1_holdouts_1_2_3/holdout_{fold}'),
                           ('S', PROJECT / f'outputs/s_holdouts_1_2_3/holdout_{fold}'),
                           *[(f'G {mode}', folder / mode) for mode in ('real', 'zero', 'shuffle')]]:
            if not (path / 'generated.h5ad').is_file():
                continue
            if name.startswith('G ') and status[(status.holdout == fold) & (status['mode'] == name[2:])].action.iloc[0] != 'skip':
                continue
            volume = ad.read_h5ad(path / 'generated.h5ad')
            assert truth.var_names.equals(volume.var_names)
            assert not volume.obs.is_observed.any() and fold * 50 not in set(volume.obs.source_z)
            if name == 'S':
                verify_schedule(volume.obs, 'z_coord', expected)
                reference_obs = volume.obs.copy()
                assert file_sha256(path / 'generated.h5ad') == shared['experiment_g']['source_s_h5ad_sha256']
            if name.startswith('G '):
                verify_schedule(volume.obs, 'z_coord', expected)
                cfg = json.loads((path / 'run_config.json').read_text(encoding='utf-8'))
                assert cfg['experiment_g']['shared_pair_sha256'] == pair_hash
                assert cfg['run']['heldout_indices'] == [fold]
                assert cfg['experiment_g']['implementation_sha256'] == shared['experiment_g']['implementation_sha256']
                for key in ('seed', 'model', 'training', 'slice_z_um'):
                    assert cfg[key] == shared[key]
                observed_source = volume.obs.source_cell_class.astype(str).to_numpy()
                anchors = volume.obs.generation_mode.to_numpy() == 'source_anchor'
                check_rows.append(dict(heldout_id=fold, run=name, volume_n=volume.n_obs,
                                       source_anchor_n=int(anchors.sum()), source_anchor_fraction=float(anchors.mean()),
                                       class_switches=int((observed_source != volume.obs.cell_class.astype(str).to_numpy()).sum()),
                                       initial_state_sha256=cfg['experiment_g']['initial_state_sha256'],
                                       paired_condition_sha256=cfg['experiment_g']['paired_condition_sha256'],
                                       resumed=(path / 'resume_history.jsonl').is_file()))
                np.testing.assert_allclose(volume.obsm['spatial_3d'][:, :2], volume.obsm['spatial'])
                np.testing.assert_array_equal(volume.obsm['spatial_3d'][:, 2], volume.obs.z_coord)
                assert np.array_equal(anchors, volume.obs.matching_candidate_count.to_numpy() == 0)
            pred = volume[abs(volume.obs.z_coord - fold * 50) <= 5].copy()
            runs[name] = pred
            del volume
        assert reference_obs is not None
        copy_pred = parent_copy(runs['S'], original)
        runs['Source copy'] = copy_pred
        plots[fold] = {name: (np.asarray(a.obsm['spatial']), a.obs.cell_class.astype(str).to_numpy())
                       for name, a in [('Truth', truth), *runs.items()]}
        niche = niche_composition(truth, runs)
        local_maps = [dict(heldout_id=fold, **row) for width in (100, 200) for row in spatial_expression(truth, runs, width)]
        spatial_rows.extend(local_maps)
        spatial_frame = pd.DataFrame(local_maps)
        local = {}
        for name, pred in runs.items():
            table, summary = evaluate(truth, pred, training)
            rare = table[table.is_rare_train & (table.truth_n > 0)]
            summary.update(heldout_id=fold, run=name, **extra_metrics(truth, pred, table), **niche[name])
            for radius in (25, 50, 100):
                summary[f'rare_micro_recall_{radius}um'] = float(np.average(rare[f'coverage_recall_{radius}um'], weights=rare.truth_n))
                present = rare.predicted_n > 0
                summary[f'rare_micro_precision_{radius}um'] = float(np.average(rare.loc[present, f'coverage_precision_{radius}um'], weights=rare.loc[present, 'predicted_n']))
            for width in (100, 200):
                frame = spatial_frame[(spatial_frame.run == name) & (spatial_frame.bin_width_um == width)]
                summary[f'spatial_expression_pcc_{width}um'] = float(frame.spatial_expression_pcc.median())
                summary[f'spatial_expression_bins_{width}um'] = int(frame.shared_bins.iloc[0])
            if name.startswith('G '):
                cp = parent_copy(pred, original)
                distance = np.linalg.norm(pred.obsm['spatial'] - cp.obsm['spatial'], axis=1)
                labels = pred.obs.source_cell_class.astype(str).to_numpy()
                anchors = pred.obs.generation_mode.to_numpy() == 'source_anchor'
                if anchors.any():
                    np.testing.assert_array_equal(pred.obsm['spatial'][anchors], cp.obsm['spatial'][anchors])
                    np.testing.assert_array_equal(dense(pred)[anchors], dense(cp)[anchors])
                for label in sorted(training.index):
                    mask = labels == label
                    displacement_rows.append(dict(heldout_id=fold, run=name, cell_class=label, is_rare_train=label in rare_classes,
                                                  predicted_n=int(mask.sum()), source_anchor_n=int((mask & anchors).sum()),
                                                  source_anchor_fraction=float(anchors[mask].mean()) if mask.any() else np.nan,
                                                  displacement_median_um=float(np.median(distance[mask])) if mask.any() else np.nan,
                                                  ode_displacement_median_um=float(np.median(distance[mask & ~anchors])) if (mask & ~anchors).any() else np.nan))
                    if label in rare_classes:
                        query = truth.obsm['spatial'][truth.obs.cell_class.astype(str).to_numpy() == label]
                        pm = pred.obs.cell_class.astype(str).to_numpy() == label
                        ode = pred.obsm['spatial'][pm & ~anchors]
                        both = pred.obsm['spatial'][pm]
                        ode_d = cKDTree(ode).query(query)[0] if len(ode) else np.full(len(query), np.inf)
                        all_d = cKDTree(both).query(query)[0] if len(both) else np.full(len(query), np.inf)
                        anchor_rows.append(dict(heldout_id=fold, run=name, cell_class=label, truth_n=len(query),
                                                full_recall_100um=float((all_d <= 100).mean()),
                                                ode_only_recall_100um=float((ode_d <= 100).mean()),
                                                truth_added_by_anchors=int(((all_d <= 100) & (ode_d > 100)).sum())))
                summary['source_anchor_fraction'] = float(anchors.mean())
                summary['rare_source_anchor_fraction'] = float(anchors[np.isin(labels, list(rare_classes))].mean())
                summary['parent_displacement_median_um'] = float(np.median(distance))
            local[name] = (table, summary)
            tables.append(table.assign(heldout_id=fold, run=name))
        common = set.intersection(*(set(t.loc[t.predicted_n > 0, 'cell_class']) for t, s in local.values()))
        for table, summary in local.values():
            shared_rare = table[table.cell_class.isin(common) & table.is_rare_train & (table.truth_n > 0)]
            summary['rare_common_expression_pcc'] = float(shared_rare.mean_log_expression_pcc.mean())
            summary['rare_common_expression_class_n'] = len(shared_rare)
            summaries.append(summary)
        support = pd.read_csv(folder / 'shared/matching_support.csv')
        support['is_rare_train'] = support.cell_class.isin(rare_classes)
        support_rows.append(support.assign(heldout_id=fold))
        print('FOLD', fold, json.dumps({name: s['rare_macro_coverage_recall_100um'] for name, (t, s) in local.items()}), flush=True)
        del runs, local
        gc.collect()
    result = pd.DataFrame(summaries)
    table = pd.concat(tables, ignore_index=True)
    checks = pd.DataFrame(check_rows)
    assert (checks.groupby('heldout_id').initial_state_sha256.nunique() == 1).all()
    for name, frame in [('metrics', result), ('cell_type_metrics', table), ('qc', checks),
                        ('source_displacement', pd.DataFrame(displacement_rows)), ('anchor_contribution', pd.DataFrame(anchor_rows)),
                        ('spatial_expression', pd.DataFrame(spatial_rows)), ('matching_support', pd.concat(support_rows))]:
        frame.to_csv(DEST / f'{name}.csv', index=False)
    metrics = ['rare_macro_coverage_recall_25um', 'rare_macro_coverage_recall_50um', 'rare_macro_coverage_recall_100um',
               'rare_micro_recall_100um', 'rare_micro_precision_100um', 'rare_common_expression_pcc',
               'spatial_expression_pcc_100um', 'tissue_histogram_tv', 'niche_js_divergence_mean', 'cell_fraction_total_variation']
    means = result.groupby('run')[metrics].agg(['mean', 'std', 'count'])
    means.columns = ['_'.join(v) for v in means.columns]
    means.to_csv(DEST / 'metrics_mean_std.csv')
    comparisons = []
    for baseline, new in [('S', 'G real'), ('Source copy', 'G real'), ('G shuffle', 'G real'), ('G zero', 'G real')]:
        a = result[result.run == baseline].set_index('heldout_id')
        b = result[result.run == new].set_index('heldout_id')
        paired = a.index.intersection(b.index)
        for metric in metrics:
            for fold in paired:
                comparisons.append(dict(baseline=baseline, new=new, heldout_id=fold, metric=metric,
                                        baseline_value=a.loc[fold, metric], new_value=b.loc[fold, metric],
                                        delta=b.loc[fold, metric] - a.loc[fold, metric]))
    pd.DataFrame(comparisons).to_csv(DEST / 'paired_comparisons.csv', index=False)
    make_plots(result, table, plots)
    protocol = dict(rare_threshold=.05, inclusive=True, evaluation_slab_half_width_um=5, radii_um=[25, 50, 100],
                    rarity_source='pooled visible training slices', folds=[1, 2, 3], seed=42,
                    incomplete=[dict(heldout=int(r.holdout), mode=r.mode, action=r.action) for r in status.itertuples() if r.action != 'skip'],
                    fold_sd_is_not_multiple_seed_uncertainty=True, zero_comparison_uses_only_common_completed_folds=True)
    (DEST / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    write_report()
    print('MEANS\n', means.to_string(), flush=True)


def make_plots(result, table, plots):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none'})
    colors = {'S': '#758493', 'G real': '#cb503e', 'G zero': '#55945f', 'G shuffle': '#655ab0', 'Source copy': '#bd9653'}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for ax, (key, title, scale) in zip(axes, [('rare_macro_coverage_recall_100um', 'Rare macro recall @100 um (%)', 100),
                                          ('rare_micro_recall_100um', 'Rare cell-weighted recall @100 um (%)', 100),
                                          ('spatial_expression_pcc_100um', 'Spatial expression PCC (100 um bins)', 1)]):
        for name, color in colors.items():
            frame = result[result.run == name].sort_values('heldout_id')
            ax.plot(frame.heldout_id, frame[key] * scale, '-o', label=name, color=color, lw=1.8, ms=5)
        ax.set(xticks=[1, 2, 3], xlabel='Held-out slice', title=title)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    for ext in ('png', 'svg'):
        fig.savefig(DEST / f'comparison.{ext}', dpi=180)
    plt.close(fig)
    rare = table[table.is_rare_train]
    classes = rare.groupby('cell_class').train_fraction.mean().sort_values().index
    methods = [m for m in ('S', 'Source copy', 'G real', 'G zero', 'G shuffle') if m in set(rare.run)]
    # 分折展示，避免将缺少第三折的 zero 与三折均值混为一谈。
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.8), layout='constrained')
    for ax, fold in zip(axes, (1, 2, 3)):
        pivot = rare[rare.heldout_id == fold].pivot(index='cell_class', columns='run', values='coverage_recall_100um').reindex(index=classes, columns=methods)
        values = pivot.to_numpy() * 100
        ax.imshow(np.ma.masked_invalid(values), vmin=0, vmax=100, cmap='YlGnBu', aspect='auto')
        for r in range(len(classes)):
            for c in range(len(methods)):
                value = values[r, c]
                ax.text(c, r, 'NA' if np.isnan(value) else f'{value:.0f}', ha='center', va='center', fontsize=8, color='white' if value > 65 else '#222')
        ax.set(xticks=range(len(methods)), xticklabels=methods, yticks=range(len(classes)), yticklabels=classes, title=f'Holdout {fold}: recall @100 um (%)')
        ax.tick_params(axis='x', rotation=40)
    for ext in ('png', 'svg'):
        fig.savefig(DEST / f'rare_types.{ext}', dpi=180)
    plt.close(fig)


def write_report():
    """Generate the interpretation from saved numeric evidence without modifying experiment results."""
    s = pd.read_csv(DEST / 'metrics.csv')
    t = pd.read_csv(DEST / 'cell_type_metrics.csv')
    status = pd.read_csv(DEST / 'integrity.csv')
    qc = pd.read_csv(DEST / 'qc.csv')
    comparison = pd.read_csv(DEST / 'paired_comparisons.csv')
    displacement = pd.read_csv(DEST / 'source_displacement.csv')
    anchors = pd.read_csv(DEST / 'anchor_contribution.csv')
    means = s.groupby('run').mean(numeric_only=True)
    # Canonical baseline measurements must remain unchanged by adding the G methods.
    previous = pd.read_csv(PROJECT / 'outputs/s_holdouts_1_2_3/analysis/rare_cell_type_metrics.csv')
    rename = {'DeepSpatial': 'DeepSpatial', 'V1 reg=0.05': 'V1', 'Experiment S': 'S'}
    columns = ['truth_n', 'predicted_n', 'train_fraction', 'coverage_recall_25um',
               'coverage_recall_50um', 'coverage_recall_100um', 'mean_log_expression_pcc']
    for old, new in rename.items():
        a = previous[previous.run == old].set_index(['heldout_id', 'cell_class']).sort_index()
        b = t[t.run == new].set_index(['heldout_id', 'cell_class']).sort_index()
        assert a.index.equals(b.index)
        np.testing.assert_allclose(a[columns], b[columns], rtol=1e-7, atol=1e-7, equal_nan=True)
    overall = []
    for method in RUNS:
        row = means.loc[method]
        overall.append({'方法': method, '完成折数': str(len(s[s.run == method])),
                        '稀有宏召回25µm': f'{row.rare_macro_coverage_recall_25um:.2%}',
                        '稀有宏召回50µm': f'{row.rare_macro_coverage_recall_50um:.2%}',
                        '稀有宏召回100µm': f'{row.rare_macro_coverage_recall_100um:.2%}',
                        '稀有加权召回100µm': f'{row.rare_micro_recall_100um:.2%}',
                        '空间表达PCC': f'{row.spatial_expression_pcc_100um:.4f}',
                        '密度TV↓': f'{row.tissue_histogram_tv:.4f}'})
    fold_table = s.pivot(index='heldout_id', columns='run', values='rare_macro_coverage_recall_100um')
    fold_rows = []
    for fold in (1, 2, 3):
        row = {'留出': str(fold)}
        for method in ('S', 'Source copy', 'G real', 'G zero', 'G shuffle'):
            value = fold_table.loc[fold].get(method, np.nan)
            row[method] = '缺失' if np.isnan(value) else f'{value:.2%}'
        fold_rows.append(row)
    controls = []
    for base in ('S', 'Source copy', 'G zero', 'G shuffle'):
        frame = comparison[comparison.baseline == base]
        def delta(metric):
            return float(frame[frame.metric == metric].delta.mean())
        controls.append({'比较': f'G real − {base}',
                         '共同完成折数': str(frame.heldout_id.nunique()),
                         '100µm宏召回变化(pp)': f"{100 * delta('rare_macro_coverage_recall_100um'):+.3f}",
                         '100µm加权召回变化(pp)': f"{100 * delta('rare_micro_recall_100um'):+.3f}",
                         '空间表达PCC变化': f"{delta('spatial_expression_pcc_100um'):+.5f}"})
    rare = t[t.is_rare_train]
    types = rare.pivot_table(index='cell_class', columns='run', values='coverage_recall_100um')
    type_rows = []
    for label in rare.groupby('cell_class').train_fraction.mean().sort_values().index:
        row = {'细胞类型': label}
        for method in ('S', 'Source copy', 'G real', 'G zero', 'G shuffle'):
            row[method] = f'{types.loc[label, method]:.2%}'
        type_rows.append(row)
    real = means.loc['G real']
    base = means.loc['S']
    copied = means.loc['Source copy']
    per_type_displacement = displacement[displacement.run == 'G real']
    anchor_rows = []
    for label in ('Endothelial 2', 'OD Immature 2', 'OD Mature 3', 'OD Mature 4', 'Pericytes'):
        d = per_type_displacement[per_type_displacement.cell_class == label].sort_values('heldout_id')
        anchor_rows.append({'类别': label, '来源复制占比（留出1/2/3）': ' / '.join(f'{v:.1%}' for v in d.source_anchor_fraction)})
    contribution = anchors[anchors.run == 'G real'].groupby('heldout_id').agg(
        full=('full_recall_100um', 'mean'), ode_only=('ode_only_recall_100um', 'mean'), added=('truth_added_by_anchors', 'sum'))
    report = f'''**实验 G 分析：九组结果、来源复制基线与生态位对照**

结论：G real 相比 S 减轻了组织密度收缩，并提高了部分稀有类的宽半径覆盖；它的主要空间结果接近同调度来源复制。real/zero/shuffle 未显示真实生态位条件具有跨指标稳定的独立正贡献。极稀有类定位仍未解决。

**完整性与评价口径**

- 本地最新状态为 {int((status.action == 'skip').sum())}/9 组完整；第三折 zero 与 shuffle 均已达到 100 epochs，最后检查点 epoch=99、global_step=9800。先前第三折 zero 缺失是当时本地文件不全，最新同步后已补齐，无需重训。
- 三折每组全体积分别为 114757、112777、113664 个生成细胞；隐藏薄层生成数为 5707、5680、5735。九组模型可读，最终模型与最后训练检查点一致，生成数组无 NaN/Inf，预算和支持计数匹配。
- 核对同折各组的配对、初始权重、实现身份和 S 的来源/目标深度调度；类别切换总数为 {int(qc.class_switches.sum())}。来源复制使用 S 的相同 parent、数量和目标深度，只保留来源 XY/表达/类型。
- 稀有类为每折四张可见训练切片合并丰度 ≤5%，三折各 11 类；隐藏评价薄层为 Z±5µm，同类覆盖半径 25/50/100µm。未使用隐藏丰度选择稀有类。
- 各表为三折等权均值。宏召回对类型等权；加权召回按各类真实数量加权。空间表达 PCC 是共同 100µm 网格中各基因相关性的中位数，不等同于类别平均表达 PCC。
- 原版/V1/S 的逐类结果与先前 ≤5% 复算表在 1e-7 容差内一致。`metrics_mean_std.csv` 的标准差表示切片间变化，不是多随机种子不确定性。

**总体结果**

{md_table(pd.DataFrame(overall))}

G real 相对 S：100µm 稀有宏召回提高 {100 * (real.rare_macro_coverage_recall_100um - base.rare_macro_coverage_recall_100um):.2f} 个百分点，但稀有细胞加权召回仅提高 {100 * (real.rare_micro_recall_100um - base.rare_micro_recall_100um):.2f} 个百分点。25µm 宏召回几乎不变，50µm 有提高。宽半径收益集中于少数类型，不能视为所有稀有细胞定位同时改善。

密度 TV 从 {base.tissue_histogram_tv:.4f} 降至 {real.tissue_histogram_tv:.4f}，XY 标准差/真值标准差的均值从 {base.xy_sd_ratio_mean:.4f} 恢复到 {real.xy_sd_ratio_mean:.4f}。同时，空间表达 PCC 从 {base.spatial_expression_pcc_100um:.4f} 降至 {real.spatial_expression_pcc_100um:.4f}。G 并非全面优于 S。

**分折及对照**

{md_table(pd.DataFrame(fold_rows))}

G real 对 S 在留出 1/2/3 的 100µm 宏召回分别变化 +6.28 / −0.93 / +5.99 个百分点，存在划分依赖。

{md_table(pd.DataFrame(controls))}

G real 的 100µm 宏召回为 {real.rare_macro_coverage_recall_100um:.4%}，同调度来源复制为 {copied.rare_macro_coverage_recall_100um:.4%}，仅差 {100 * (real.rare_macro_coverage_recall_100um - copied.rare_macro_coverage_recall_100um):.4f} 个百分点。G 的 25µm 宏召回比复制略高，50µm 略低，空间表达 PCC 则低于复制。当前不能据此认为模型已学到明显超出来源保留的空间信息。

zero 的空间表达 PCC 在三个切片均高于 real；shuffle 也在三个切片均高于 real。真实生态位在 25µm 宏召回、100µm precision 上有小幅正向迹象，但 100µm 宏召回近乎持平，加权召回和空间表达没有对应正收益。因此本轮不支持“生态位条件独立提高整体重建质量”的结论，也不足以证明生态位在其他设计中无用。

只有 seed=42 的三折实验，没有多种子显著性证据。第三折 shuffle 有从 10 epochs 检查点恢复的记录，其后随机数流不保证与不中断运行逐步相同；细小组间差异尤其不宜作强因果结论。

**逐类及无支持来源**

{md_table(pd.DataFrame(type_rows))}

Endothelial 2 与 OD Mature 4 对 S 的三折宏召回改善较大，但 G real 在每一折都与来源复制完全相同。OD Mature 4 的真实细胞仅 11/11/8 个，留出 3 的 62.5% 只对应 8 个真实细胞中的 5 个被覆盖；小分母会放大宏平均波动。

OD Mature 3 三折仍全部为 0。OD Immature 2 为 20%/0/0，第一折仅覆盖 5 个真实细胞中的 1 个，而且该覆盖完全来自 source_anchor；ODE 支持部分对这一类三折召回仍为 0。因此不能将此结果写成极稀有类定位问题已解决。

{md_table(pd.DataFrame(anchor_rows))}

全体积 source_anchor 占比约 1.03%–1.09%；评价薄层仅约 0.45%–0.54%，但极稀有类型的比例很高。G real 在薄层所有生成细胞相对来源的位移中位数为 2.84/3.47/4.33µm，多数常见类和中等稀有类的 ODE 位移中位数也只有几微米。结果接近复制不仅由显式 fallback 引起，也体现为有候选分支的运动很小。

作为机制诊断，移除 source_anchor 后 100µm 宏召回从 {contribution.loc[1, 'full']:.2%}/{contribution.loc[2, 'full']:.2%}/{contribution.loc[3, 'full']:.2%} 变为 {contribution.loc[1, 'ode_only']:.2%}/{contribution.loc[2, 'ode_only']:.2%}/{contribution.loc[3, 'ode_only']:.2%}，复制分支额外覆盖 12/3/14 个稀有真实细胞。主结果保留所有细胞；该拆分改变了预测数量，只用于解释来源贡献，不能拿来替代主要比较。

**原因判断与下一步**

观测支持“局部配对及保守推理减少错误漂移，使结果回到来源保留附近”。物理支持、物理成本和 fallback 本轮同时改变，尚不能分离三者贡献；细胞位移很小也可能反映局部候选方向相互抵消或模型采用近似恒等映射，需查看实际训练目标位移及条件速度后再确定。

建议按以下次序推进：

1. 把同调度来源复制固定为必报基线，明确同时超过它的空间覆盖与空间表达证据，再判断新增机制是否成立。
2. 固定 zero 条件拆开候选约束、物理成本、fallback；保持 S 调度、模型和训练预算一致，定位收益由哪一项产生。先审查每类训练目标位移、同一 parent 候选方向与预测位移，避免只调整半径而重新引入漂移。
3. 对 OD Mature 3/OD Immature 2 保留无支持标记和逐类结果。候选缺失时不能靠补抽样或强行远距离配对声称恢复位置，需要额外可见解剖约束或可验证的跨切片匹配依据。
4. 在几何行为明确后，再用多个随机种子的 real/zero/shuffle 检查生态位增益。当前真实条件没有稳定优势，单纯增加模块容量缺少本轮证据支持。

**证据文件**

- `integrity.csv`：九组训练和生成完整性。
- `metrics.csv` / `metrics_mean_std.csv`：每折指标及切片间均值/标准差。
- `cell_type_metrics.csv`：全部类型的计数、训练丰度、空间覆盖和表达相关。
- `paired_comparisons.csv`：同折的 G real−S / copy / zero / shuffle 差值。
- `source_displacement.csv` / `anchor_contribution.csv`：逐类位移、fallback 比例及覆盖贡献。
- `qc.csv` / `protocol.json`：身份核验和评价定义。
- `comparison.png` / `rare_types.png`：三折对比与逐类热图，另有 SVG。
'''
    (DEST / 'report.md').write_text(report, encoding='utf-8')


if __name__ == '__main__':
    main()
