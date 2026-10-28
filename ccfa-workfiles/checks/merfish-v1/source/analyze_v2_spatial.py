"""Read-only evaluation of completed V2 spatial ablations and identical-schedule controls."""
from pathlib import Path
import gc
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[4]
PROJECT = ROOT / 'HM_DeepSpital'
RUN = PROJECT / 'outputs/v2_spatial_holdouts_1_2_3'
DEST = RUN / 'analysis'
DEST.mkdir(exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(DEST / '.matplotlib')
sys.path[:0] = [str(PROJECT), str(PROJECT / '.runtime_deps')]

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from examples.resume_g import verify_schedule
from examples.run_s import file_sha256
from deepspatial.data_utils.spatial_transport import spatial_transport_volume
from recompute_rare_metrics import evaluation_functions
from analyze_three_holdouts import parent_copy, extra_metrics
from analyze_v1 import dense, md_table, niche_composition, spatial_expression

MODES = {'source_xy': 'V2 source XY', 'ot_barycentric': 'V2 barycentric', 'ot_sampled': 'V2 sampled'}
METHODS = ['V1', 'S', 'G real', 'G zero', 'Source copy', *MODES.values()]
KEYS = ['rare_macro_coverage_recall_25um', 'rare_macro_coverage_recall_50um', 'rare_macro_coverage_recall_100um',
        'rare_micro_recall_100um', 'rare_micro_precision_100um', 'spatial_expression_pcc_100um',
        'spatial_expression_pcc_200um', 'tissue_histogram_tv', 'niche_js_divergence_mean', 'xy_sd_ratio_mean']


def main():
    evaluate, threshold = evaluation_functions()
    assert threshold == .05
    original = {i: ad.read_h5ad(next((ROOT / 'data/merfish_mouse_hypothalamus').rglob(f'merfish_{i}.h5ad'))) for i in range(5)}
    summaries, tables, qcs, maps, displacements, traces = [], [], [], [], [], []
    for fold in (1, 2, 3):
        truth = original[fold]
        observed = [a.copy() for i, a in original.items() if i != fold]
        for i, a in zip([i for i in range(5) if i != fold], observed):
            a.obs['z_coord'] = float(i * 50)
        train = pd.concat([a.obs.cell_class.astype(str) for a in observed]).value_counts()
        rare_classes = set(train[(train / train.sum()) <= .05].index)
        source = PROJECT / f'outputs/g_holdouts_1_2_3/holdout_{fold}/zero'
        base = ad.read_h5ad(source / 'generated.h5ad')
        base_hash = file_sha256(source / 'generated.h5ad')
        base_config = json.loads((source / 'run_config.json').read_text(encoding='utf-8'))
        expected = base_config['experiment_g']['source_schedule_sha256']
        verify_schedule(base.obs, 'z_coord', expected)
        couplings = [sparse.load_npz(RUN / f'holdout_{fold}/shared/coupling_{i}.npz') for i in range(3)]
        slab = abs(base.obs.z_coord - fold * 50) <= 5
        runs = {'G zero': base[slab].copy()}
        for name, path in [('V1', PROJECT / f'outputs/v1_holdouts_1_2_3/holdout_{fold}/generated.h5ad'),
                           ('S', PROJECT / f'outputs/s_holdouts_1_2_3/holdout_{fold}/generated.h5ad'),
                           ('G real', PROJECT / f'outputs/g_holdouts_1_2_3/holdout_{fold}/real/generated.h5ad')]:
            volume = ad.read_h5ad(path)
            runs[name] = volume[abs(volume.obs.z_coord - fold * 50) <= 5].copy()
            del volume
        runs['Source copy'] = parent_copy(runs['G zero'], original)
        for mode, name in MODES.items():
            folder = RUN / f'holdout_{fold}/{mode}'
            config = json.loads((folder / 'run_config.json').read_text(encoding='utf-8'))
            meta = config['experiment_v2_spatial']
            assert meta['source_h5ad_sha256'] == base_hash and not meta['training_performed']
            assert meta['mode'] == mode and meta['target_sampling_seed'] == 42
            assert config['run']['heldout_indices'] == [fold]
            for key in ('data', 'model', 'training', 'slice_z_um', 'seed'):
                assert config[key] == base_config[key]
            volume = ad.read_h5ad(folder / 'generated.h5ad')
            assert volume.shape == base.shape and volume.obs_names.equals(base.obs_names)
            assert volume.var_names.equals(base.var_names)
            verify_schedule(volume.obs, 'z_coord', expected)
            np.testing.assert_array_equal(dense(volume), dense(base))
            for key in base.layers:
                a, b = volume.layers[key], base.layers[key]
                np.testing.assert_array_equal(a.toarray() if sparse.issparse(a) else a, b.toarray() if sparse.issparse(b) else b)
            np.testing.assert_array_equal(volume.obs.cell_class.astype(str), base.obs.cell_class.astype(str))
            np.testing.assert_array_equal(volume.obsm['spatial_before_v2'], base.obsm['spatial'])
            np.testing.assert_array_equal(volume.obsm['spatial_3d'][:, :2], volume.obsm['spatial'])
            np.testing.assert_array_equal(volume.obsm['spatial_3d'][:, 2], volume.obs.z_coord)
            assert np.isfinite(volume.obsm['spatial_3d']).all() and not volume.obs.is_observed.any()
            # Replay from the saved plans; no fitting, new UOT solve or hidden-slice input.
            replay = spatial_transport_volume(base, observed, couplings, mode, 42)
            np.testing.assert_allclose(volume.obsm['spatial'], replay.obsm['spatial'], rtol=0, atol=1e-10)
            np.testing.assert_array_equal(volume.obs.v2_target_cell_index, replay.obs.v2_target_cell_index)
            del replay
            budget = pd.read_csv(folder / 'sampling_budget.csv')
            assert int(budget.allocated_n.sum()) == volume.n_obs
            summary = pd.read_csv(folder / 'spatial_displacement.csv')
            assert int(summary.cells.sum()) == volume.n_obs
            for field in ('supported_cells', 'source_anchor_cells'):
                assert int(volume.uns['v2_spatial'][field]) == int(meta['spatial_metadata'][field])
            pred = volume[slab].copy()
            copied = parent_copy(pred, original)
            distance = np.linalg.norm(pred.obsm['spatial'] - copied.obsm['spatial'], axis=1)
            anchors = pred.obs.generation_mode.to_numpy() == 'source_anchor'
            np.testing.assert_array_equal(pred.obsm['spatial'][anchors], copied.obsm['spatial'][anchors])
            for label in sorted(train.index):
                mask = pred.obs.source_cell_class.astype(str).to_numpy() == label
                displacements.append(dict(heldout_id=fold, run=name, cell_class=label, is_rare_train=label in rare_classes,
                    cells=int(mask.sum()), anchors=int((mask & anchors).sum()),
                    displacement_median_um=float(np.median(distance[mask])) if mask.any() else np.nan,
                    displacement_mean_um=float(distance[mask].mean()) if mask.any() else np.nan))
            qcs.append(dict(heldout_id=fold, run=name, volume_n=volume.n_obs, slab_n=pred.n_obs,
                source_hash_matches=True, schedule_matches=True, frozen_expression_verified=True,
                frozen_types_verified=True, replay_xy_verified=True, finite_xy=True,
                source_anchor_cells=int((volume.obs.generation_mode == 'source_anchor').sum()),
                slab_displacement_median_um=float(np.median(distance)), slab_displacement_mean_um=float(distance.mean()),
                slab_displacement_p95_um=float(np.quantile(distance, .95)),
                spatial_implementation_sha256=meta['implementation_sha256']['spatial_transport'],
                runner_sha256=meta['implementation_sha256']['runner']))
            runs[name] = pred
            del volume
        np.testing.assert_array_equal(runs['Source copy'].obsm['spatial'], runs['V2 source XY'].obsm['spatial'])
        niche = niche_composition(truth, runs)
        local_maps = [dict(heldout_id=fold, **row) for width in (100, 200) for row in spatial_expression(truth, runs, width)]
        maps.extend(local_maps)
        spatial = pd.DataFrame(local_maps)
        local = {}
        for name, pred in runs.items():
            table, metrics = evaluate(truth, pred, train)
            rare = table[table.is_rare_train & (table.truth_n > 0)]
            metrics.update(heldout_id=fold, run=name, **extra_metrics(truth, pred, table), **niche[name])
            for radius in (25, 50, 100):
                metrics[f'rare_micro_recall_{radius}um'] = float(np.average(rare[f'coverage_recall_{radius}um'], weights=rare.truth_n))
                present = rare.predicted_n > 0
                metrics[f'rare_micro_precision_{radius}um'] = float(np.average(rare.loc[present, f'coverage_precision_{radius}um'], weights=rare.loc[present, 'predicted_n']))
            for width in (100, 200):
                frame = spatial[(spatial.run == name) & (spatial.bin_width_um == width)]
                metrics[f'spatial_expression_pcc_{width}um'] = float(frame.spatial_expression_pcc.median())
                metrics[f'spatial_expression_shared_bins_{width}um'] = int(frame.shared_bins.iloc[0])
            local[name] = (table, metrics)
            tables.append(table.assign(run=name, heldout_id=fold))
            for label in rare.cell_class:
                tm = truth.obs.cell_class.astype(str).to_numpy() == label
                pm = pred.obs.cell_class.astype(str).to_numpy() == label
                distance = cKDTree(pred.obsm['spatial'][pm]).query(truth.obsm['spatial'][tm])[0] if pm.any() else np.full(tm.sum(), np.inf)
                for index, dist in zip(np.flatnonzero(tm), distance):
                    traces.append(dict(heldout_id=fold, run=name, cell_class=label,
                        truth_cell_id=truth.obs_names[index], nearest_same_type_um=float(dist),
                        within_25um=bool(dist <= 25), within_50um=bool(dist <= 50), within_100um=bool(dist <= 100)))
        common = set.intersection(*(set(t.loc[t.predicted_n > 0, 'cell_class']) for t, m in local.values()))
        for table, metrics in local.values():
            valid = table[table.is_rare_train & table.cell_class.isin(common) & (table.truth_n > 0)]
            metrics['rare_common_expression_pcc'] = float(valid.mean_log_expression_pcc.mean())
            summaries.append(metrics)
        print('FOLD', fold, json.dumps({name: metrics['rare_macro_coverage_recall_100um'] for name, (t, metrics) in local.items()}), flush=True)
        del base, runs, local, couplings
        gc.collect()
    result, table = pd.DataFrame(summaries), pd.concat(tables, ignore_index=True)
    for name, frame in [('metrics', result), ('cell_type_metrics', table), ('integrity', pd.DataFrame(qcs)),
                        ('spatial_expression', pd.DataFrame(maps)), ('displacement', pd.DataFrame(displacements)),
                        ('rare_cell_distances', pd.DataFrame(traces))]:
        frame.to_csv(DEST / f'{name}.csv', index=False)
    means = result.groupby('run')[KEYS].agg(['mean', 'std'])
    means.columns = ['_'.join(k) for k in means.columns]
    means.to_csv(DEST / 'metrics_mean_std.csv')
    pairs = []
    for new in ('V2 barycentric', 'V2 sampled'):
        for base in ('V2 source XY', 'G zero', 'Source copy', 'S'):
            for fold in (1, 2, 3):
                a = result[(result.run == new) & (result.heldout_id == fold)].iloc[0]
                b = result[(result.run == base) & (result.heldout_id == fold)].iloc[0]
                for key in KEYS:
                    pairs.append(dict(new=new, baseline=base, heldout_id=fold, metric=key,
                                      new_value=a[key], baseline_value=b[key], delta=a[key] - b[key]))
    pd.DataFrame(pairs).to_csv(DEST / 'paired_comparisons.csv', index=False)
    protocol = dict(rare_threshold=.05, inclusive=True, folds=[1, 2, 3], z_slab_half_width_um=5,
        seed=42, methods=METHODS, expression_frozen_relative_to='G zero', spatial_expression_shared_bins='intersection of all eight methods and truth; >=3 cells per dataset/bin',
        replay_from_saved_couplings_verified=True, new_training_performed=False,
        standard_deviation='between held-out slices, not repeated seeds', source_xy_is_position_only_control=True)
    (DEST / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    make_plot(result, table)
    write_report()
    print('MEANS\n' + result.groupby('run')[KEYS].mean().round(6).to_string(), flush=True)


def make_plot(result, table):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none'})
    colors = {'G zero': '#748594', 'V2 source XY': '#BD9653', 'V2 barycentric': '#CB503E', 'V2 sampled': '#3D7E92'}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for ax, key, title, scale in zip(axes, ['rare_macro_coverage_recall_100um', 'rare_micro_recall_100um', 'spatial_expression_pcc_100um'],
            ['Rare macro recall @100 um (%)', 'Rare cell-weighted recall @100 um (%)', 'Spatial expression PCC (100 um bins)'], [100, 100, 1]):
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
    methods = list(colors)
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.8), layout='constrained')
    for ax, fold in zip(axes, (1, 2, 3)):
        pivot = rare[rare.heldout_id == fold].pivot(index='cell_class', columns='run', values='coverage_recall_100um').reindex(index=classes, columns=methods)
        values = pivot.to_numpy() * 100
        ax.imshow(values, vmin=0, vmax=100, cmap='YlGnBu', aspect='auto')
        for r in range(len(classes)):
            for c in range(len(methods)):
                ax.text(c, r, f'{values[r,c]:.0f}', ha='center', va='center', color='white' if values[r,c] > 65 else '#222', fontsize=8)
        ax.set(xticks=range(len(methods)), xticklabels=methods, yticks=range(len(classes)), yticklabels=classes, title=f'Holdout {fold}: recall @100 um (%)')
        ax.tick_params(axis='x', rotation=40)
    for ext in ('png', 'svg'):
        fig.savefig(DEST / f'rare_types.{ext}', dpi=180)
    plt.close(fig)


def write_report():
    s = pd.read_csv(DEST / 'metrics.csv')
    table = pd.read_csv(DEST / 'cell_type_metrics.csv')
    qc = pd.read_csv(DEST / 'integrity.csv')
    assert len(qc) == 9 and len(s) == 24
    assert qc[['frozen_expression_verified', 'frozen_types_verified', 'replay_xy_verified', 'finite_xy']].all().all()
    means = s.groupby('run').mean(numeric_only=True)
    previous = pd.read_csv(PROJECT / 'outputs/g_holdouts_1_2_3/analysis/cell_type_metrics.csv')
    invariant = ['truth_n', 'predicted_n', 'train_fraction', 'coverage_recall_25um', 'coverage_recall_50um',
                 'coverage_recall_100um', 'mean_log_expression_pcc']
    for method in ('V1', 'S', 'G real', 'G zero', 'Source copy'):
        a = table[table.run == method].set_index(['heldout_id', 'cell_class']).sort_index()
        b = previous[previous.run == method].set_index(['heldout_id', 'cell_class']).sort_index()
        assert a.index.equals(b.index)
        np.testing.assert_allclose(a[invariant], b[invariant], rtol=1e-7, atol=1e-7, equal_nan=True)
    for method in MODES.values():
        for fold in (1, 2, 3):
            a = table[(table.run == method) & (table.heldout_id == fold)].set_index('cell_class').sort_index()
            b = table[(table.run == 'G zero') & (table.heldout_id == fold)].set_index('cell_class').sort_index()
            np.testing.assert_array_equal(a[['truth_n', 'predicted_n', 'mean_log_expression_pcc']], b[['truth_n', 'predicted_n', 'mean_log_expression_pcc']])
    overall = []
    for method in METHODS:
        row = means.loc[method]
        overall.append({'方法': method, '宏R25': f'{row.rare_macro_coverage_recall_25um:.2%}',
            '宏R50': f'{row.rare_macro_coverage_recall_50um:.2%}', '宏R100': f'{row.rare_macro_coverage_recall_100um:.2%}',
            '加权R100': f'{row.rare_micro_recall_100um:.2%}', '加权P100': f'{row.rare_micro_precision_100um:.2%}',
            '空间PCC100': f'{row.spatial_expression_pcc_100um:.4f}', '空间PCC200': f'{row.spatial_expression_pcc_200um:.4f}',
            '密度TV↓': f'{row.tissue_histogram_tv:.4f}'})
    folds = []
    for fold in (1, 2, 3):
        row = {'留出': str(fold)}
        for method in ('G zero', *MODES.values()):
            value = s[(s.run == method) & (s.heldout_id == fold)].iloc[0]
            row[method] = f'{value.rare_macro_coverage_recall_100um:.2%}'
        folds.append(row)
    details = []
    rare = table[table.is_rare_train]
    for label in rare.groupby('cell_class').train_fraction.mean().sort_values().index:
        row = {'类别': label}
        for method in ('V2 source XY', 'V2 barycentric', 'V2 sampled'):
            a = rare[(rare.cell_class == label) & (rare.run == method)].sort_values('heldout_id')
            row[method] = ' / '.join(f'{r:.1%}' for r in a.coverage_recall_100um)
        details.append(row)
    # Count gained/lost individual true cells, retaining the same cohort in each paired comparison.
    distances = pd.read_csv(DEST / 'rare_cell_distances.csv')
    gains = []
    for method in ('V2 barycentric', 'V2 sampled'):
        keys = ['heldout_id', 'cell_class', 'truth_cell_id']
        a = distances[distances.run == method].set_index(keys)
        b = distances[distances.run == 'V2 source XY'].set_index(keys)
        a = a.reindex(b.index)
        for radius in (25, 50, 100):
            before, after = b[f'within_{radius}um'], a[f'within_{radius}um']
            frame = pd.DataFrame({'gained': (~before & after).astype(int), 'lost': (before & ~after).astype(int)}, index=b.index)
            for (fold, label), group in frame.groupby(level=[0, 1]):
                gains.append(dict(run=method, heldout_id=fold, cell_class=label, radius_um=radius,
                                  gained_truth=int(group.gained.sum()), lost_truth=int(group.lost.sum()),
                                  net_truth=int(group.gained.sum() - group.lost.sum())))
    gains = pd.DataFrame(gains)
    gains.to_csv(DEST / 'coverage_gains_losses.csv', index=False)
    delta_rows = []
    for method in ('V2 barycentric', 'V2 sampled'):
        a, b = means.loc[method], means.loc['V2 source XY']
        lost = gains[(gains.run == method) & (gains.radius_um == 100)]
        delta_rows.append({'方法': method, '宏R100差(pp)': f'{100 * (a.rare_macro_coverage_recall_100um - b.rare_macro_coverage_recall_100um):+.2f}',
                           '加权R100差(pp)': f'{100 * (a.rare_micro_recall_100um - b.rare_micro_recall_100um):+.2f}',
                           '空间PCC100差': f'{a.spatial_expression_pcc_100um - b.spatial_expression_pcc_100um:+.4f}',
                           '新增覆盖真实细胞': str(int(lost.gained_truth.sum())), '失去覆盖真实细胞': str(int(lost.lost_truth.sum()))})
    motion = []
    for method in MODES.values():
        frame = qc[qc.run == method].sort_values('heldout_id')
        motion.append({'路径': method, '薄层来源位移中位数µm（留出1/2/3）': ' / '.join(f'{v:.2f}' for v in frame.slab_displacement_median_um),
                       '薄层来源位移P95µm（留出1/2/3）': ' / '.join(f'{v:.2f}' for v in frame.slab_displacement_p95_um)})
    report = f'''**V2 空间生成首轮分析**

这轮 V2 没有达到“超过来源复制的稀有细胞空间覆盖”的目标。显式 OT 重心路径和随机目标路径在三个留出切片的 100µm 稀有宏召回、细胞加权召回都低于来源 XY 对照；重心路径在 100µm 空间表达相关上有部分收益，伴随召回下降，不能称为整体重建改善。

**结果与协议核对**

三折各三个空间条件共九组完整，全体积细胞数分别为 114757、112777、113664，隐藏薄层为 5707、5680、5735。核对原 G zero 文件 SHA-256、配置、行顺序、来源/目标深度、表达、表达层、类型及预算。九组坐标均有限；用保存的 coupling 和 seed=42 重放，XY 在 1e-10 绝对容差内一致，随机目标索引一致。因此本轮差异可定位到空间路径，不是表达/类型变化或文件不完整造成。

本轮没有新训练。V2 source XY 是位置对照：使用与两个 OT 版本相同的 G zero 表达和类型，只保留来源 XY。Source copy 则连表达也直接复制来源。两者空间覆盖应相同，空间表达相关可以不同。

稀有类仍为可见四张训练切片合并丰度 ≤5%，各折 11 类，评价 Z±5µm，同类型最近邻半径 25/50/100µm。均值为三折等权，标准差只表示切片差异，不是多随机种子不确定性。

**三折均值**

{md_table(pd.DataFrame(overall))}

R/P 是同类覆盖召回/精确率，单位半径 µm。PCC100/200 使用本次八种方法与真值共同网格重算，每组每格至少三个细胞：100µm 的三折共同格数为 310/299/310，200µm 为 96/93/95。由于共同网格集合随比较方法变化，PCC 绝对值与上一份 G 表不完全相同，应在本表内比较；原 V1/S/G 的逐类召回与表达相关已核对保持不变。

**与冻结表达的来源 XY 对照比较**

{md_table(pd.DataFrame(delta_rows))}

重心方案宏召回从 49.79% 降到 46.59%，随机目标为 46.70%。两个方案还低于 G zero 的 49.85%。新增覆盖的真实细胞不足以抵消丢失的覆盖；重心净少覆盖 65 个稀有真实细胞，随机目标净少 67 个。

{md_table(pd.DataFrame(folds))}

100µm 宏召回的下降在三折方向一致。重心相对来源 XY 的变化分别为 −3.62/−1.49/−4.47 个百分点，随机目标为 −3.55/−1.20/−4.51 个百分点。这比仅比较总体均值更明确。

精确率从 80.37% 升至约 81.67%/81.71%，但加权召回从 76.99% 降至 74.89%/74.82%。本指标不要求一一对应，因此更多预测点接近某个真值，并不保证覆盖到更多真实细胞。

重心的空间 PCC100 从 0.3710 升至 0.3858，分折为正/正/负；PCC200 从 0.5814 略降至 0.5791。密度 TV 从 0.1472 略降至 0.1448，而局部类型组成 JS 从 0.2293 升至 0.2367（变差）。随机目标的 PCC100 基本持平、PCC200 和密度 TV 更差。重心表现是指标取舍，尚非稳定的全面收益。

**位移与类型损失**

{md_table(pd.DataFrame(motion))}

此前 G zero 的位移中位数约 3µm，本轮重心约 9.3–9.6µm，采样约 24µm。位移确实增大了，但覆盖变差。因此不能再仅凭“位移小、接近复制”判断空间分支有缺陷；当前结果说明，对本套配对直接增加运动反而损害定位。

{md_table(pd.DataFrame(details))}

OD Mature 4 的三折平均 100µm 召回从 32.95% 降到 15.53%，两个 OT 方案相同：第一折 4/11 个真实细胞被覆盖变为 1/11，第三折从 5/8 变为 3/8。总计丢失5个覆盖且没有新增。Microglia、OD Immature 1、Pericytes 也下降，说明损失不只来自极小分母。

OD Mature 3 仍为 0/0/0；OD Immature 2 仍为 20%/0/0，空间修改没有解决两类原有问题。无支持端点的 source_anchor 总数在三种路径内均为 1220/1163/1241，没有删除不确定细胞来优化指标。

**原因判断**

可观察到的事实是：沿当前同类局部 UOT 目标移动后，新增的真实覆盖少于丢失的覆盖，且多个类型在重心与采样路径都变差。这支持“当前 OT 目标作为连续细胞运动目标不够可信”的解释。物理邻近、相同类别和表达相似定义了统计候选，并不证明两张不同切片中的细胞具有真实对应关系。

随机目标方案没有修复重心方案的主要损失，因此不能把失败仅归因于重心平均抵消运动。另一方面，重心在局部表达相关上有收益，说明运输关系包含部分分布信息；分布信息有用不等于适合无条件驱动每个来源移动。仅凭本轮不能证明任何形式的 OT 都无效。

**后续修改建议**

1. 保留 source XY / G zero 为固定回退与必报基线，把下一轮目标设为同折同时维持召回并改善空间表达。
2. 优先检查并校准“是否应移动”的门控：结合局部位移方向一致性、双向一致性、候选离散度和多张可见切片的支持。门控阈值只由可见数据的内部留出选择，外层隐藏切片只评价。
3. 以门控后的、小幅共享形变或残差位移为下一步候选，继续保留无支持标记。避免直接把更大的位移作为优化目标。
4. 在召回损失修复前，不把当前随机 OT 路径作为默认生成方式。随机版本目前只有 seed=42；本轮结果也不支持宣称超越来源复制或解决极稀有定位。

**文件**

`metrics.csv`、`metrics_mean_std.csv` 保存每折及均值/切片间标准差；`cell_type_metrics.csv` 保存全部逐类指标；`paired_comparisons.csv` 保存同折差值；`coverage_gains_losses.csv` 和 `rare_cell_distances.csv` 追溯真实细胞覆盖变化；`integrity.csv` 保存冻结及重放检查；`protocol.json` 保存口径；`comparison.png`、`rare_types.png` 另有 SVG。
'''
    (DEST / 'report.md').write_text(report, encoding='utf-8')


if __name__ == '__main__':
    main()
