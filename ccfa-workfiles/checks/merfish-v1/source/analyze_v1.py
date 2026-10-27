"""Read-only V1 holdout-2 audit; reuse the original notebook's metrics."""
from pathlib import Path
import json
import os

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / 'HM_DeepSpital/outputs/v1_holdout_2_run2'
DEST = RUN / 'analysis'
DEST.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(DEST / '.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def dense(a):
    return a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)


def pcc(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else np.nan


def niche_composition(truth, runs, radius=50):
    """Gaussian local class composition at true XY, excluding truth self-neighbors."""
    from scipy.spatial.distance import jensenshannon
    classes = sorted(set(truth.obs.cell_class.astype(str)))
    query = np.asarray(truth.obsm['spatial'])
    compositions = []
    nonempty = []
    for a, exclude_self in [(truth, True), *[(a, False) for a in runs.values()]]:
        xy = np.asarray(a.obsm['spatial'])
        codes = pd.Categorical(a.obs.cell_class.astype(str), categories=classes).codes
        neighborhoods = cKDTree(xy).query_ball_point(query, radius)
        matrix = np.zeros((truth.n_obs, len(classes)))
        for i, ids in enumerate(neighborhoods):
            ids = np.asarray(ids, dtype=int)
            if exclude_self:
                ids = ids[ids != i]
            weights = np.exp(-np.sum((xy[ids]-query[i])**2, axis=1)/(2*25**2))
            matrix[i] = np.bincount(codes[ids], weights=weights, minlength=len(classes))
        present = matrix.sum(1)>0
        nonempty.append(present)
        matrix[present] /= matrix[present].sum(1,keepdims=True)
        compositions.append(matrix)
    shared = np.logical_and.reduce(nonempty)
    result = {}
    for name, matrix, present in zip(runs,compositions[1:],nonempty[1:]):
        js = jensenshannon(compositions[0][shared],matrix[shared],base=2,axis=1)**2
        result[name] = dict(niche_shared_centers=int(shared.sum()),
                            niche_prediction_empty_fraction=float((~present).mean()),
                            niche_js_divergence_mean=float(js.mean()))
    return result


def spatial_expression(truth, runs, width):
    """Use identical spatial bins for all runs, with >=3 cells per dataset."""
    maps = []
    for a in [truth, *runs.values()]:
        xy = np.asarray(a.obsm['spatial'])
        bins = np.floor((xy + 1000) / width).astype(int)
        frame = pd.DataFrame(np.log1p(np.maximum(dense(a), 0)), columns=a.var_names)
        frame['bx'], frame['by'] = bins.T
        groups = frame.groupby(['bx', 'by'])
        means = groups.mean()
        maps.append(means.loc[groups.size() >= 3])
    common = maps[0].index
    for frame in maps[1:]:
        common = common.intersection(frame.index)
    return [dict(run=name, gene=g, bin_width_um=width, shared_bins=len(common),
                 spatial_expression_pcc=pcc(maps[0].loc[common, g], frame.loc[common, g]))
            for name, frame in zip(runs, maps[1:]) for g in truth.var_names]


def main():
    notebook = json.loads((ROOT / 'merfish_leave_one_slice_out.ipynb').read_text(encoding='utf-8'))
    source = next(''.join(c['source']) for c in notebook['cells'] if c.get('id') == 'metric-helpers')
    scope = dict(np=np, pd=pd, ad=ad, sp=sp, cKDTree=cKDTree, LABEL_KEY='cell_class',
                 RARE_TRAIN_FRACTION=.01, COVERAGE_RADII_UM=(25, 50, 100))
    exec(compile(source, 'notebook:metric-helpers', 'exec'), scope)
    evaluate = scope['evaluate_holdout']
    original = {i: ad.read_h5ad(next((ROOT / 'data/merfish_mouse_hypothalamus').rglob(f'merfish_{i}.h5ad')))
                for i in range(5)}
    truth = original[2]
    baseline_dir = ROOT / 'output/merfish_holdout/holdout_2'
    baseline = ad.read_h5ad(baseline_dir / 'predicted_holdout.h5ad')
    saved_truth = ad.read_h5ad(baseline_dir / 'heldout_truth.h5ad')
    assert truth.obs_names.equals(saved_truth.obs_names)
    np.testing.assert_array_equal(dense(truth), dense(saved_truth))
    np.testing.assert_array_equal(truth.obsm['spatial'], saved_truth.obsm['spatial'])
    train = pd.concat([original[i].obs.cell_class.astype(str) for i in (0, 1, 3, 4)]).value_counts()
    volume = ad.read_h5ad(RUN / 'generated.h5ad')
    pred = volume[np.abs(volume.obs.z_coord.to_numpy() - 100) <= 5].copy()
    assert not volume.obs.is_observed.any()
    assert set(pred.obs.source_z) == {50., 150.}
    assert truth.var_names.equals(pred.var_names) and truth.var_names.equals(baseline.var_names)
    np.testing.assert_allclose(volume.obsm['spatial_3d'][:, :2], volume.obsm['spatial'])
    np.testing.assert_allclose(volume.obsm['spatial_3d'][:, 2], volume.obs.z_coord)
    config = json.loads((RUN / 'model/config.json').read_text(encoding='utf-8'))
    xy = np.vstack([original[i].obsm['spatial'] for i in (0, 1, 3, 4)])
    np.testing.assert_allclose([config['spatial_stats'][k] for k in ('x_min', 'y_min', 'x_range', 'y_range')],
                               [*xy.min(0), *(np.ptp(xy, axis=0) + 1e-8)])
    assert config['categories'] == sorted(train.index)
    assert config['gene_names'] == list(truth.var_names)

    # Matched diagnostic control: retain the exact V1 sampled sources without ODE transport.
    parent_x, parent_xy, parent_labels = np.zeros(pred.shape), np.zeros((pred.n_obs, 2)), np.empty(pred.n_obs, object)
    for z in (50., 150.):
        mask = pred.obs.source_z.to_numpy() == z
        src = original[int(z / 50)]
        ids = pred.obs.loc[mask, 'source_cell_index'].to_numpy()
        assert list(src.obs_names[ids]) == list(pred.obs.loc[mask, 'source_cell_id'].astype(str))
        parent_x[mask], parent_xy[mask] = dense(src)[ids], src.obsm['spatial'][ids]
        parent_labels[mask] = src.obs.cell_class.astype(str).to_numpy()[ids]
    np.testing.assert_array_equal(parent_labels, pred.obs.source_cell_class.astype(str))
    parent = ad.AnnData(X=parent_x, obs=pd.DataFrame({'cell_class': parent_labels}, index=pred.obs_names),
                        var=pred.var.copy(), obsm={'spatial': parent_xy})
    runs = {'DeepSpatial': baseline, 'V1': pred, 'V1 source copy': parent}
    summaries, tables, gene_rows, map_rows, program_rows, trace = [], [], [], [], [], []
    tx = np.log1p(np.maximum(dense(truth), 0))
    programs = ['Mbp', 'Opalin', 'Ermn', 'Gjc3']
    program_ids = truth.var_names.get_indexer(programs)
    assert (program_ids >= 0).all()
    for name, a in runs.items():
        table, summary = evaluate(truth, a, train)
        rare = table[table.is_rare_train & (table.truth_n > 0)]
        summary.update(run=name, cell_fraction_total_variation=float(abs(table.truth_fraction - table.predicted_fraction).sum()/2),
                       macro_class_expression_pcc=float(table.mean_log_expression_pcc.mean()),
                       rare_macro_expression_pcc=float(rare.mean_log_expression_pcc.mean()),
                       rare_micro_recall_100um=float(np.average(rare.coverage_recall_100um, weights=rare.truth_n)),
                       rare_true_n=int(rare.truth_n.sum()), rare_predicted_n=int(rare.predicted_n.sum()))
        summaries.append(summary)
        truth_xy, pred_xy = np.asarray(truth.obsm['spatial']), np.asarray(a.obsm['spatial'])
        t_radius, p_radius = np.linalg.norm(truth_xy, axis=1), np.linalg.norm(pred_xy, axis=1)
        t2p = cKDTree(pred_xy).query(truth_xy)[0]
        edges = np.arange(-1000, 1001, 100)
        th = np.histogram2d(*truth_xy.T, bins=(edges,edges))[0] / truth.n_obs
        ph = np.histogram2d(*pred_xy.T, bins=(edges,edges))[0] / a.n_obs
        summary.update(x_sd_um=float(pred_xy[:,0].std()), y_sd_um=float(pred_xy[:,1].std()),
                       tissue_histogram_tv=float(abs(th-ph).sum()/2),
                       fraction_outside_600um=float((p_radius>600).mean()),
                       truth_fraction_outside_600um=float((t_radius>600).mean()),
                       truth_outer_nn_mean_um=float(t2p[t_radius>600].mean()),
                       truth_outer_coverage_50um=float((t2p[t_radius>600]<=50).mean()),
                       truth_inner_coverage_50um=float((t2p[t_radius<=600]<=50).mean()))
        tables.append(table.assign(run=name))
        px = np.log1p(np.maximum(dense(a), 0))
        for j, gene in enumerate(truth.var_names):
            gene_rows.append(dict(run=name, gene=gene, truth_mean_log=float(tx[:, j].mean()),
                                  prediction_mean_log=float(px[:, j].mean()),
                                  truth_detection=float((tx[:, j] > 0).mean()),
                                  prediction_detection=float((px[:, j] > 0).mean()),
                                  log_variance_ratio=float(px[:, j].var()/tx[:, j].var()) if tx[:, j].var() else np.nan,
                                  log_wasserstein=float(wasserstein_distance(tx[:, j], px[:, j]))))
        for label in sorted(train.index):
            tm = truth.obs.cell_class.astype(str).to_numpy() == label
            pm = a.obs.cell_class.astype(str).to_numpy() == label
            ts, ps = tx[tm][:, program_ids].mean(1), px[pm][:, program_ids].mean(1)
            program_rows.append(dict(run=name, cell_class=label, truth_n=len(ts), predicted_n=len(ps),
                                     truth_program_mean=float(ts.mean()), prediction_program_mean=float(ps.mean()) if len(ps) else np.nan,
                                     program_mean_error=float(ps.mean()-ts.mean()) if len(ps) else np.nan,
                                     program_wasserstein=float(wasserstein_distance(ts, ps)) if len(ps) else np.nan))
        print(name, json.dumps(summary, ensure_ascii=False), flush=True)
    for width in (100, 200):
        map_rows.extend(spatial_expression(truth, runs, width))
    niches = niche_composition(truth,runs)
    for summary in summaries:
        summary.update(niches[summary['run']])
    summaries = pd.DataFrame(summaries)
    tables = pd.concat(tables, ignore_index=True)
    genes, maps, programs_table = pd.DataFrame(gene_rows), pd.DataFrame(map_rows), pd.DataFrame(program_rows)
    # Recompute the original metrics, rather than trusting comparison CSVs.
    saved = json.loads((baseline_dir / 'metrics.json').read_text(encoding='utf-8'))
    for key, value in summaries.iloc[0].items():
        if key in saved:
            np.testing.assert_allclose(value, saved[key], rtol=1e-7, atol=1e-7, equal_nan=True)
    pred_labels = pred.obs.cell_class.astype(str).to_numpy()
    delta = np.linalg.norm(pred.obsm['spatial'] - parent_xy, axis=1)
    for label in sorted(train.index):
        m = parent_labels == label
        trace.append(dict(source_class=label, source_n=int(m.sum()), unique_parents=int(pred.obs.loc[m, 'parent_id'].nunique()),
                          class_switches=int((pred_labels[m] != parent_labels[m]).sum()),
                          displacement_mean_um=float(delta[m].mean()) if m.any() else np.nan,
                          displacement_median_um=float(np.median(delta[m])) if m.any() else np.nan,
                          expression_parent_pcc=float(np.nanmean([pcc(x,y) for x,y in zip(np.log1p(parent_x[m]), np.log1p(np.maximum(dense(pred)[m],0)))])) if m.any() else np.nan))
    bins = np.arange(0, 201, 10)
    depth = pd.crosstab(pd.cut(volume.obs.z_coord, bins=bins, right=False), volume.obs.cell_class)
    depth.to_csv(DEST / 'depth_cell_counts.csv')
    log = pd.read_csv(RUN / 'checkpoints/logs/version_0/metrics.csv').dropna(subset=['loss_epoch'])
    assert list(log.epoch.astype(int)) == list(range(100))
    log.to_csv(DEST / 'training_epoch_losses.csv', index=False)
    qc = dict(volume_cells=volume.n_obs, holdout_cells=pred.n_obs, genes=volume.n_vars,
              finite_expression=bool(np.isfinite(dense(volume)).all()), finite_coordinates=bool(np.isfinite(volume.obsm['spatial_3d']).all()),
              negative_expression_entries=int((dense(volume)<0).sum()),
              slab_negative_before_sparsification=int((pred.layers['expression_before_sparsification']<0).sum()),
              slab_total_before_sparsification=int(np.prod(pred.shape)),
              truth_zero_fraction=float((dense(truth)==0).mean()), baseline_zero_fraction=float((dense(baseline)==0).mean()),
              v1_zero_fraction=float((dense(pred)==0).mean()), source_zero_fraction=float((parent_x==0).mean()),
              v1_zero_expression_cells=int((dense(pred)==0).all(1).sum()),
              class_switches=int((pred_labels != parent_labels).sum()), unique_parents_slab=int(pred.obs.parent_id.nunique()),
              volume_class_switches=int((volume.obs.cell_class.astype(str)!=volume.obs.source_cell_class.astype(str)).sum()),
              truth_x_sd_um=float(np.asarray(truth.obsm['spatial'])[:,0].std()),
              truth_y_sd_um=float(np.asarray(truth.obsm['spatial'])[:,1].std()),
              median_displacement_um=float(np.median(delta)), mean_displacement_um=float(delta.mean()),
              split_and_gene_order_verified=True, baseline_metrics_reproduced=True,
              inherited_z_norm_max_error=float(np.abs(volume.obs.z_norm-volume.obs.z_coord/200).max()),
              inherited_slice_ids=list(volume.obs.slice_id.cat.categories.astype(str)))
    (DEST / 'quality_checks.json').write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding='utf-8')
    for name, frame in [('summary',summaries), ('cell_type_metrics',tables), ('gene_metrics',genes),
                        ('spatial_expression_metrics',maps), ('program_metrics',programs_table), ('source_diagnostics',pd.DataFrame(trace))]:
        frame.to_csv(DEST / f'{name}.csv', index=False)
    make_plots(truth, baseline, pred, parent, tables, log)
    write_report(summaries,tables,genes,maps,programs_table,qc,log)
    print('QC',json.dumps(qc),flush=True)
    print('Spatial expression', maps.groupby(['run','bin_width_um']).spatial_expression_pcc.median().to_string(),flush=True)
    print('Gene variance ratios', genes.groupby('run').log_variance_ratio.median().to_string(),flush=True)


def md_table(frame):
    return '\n'.join(['| '+' | '.join(frame.columns)+' |',
                      '| '+' | '.join(['---']*len(frame.columns))+' |',
                      *['| '+' | '.join(map(str,row))+' |' for row in frame.itertuples(index=False,name=None)]])


def write_report(summaries,tables,genes,maps,programs,qc,log):
    s=summaries.set_index('run')
    rows=[]
    for title,key,fmt in [('评价薄层细胞数','predicted_n','.0f'),
                          ('类别比例总变差 ↓','cell_fraction_total_variation','.4f'),
                          ('整体均值表达 PCC ↑','global_mean_log_expression_pcc','.6f'),
                          ('15 类均值表达 PCC 宏平均 ↑','macro_class_expression_pcc','.4f'),
                          ('6 个稀有类表达 PCC 宏平均 ↑','rare_macro_expression_pcc','.4f'),
                          ('稀有类覆盖召回率 @50 µm ↑','rare_macro_coverage_recall_50um','.2%'),
                          ('稀有类覆盖召回率 @100 µm ↑','rare_macro_coverage_recall_100um','.2%'),
                          ('稀有类覆盖精确率 @100 µm ↑','rare_macro_coverage_precision_100um','.2%'),
                          ('稀有类细胞加权召回率 @100 µm ↑','rare_micro_recall_100um','.2%'),
                          ('双向平均最近邻距离 µm ↓','symmetric_mean_nn_um','.2f'),
                          ('XY 密度总变差，100 µm 网格 ↓','tissue_histogram_tv','.4f'),
                          ('50 µm 邻域类型组成 JS 散度 ↓','niche_js_divergence_mean','.4f'),
                          ('真实坐标周围 50 µm 无预测的比例 ↓','niche_prediction_empty_fraction','.2%')]:
        rows.append({'指标':title,**{name:format(s.loc[name,key],fmt) for name in s.index}})
    rare=tables[(tables.run=='V1') & tables.is_rare_train].cell_class.tolist()
    rr=[]
    for label in rare:
        b=tables[(tables.run=='DeepSpatial') & (tables.cell_class==label)].iloc[0]
        v=tables[(tables.run=='V1') & (tables.cell_class==label)].iloc[0]
        rr.append({'稀有类别':label,'真值 / 原版 / V1':f'{v.truth_n} / {b.predicted_n} / {v.predicted_n}',
                   '100 µm 召回率：原版 → V1':f'{b.coverage_recall_100um:.1%} → {v.coverage_recall_100um:.1%}',
                   '表达 PCC：原版 → V1':f'{b.mean_log_expression_pcc:.3f} → {v.mean_log_expression_pcc:.3f}'})
    mr=maps.groupby(['run','bin_width_um']).spatial_expression_pcc.median()
    prog=programs[programs.cell_class.str.startswith('OD')].copy()
    prows=[]
    for label in prog.cell_class.unique():
        q=prog[prog.cell_class==label].set_index('run')
        prows.append({'类别':label,'真实均值':f'{q.iloc[0].truth_program_mean:.4f}',
                      **{name:f'{q.loc[name,"prediction_program_mean"]:.4f}' for name in s.index}})
    report=f'''# V1 留出切片 2 效果分析

分析日期：2026-10-27。输入：`../generated.h5ad`；结果未被修改。

## 判断

V1 完成了完整训练与重建，保留了 15 个细胞大类，类别均值表达有小幅改善；稀有类空间恢复和整体组织几何没有改善，中心收缩比现有原版运行更明显。当前证据不支持“V1 已解决稀有细胞空间缺失”或“生态位条件已提高空间保真度”。

## 评价对象与核验

- 总输出 112,777 个细胞、155 个基因，是 z≈0–200 µm 的连续重建体积，并非单张留出切片。全部 `is_observed=False`。
- 固定评价 z=100±5 µm：V1 5,632 个预测，对应 `merfish_2.h5ad` 的 5,926 个真实细胞；现有原版相同薄层 5,630 个预测。两者生成数均约为真值的 95%，彼此近乎相同。
- 训练可见切片均为 0、1、3、4，留出 2。V1 的归一化参数、类别、基因顺序与本地可见数据一致；原版保存的隐藏真值表达、XY、细胞顺序与原始文件一致。原版既有整体指标从 H5AD 复算通过 1e-7 容差检查。
- 两次运行均记录 seed=42、100 epochs、batch=512、50,000 基础配对、hidden=256、depth=6、8 heads；UOT alpha=0.5/reg=0.8/tau=0.05；lambda_g=0.1/lambda_c=10；ODE 100 时间点、dopri5、rtol=atol=1e-5、thickness=10 µm，使用 EMA 生成。
- V1 同时启用 19 维生态位条件、50 µm 半径/25 µm sigma、四基因代理程序、0.2 同类配对混合、分层来源抽样、双向训练。因此本比较无法分离各模块的贡献。
- 原版生成夹住隐藏层的区间，V1 生成完整体积后取相同薄层；前序生成会消耗随机数。同一 seed 不意味着逐细胞来源或轨迹相同。本报告是同一留出位置的已有运行诊断，尚无多种子或其他留出位置的 V1 结果。

## 主指标

{md_table(pd.DataFrame(rows))}

“V1 source copy”是诊断对照：固定 V1 薄层的每一个来源、数量及标签，直接使用对应来源的 XY 和表达，不执行 ODE 运输；未使用隐藏真值构造。它只能检验本次运输相对保留来源状态的效果，不能替代完整三维重建模型或训练消融。

整体 PCC 是跨 155 个基因的平均 log1p 表达向量相关性；类别 PCC 在每个类别内先取均值。它们均不表示逐细胞预测准确率。空间覆盖允许多个真实细胞共享一个预测邻居，不能解释为细胞识别率。

## 稀有类：数量保留，位置恢复退化

稀有类按四张训练切片合计占比 <1% 定义，名单固定为以下六类；真值共 103 个，对应原版 119 个、V1 124 个。

{md_table(pd.DataFrame(rr))}

V1 的稀有类宏平均 100 µm 召回率下降 4.38 个百分点，约相对下降 23.1%；细胞数加权召回率也由 24.27% 降至 22.33%，所以退化并非单纯由最小类别等权放大。

- OD Mature 3：V1 生成 5 个，真实只有 2 个，真实位置周围 100 µm 覆盖仍为 0；真实到预测同类最近邻均值约 691 µm，原版约 607 µm。
- OD Immature 2：V1 生成 6 个，真实 5 个，但 100 µm 召回从 20% 降为 0；同类最近邻均值从约 319 增至 651 µm，表达 PCC 也由 0.710 降至 0.641。
- Pericytes：100 µm 覆盖从 8% 升至 24%，存在局部收益；同时 46 个预测明显多于 25 个真值，覆盖精确率仅 13.0%。
- 关键极稀有类别只有 2–5 个真值，单个细胞即可大幅改变百分比；这些观察应在更多留出位置复核。

![稀有类指标](rare_cell_comparison.png)

![稀有类空间分布](rare_cell_spatial.png)

空间图蓝色空心圈是真实细胞，蓝色圆盘半径 100 µm，橙色叉为预测同类细胞，灰点为全部真实细胞。

## 主要失败模式：空间分布向中心收缩

真实切片 X/Y 标准差为 501.74/499.53 µm；原版为 373.82/371.95 µm；V1 为 358.31/343.94 µm，分别比真值小约 28.6%/31.1%。保留 V1 来源坐标的对照为 505.06/487.44 µm。

距离固定 XY 原点 >600 µm 的细胞占比：真实 60.58%，原版 31.60%，V1 25.60%，保留来源对照 59.11%。该半径只是本次组织边缘的描述性划分，不是生物学边界。

100×100 µm 网格密度总变差由原版 0.3200 增至 V1 0.3799；保留来源对照仅 0.1492。V1 相对来源的位移中位数 239.36 µm、均值 262.09 µm。正常重建可以包含位移，但本次位移伴随隐藏层组织范围收缩和密度偏差增大。

![组织密度](tissue_density.png)

四图使用同一空间范围、网格和色标，按各组总细胞数归一化。V1 与原版均将中心区域填得更密，组织边缘覆盖不足。原版已有该问题，V1 进一步加重。

本次所有组的共同有效真实坐标上，50 µm 半径、sigma=25 µm 的高斯邻域类型组成 JS 散度（以 2 为底，越低越好）也显示 V1 更差，具体数值见主表。真实邻域排除自身；JS 只在所有组邻域均非空的同一批坐标计算，另列每组空邻域比例，避免把无覆盖位置隐去。因此目前没有直接的局部类型生态位保真度收益。

这些结果优先指向空间配对与运输过程。熵正则配对过于分散、较长距离端点配对、速度场平均化均是待检验原因；现有结果不能判定具体根因。

## 表达：全局保持较好，局部空间表达仍弱

总体平均表达 PCC 从 0.996619 变为 0.996750，绝对变化约 0.00013；类别宏平均从 0.9472 升为 0.9530，稀有类宏平均从 0.8800 升为 0.8926。平均层面有小幅改善，个别稀有类仍明显失真。

以同一 XY 原点和所有组共同有效网格比较每个基因的网格均值 log1p 表达，要求每组每格至少 3 个细胞；跨 155 基因的空间表达 PCC 中位数如下：

| 网格宽度 | 原版 | V1 | 保留 V1 来源 |
| --- | --- | --- | --- |
| 100 µm | {mr['DeepSpatial',100]:.4f} | {mr['V1',100]:.4f} | {mr['V1 source copy',100]:.4f} |
| 200 µm | {mr['DeepSpatial',200]:.4f} | {mr['V1',200]:.4f} | {mr['V1 source copy',200]:.4f} |

因此，高全局均值 PCC 掩盖了局部分子空间分布偏差。空间表达统计仅覆盖共同有效网格，边缘缺失另外由密度、覆盖率与空邻域指标刻画。

表达 log1p 方差的预测/真实比值，跨基因中位数：原版 0.8603、V1 0.8562、保留来源 1.0265；两种运输结果都有一定表达分布收窄。

### 四基因代理程序

为便于解释，下面使用每个细胞 `mean(log1p(Mbp/Opalin/Ermn/Gjc3))` 再在类内平均。它是描述性面板得分，区别于模型内部标准化并聚合邻居的条件，不能直接当作真实功能活性或完整生态位重建指标。

{md_table(pd.DataFrame(prows))}

OD Mature 3 的四基因均值非常接近真值（2.3916 对 2.3886），但空间覆盖为 0。OD Immature 2 该得分偏差减小，155 基因类均值相关性却下降。OD Mature 1/2 的得分偏差较原版增大，故该代理程序也没有一致改善。

保留来源对照在多个类别已经接近 V1，提示不少分子特征由来源选择继承；本次数据无法把这些收益单独归因于生态位条件。

## 训练与输出注意点

- V1 日志完整覆盖 epoch 0–99，总训练损失从 {log.iloc[0].loss_epoch:.4f} 降到 {log.iloc[-1].loss_epoch:.4f}，末轮坐标/基因/类别分量分别约 0.0697/12.0338/0.00056。缺少验证损失；训练目标和抽样分布发生变化，不能凭与原版的损失高低判断优劣。
- 薄层没有负表达或全零细胞；全体积仍有 2 个微小负表达元素，约 -0.00235/-0.00866，位于 z≈3.53/4.61 µm。指标沿用原协议先截非负再 log1p，原文件保持原样。
- 薄层稀疏化前有 105,023/872,960≈12.03% 负元素，最终 top-k 稀疏化隐藏了这些负值；零值比例 57.3223% 与匹配来源完全相同，主要受 parent 检出基因预算控制，不是额外学习到的稀疏性证据。
- 全体积 112,777 个生成细胞的 `cell_class` 均与 `source_cell_class` 相同；类别存在性不能解释成独立预测到隐藏层身份。
- `neuron_class/domain/Region/annotation/slice_id/z_norm` 随来源复制。其中 `z_norm` 与实际生成深度不一致（最大偏差约 0.498）；空间评价应使用 `z_coord` 或 `spatial_3d`。继承的细分注释不能当作隐藏组织预测真值。
- `uns['hm_reconstruction']` 的细胞数/parent 数继承首段（generated_cells=27,612），不是全体积汇总；总数应以 `n_obs=112,777` 为准。

![训练与均值表达](training_expression.png)

## 下一步优先级

1. 先定位空间收缩：在固定来源与目标深度下记录正反向轨迹、端点位移及真实组织范围；检查 UOT 的配对距离、熵与空间代价归一化，判断收缩是配对、速度学习还是生成过程造成。
2. 使用相同分层来源、双向训练和生成密度，比较无生态位/zero/同类同切片 shuffle/real 条件，隔离生态位条件本身的贡献。其他模块分别消融，避免同时改变多个因素。
3. 固定现有评价口径，补齐留出 1/3 和多个随机种子，并同时报告密度总变差、同类空间覆盖和空间表达。极稀有类优先看逐点距离与空间图。
4. 如果继续 V2 表达修正，应同时保留空间质量门槛。表达修正本身不会恢复已经放错位置的细胞；优先处理空间收缩，再评价程序先验的额外收益。
5. 后续改输出逻辑时修正负值与来源元数据标记、合并段元数据；本次分析没有修改模型、训练代码或输入 H5AD。

## 可复算文件

- `summary.csv`：三个运行的全部整体指标。
- `cell_type_metrics.csv`：15 类的表达、计数和 25/50/100 µm 覆盖。
- `spatial_expression_metrics.csv`：每个基因在固定共同网格上的空间相关性。
- `gene_metrics.csv`、`program_metrics.csv`：表达检测率、方差、分布距离和四基因得分。
- `source_diagnostics.csv`、`depth_cell_counts.csv`：来源唯一性、运输位移及各深度类型计数。
- `quality_checks.json`、`training_epoch_losses.csv`：数据核验和训练曲线原始数值。
- 分析源码：`../../../../ccfa-workfiles/checks/merfish-v1/source/analyze_v1.py`。
- 复算命令：在仓库根目录使用已安装 anndata/numpy/pandas/scipy/matplotlib 的 Python 执行上述脚本。
'''
    assert '\ufffd' not in report
    path=DEST/'report.md'
    path.write_text(report,encoding='utf-8')
    assert path.read_text(encoding='utf-8')==report


def make_plots(truth, baseline, pred, parent, tables, log):
    plt.rcParams.update({'font.family':'Times New Roman','font.size':11, 'svg.fonttype':'none'})
    rare = tables[(tables.run=='V1') & tables.is_rare_train].cell_class.tolist()
    fig, axes = plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    y = np.arange(len(rare))
    for offset,(name,color) in enumerate([('DeepSpatial','#357ba8'),('V1','#d15e32'),('V1 source copy','#69966b')]):
        sub = tables[tables.run==name].set_index('cell_class').loc[rare]
        axes[0].barh(y+(offset-1)*.24,100*sub.coverage_recall_100um,height=.23,label=name,color=color)
        if name != 'V1 source copy':
            axes[1].plot(sub.predicted_n,y,'o',label=name,color=color)
    for ax in axes:
        ax.set_yticks(y,rare)
        ax.invert_yaxis()
        ax.spines[['right','top']].set_visible(False)
    axes[0].set_xlim(0,100)
    axes[0].set_xlabel('Same-type recall within 100 µm (%)')
    axes[0].legend(fontsize=10,loc='lower right')
    true_counts=tables[tables.run=='V1'].set_index('cell_class').loc[rare].truth_n
    axes[1].plot(true_counts,y,'kx',markersize=9,label='Truth')
    axes[1].set_xlabel('Cell count in z = 100 ± 5 µm')
    axes[1].legend(fontsize=10)
    for ext in ('png','svg'): fig.savefig(DEST/f'rare_cell_comparison.{ext}',dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2,2,figsize=(9,8),layout='constrained')
    edges = np.arange(-1000,1001,100)
    histograms=[]
    for name,a in [('Truth',truth),('DeepSpatial',baseline),('V1',pred),('V1 source copy',parent)]:
        histograms.append((name,np.histogram2d(*np.asarray(a.obsm['spatial']).T,bins=(edges,edges))[0]/a.n_obs*100))
    maximum=max(h.max() for _,h in histograms)
    for ax,(name,h) in zip(axes.flat,histograms):
        im=ax.imshow(h.T,origin='lower',extent=(-1000,1000,-1000,1000),vmin=0,vmax=maximum,cmap='Blues',interpolation='nearest')
        ax.set(title=name,xlabel='X (µm)',ylabel='Y (µm)')
    fig.colorbar(im,ax=axes,label='Cells in each 100 × 100 µm bin (%)',shrink=.75)
    for ext in ('png','svg'): fig.savefig(DEST/f'tissue_density.{ext}',dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2,3,figsize=(12,8),layout='constrained')
    for row,label in enumerate(['OD Mature 3','OD Immature 2']):
        for col,(name,a) in enumerate([('Truth',truth),('DeepSpatial',baseline),('V1',pred)]):
            ax=axes[row,col]
            xy=np.asarray(a.obsm['spatial'])
            txy=np.asarray(truth.obsm['spatial'])
            tm=truth.obs.cell_class.astype(str).to_numpy()==label
            pm=a.obs.cell_class.astype(str).to_numpy()==label
            ax.scatter(*txy.T,s=1,color='#c8cdd2',alpha=.4,rasterized=True)
            for point in txy[tm]: ax.add_patch(plt.Circle(point,100,color='#357ba8',alpha=.12,lw=0))
            ax.scatter(*txy[tm].T,s=65,facecolors='none',edgecolors='#357ba8',lw=1.5)
            if col: ax.scatter(*xy[pm].T,s=55,marker='x',color='#d15e32',lw=1.5)
            ax.set_title(f'{name}: {label} (n={pm.sum()})')
            ax.set(xlim=(-1000,1000),ylim=(-1000,1000),xlabel='X (µm)',ylabel='Y (µm)',aspect='equal')
            ax.spines[['right','top']].set_visible(False)
    for ext in ('png','svg'): fig.savefig(DEST/f'rare_cell_spatial.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    axes[0].plot(log.epoch,log.loss_epoch,color='#d15e32',label='V1')
    old=pd.read_csv(ROOT/'output/merfish_holdout/holdout_2/logs/version_0/metrics.csv').dropna(subset=['loss_epoch'])
    axes[0].plot(old.epoch,old.loss_epoch,color='#357ba8',label='DeepSpatial')
    axes[0].set(xlabel='Epoch',ylabel='Training loss')
    axes[0].legend()
    axes[0].set_title('Different training distributions; no validation loss')
    tx=np.log1p(np.maximum(dense(truth),0)).mean(0)
    for name,a,color in [('DeepSpatial',baseline,'#357ba8'),('V1',pred,'#d15e32')]:
        px=np.log1p(np.maximum(dense(a),0)).mean(0)
        axes[1].scatter(tx,px,s=13,color=color,alpha=.65,label=f'{name}: r={pcc(tx,px):.4f}')
    axes[1].plot([0,2],[0,2],color='gray',lw=.8)
    axes[1].set(xlabel='Truth mean log1p expression',ylabel='Generated mean log1p expression')
    axes[1].legend()
    for ax in axes: ax.spines[['right','top']].set_visible(False)
    fig.savefig(DEST/'training_expression.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    main()
