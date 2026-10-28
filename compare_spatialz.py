"""Compare saved SpatialZ slice 0.09 with DeepSpatial holdout_1 using one evaluator.

python compare_spatialz.py --spatialz-root ../SpatialZ --include-variants

No training, new registration, expression normalization, or original-file writes.
Main expression metrics compare 155-gene mean profiles, not unpaired matrix rows.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parent
DEFAULT_DS = ROOT / 'output/merfish_holdout/20261018_021947_653667'


def vector(node):
    if isinstance(node, h5py.Group):
        if node.attrs.get('encoding-type') != 'categorical':
            raise ValueError(f'Unsupported obs encoding: {node.name}')
        categories, codes = vector(node['categories']), node['codes'][...]
        if np.any(codes < 0) or np.any(codes >= len(categories)):
            raise ValueError(f'Missing or invalid labels: {node.name}')
        return categories[codes]
    return node.asstr()[...] if h5py.check_string_dtype(node.dtype) else node[...]


def read_h5ad(path):
    with h5py.File(path, 'r') as f:
        node = f['X']
        if isinstance(node, h5py.Dataset):
            x = node[...]
        else:
            encoding = node.attrs.get('encoding-type')
            if encoding not in ('csr_matrix', 'csc_matrix'):
                raise ValueError(f'Unsupported expression encoding in {path}: {encoding}')
            constructor = sparse.csr_matrix if encoding == 'csr_matrix' else sparse.csc_matrix
            x = constructor((node['data'][...], node['indices'][...], node['indptr'][...]),
                            shape=node.attrs['shape']).toarray()
        result = dict(x=np.asarray(x, dtype=float), xy=np.asarray(f['obsm/spatial'], dtype=float),
                      labels=vector(f['obs/cell_class']).astype(str),
                      genes=vector(f['var'][f['var'].attrs['_index']]).astype(str),
                      ids=vector(f['obs'][f['obs'].attrs['_index']]).astype(str),
                      obs={k: vector(f['obs'][k]) for k in ('data_type', 'z_coord') if k in f['obs']})
        result['uns_has_log1p'] = 'log1p' in f['uns']
    n, g = result['x'].shape
    if n == 0 or g < 2 or result['xy'].shape != (n, 2) or result['labels'].shape != (n,):
        raise ValueError(f'Invalid or empty expression/coordinate/label shapes: {path}')
    if len(np.unique(result['genes'])) != g:
        raise ValueError(f'Duplicate gene names: {path}')
    if not np.isfinite(result['x']).all() or not np.isfinite(result['xy']).all():
        raise ValueError(f'Nonfinite expression or XY coordinates: {path}')
    return result


def align_genes(truth, prediction):
    if set(truth['genes']) != set(prediction['genes']):
        raise ValueError('Truth/prediction gene sets differ; refusing a silent gene intersection.')
    lookup = {gene: i for i, gene in enumerate(prediction['genes'])}
    order = [lookup[gene] for gene in truth['genes']]
    return prediction['x'][:, order]


def expression_metrics(truth_values, predicted_values):
    a, b = np.asarray(truth_values).ravel(), np.asarray(predicted_values).ravel()
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Expression metric vectors must have equal shape and finite values.')
    return dict(MSE=float(np.mean((a-b)**2)), MAE=float(np.mean(np.abs(a-b))),
                PCC=float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else np.nan)


def coverage(truth, prediction, train_counts, rare_threshold=.01):
    rows = []
    total_train = sum(train_counts.values())
    if total_train <= 0:
        raise ValueError('Training counts must have positive total.')
    for cat in sorted(set(train_counts) | set(truth['labels']) | set(prediction['labels'])):
        tm, pm = truth['labels'] == cat, prediction['labels'] == cat
        nt, npred = int(tm.sum()), int(pm.sum())
        distances = cKDTree(prediction['xy'][pm]).query(truth['xy'][tm])[0] if npred else np.full(nt, np.inf)
        row = dict(cell_class=cat, train_n=int(train_counts.get(cat, 0)),
                   train_fraction=train_counts.get(cat, 0)/total_train,
                   is_rare=0 < train_counts.get(cat, 0)/total_train < rare_threshold,
                   truth_n=nt, predicted_n=npred, missing_in_prediction=nt > 0 and npred == 0)
        for radius in (50, 100):
            row[f'recall_{radius}um'] = float(np.mean(distances <= radius)) if nt else np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    rare = table[table.is_rare & (table.truth_n > 0)]
    if rare.empty:
        raise ValueError('No training-defined rare classes occur in truth.')
    return table, dict(rare_classes=len(rare), rare_missing_classes=int(rare.missing_in_prediction.sum()),
                      rare_recall_50um=float(rare.recall_50um.mean()),
                      rare_recall_100um=float(rare.recall_100um.mean()))


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(name, prediction, truth, train_counts):
    if 'data_type' in prediction['obs'] and np.any(prediction['obs']['data_type'] != 'synthetic'):
        raise ValueError(f'{name} includes observed cells; supply generated-only held-out predictions.')
    px = align_genes(truth, prediction)
    tx = truth['x']
    perclass, recalls = coverage(truth, prediction, train_counts)
    # Negative entries are clipped for log1p only, exactly as in the previous DS evaluator.
    tl, pl = np.log1p(np.maximum(tx, 0)), np.log1p(np.maximum(px, 0))
    common = dict(method=name, truth_n=len(tx), predicted_n=len(px), n_genes=tx.shape[1], **recalls)
    rows = []
    for scale, t, p in [('provided_X', tx, px), ('log1p', tl, pl)]:
        rows.append(dict(**common, scale=scale, scope='gene_mean_profile',
                         **expression_metrics(t.mean(0), p.mean(0))))
    # Supplementary spatial diagnostic: XY NN association does not assert cell identity.
    distances, neighbors = cKDTree(prediction['xy']).query(truth['xy'])
    local = dict(method=name, scope='truth_to_prediction_XY_nearest_log1p',
                 mean_nn_um=float(distances.mean()), unique_prediction_neighbors=int(len(np.unique(neighbors))),
                 **expression_metrics(tl, pl[neighbors]))
    profiles = pd.DataFrame(dict(gene=truth['genes'],truth_mean_X=tx.mean(0),
                                 predicted_mean_X=px.mean(0),truth_mean_log1p=tl.mean(0),
                                 predicted_mean_log1p=pl.mean(0)))
    qc = dict(negative_entries=int((px < 0).sum()), min_X=float(px.min()), max_X=float(px.max()),
              zero_expression_cells=int(np.count_nonzero(np.all(px == 0, axis=1))),
              uns_has_log1p=prediction['uns_has_log1p'])
    return rows, perclass.assign(method=name), local, profiles, qc


def markdown_table(frame):
    def display(value):
        return f'{value:.6f}' if isinstance(value, (float, np.floating)) else str(value)
    return '\n'.join(['| '+' | '.join(frame.columns)+' |',
                      '| '+' | '.join(['---']*len(frame.columns))+' |',
                      *['| '+' | '.join(display(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]])


def plot_comparison(main, classes, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    raw = main[main.scale == 'provided_X']
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.4), layout='constrained')
    for ax, key, title in zip(axes, ['MSE','MAE','PCC','rare_recall_50um','rare_recall_100um'],
                             ['Mean-profile MSE (X)','Mean-profile MAE (X)','Mean-profile PCC (X)',
                              'Rare coverage recall 50 um','Rare coverage recall 100 um']):
        vals = raw[key].to_numpy()
        ax.bar(range(len(raw)), vals, color=['#327b9d','#d88b43'])
        ax.set_xticks(range(len(raw)), raw.method, rotation=20, ha='right')
        ax.set_title(title, fontsize=10)
        ax.spines[['top','right']].set_visible(False)
        ax.set_ylim(0, 1.16 if key == 'PCC' else max(vals)*1.30)
        if key.startswith('rare_'):
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        for i, value in enumerate(vals):
            label = f'{value:.2%}' if key.startswith('rare_') else f'{value:.4f}'
            ax.text(i, value+ax.get_ylim()[1]*.025, label, ha='center', fontsize=9)
    fig.suptitle('Same held-out slice 0.09 | all provided genes | native output counts', fontsize=12)
    fig.savefig(out/'comparison.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    rare = classes[classes.is_rare & (classes.truth_n > 0)]
    cats = rare.cell_class.unique()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5),layout='constrained')
    for ax, radius in zip(axes, (50,100)):
        for j, (name, color) in enumerate(zip(('DeepSpatial','SpatialZ'),('#327b9d','#d88b43'))):
            part = rare[rare.method==name].set_index('cell_class').reindex(cats)
            ax.barh(np.arange(len(cats))+(j-.5)*.32,part[f'recall_{radius}um'],height=.32,color=color,label=name)
        ax.set_yticks(np.arange(len(cats)),cats)
        ax.invert_yaxis();ax.set_xlim(0,1);ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title(f'Same-class coverage recall within {radius} um');ax.legend(frameon=False)
    fig.savefig(out/'rare_recall_by_class.png',dpi=200,bbox_inches='tight')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--spatialz-root',type=Path,default=ROOT.parent/'SpatialZ')
    parser.add_argument('--deepspatial-run',type=Path,default=DEFAULT_DS)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'output/spatialz_comparison_20261018')
    parser.add_argument('--include-variants',action='store_true')
    args=parser.parse_args()
    sz, ds, out = args.spatialz_root.resolve(), args.deepspatial_run.resolve(), args.output_dir.resolve()
    sources = {'truth':ds/'holdout_1/heldout_truth.h5ad',
               'SpatialZ_truth':sz/'data/merfish/merfish_9_paste.h5ad',
               'training_counts':ds/'holdout_1/train_cell_counts.csv',
               'SpatialZ_run':sz/'output/merfish_holdout_009/run.json'}
    run=json.loads(sources['SpatialZ_run'].read_text())
    if (run['status']!='complete' or Path(run['held_out']).name!='merfish_9_paste.h5ad' or
        [Path(p).name for p in run['inputs']]!=['merfish_4_paste.h5ad','merfish_14_paste.h5ad']):
        raise ValueError('SpatialZ run does not document the required completed 4/14 -> 9 holdout.')
    for filename, expected in zip([*run['inputs'],run['held_out']],
                                  [*run['input_sha256'],run['held_out_sha256']]):
        path=sz/'data/merfish'/Path(filename).name
        if fingerprint(path)!=expected:
            raise ValueError(f'SpatialZ input file no longer matches its run hash: {path}')
        sources[f'input_{path.stem}']=path
    truth, sz_truth=read_h5ad(sources['truth']), read_h5ad(sources['SpatialZ_truth'])
    identity={key:bool(np.array_equal(truth[key],sz_truth[key])) for key in ('x','xy','labels','genes','ids')}
    if not all(identity.values()):
        raise ValueError(f'Saved truth differs between methods: {identity}')
    if truth['uns_has_log1p'] or sz_truth['uns_has_log1p']:
        raise ValueError('Truth declares prior log1p; choose an explicit shared scale before comparison.')
    counts=pd.read_csv(sources['training_counts']).set_index('cell_class').train_n.to_dict()
    pair_counts={}
    for filename in run['inputs']:
        with h5py.File(sz/'data/merfish'/Path(filename).name) as f:
            cats, nums=np.unique(vector(f['obs/cell_class']),return_counts=True)
            for cat,n in zip(cats,nums):
                pair_counts[cat]=pair_counts.get(cat,0)+int(n)
    rare_classes=sorted(k for k,n in counts.items() if 0<n/sum(counts.values())<.01)
    pair_rare=sorted(k for k,n in pair_counts.items() if 0<n/sum(pair_counts.values())<.01)
    if rare_classes!=pair_rare:
        raise ValueError('Two-input and four-input rare sets differ; explicitly choose the common benchmark set.')
    predictions={'DeepSpatial':ds/'holdout_1/predicted_holdout.h5ad',
                 'SpatialZ':sz/'output/merfish_holdout_009/generated_009.h5ad'}
    if args.include_variants:
        for seed in (42,43,44,45):
            for arm in ('gene','cell'):
                path=sz/f'output/step01_joint_sampling_multiseed/seed{seed}/{arm}/generated_009.h5ad'
                if not path.is_file():
                    raise FileNotFoundError(f'Requested variant is missing: {path}')
                predictions[f'SpatialZ_joint_{arm}_seed{seed}']=path
    results, class_tables, local_rows, profiles, qc = [], [], [], {}, {}
    ds_run=json.loads((ds/'holdout_1/run_parameters.json').read_text())
    if ds_run['heldout_id']!=1 or ds_run['train_ids']!=[0,2,3,4]:
        raise ValueError('Unexpected DeepSpatial holdout configuration.')
    base_spatialz=None
    for name,path in predictions.items():
        prediction=read_h5ad(path)
        if name=='SpatialZ':
            base_spatialz=prediction
        elif name.startswith('SpatialZ_joint_'):
            if not all(np.array_equal(prediction[key],base_spatialz[key]) for key in ('xy','labels','genes')):
                raise ValueError(f'{name} does not share the documented fixed SpatialZ geometry/labels.')
        if prediction['uns_has_log1p']:
            raise ValueError(f'{name} declares log1p; refusing an unverified mixed expression scale.')
        if name=='DeepSpatial':
            z=prediction['obs'].get('z_coord')
            center=ds_run['heldout_id']*ds_run['slice_gap_um']
            if z is None or not np.all(np.abs(z-center)<=ds_run['slab_half_width_um']+1e-6):
                raise ValueError('DeepSpatial prediction is not the documented held-out thin slab.')
        rows, table, local, profile, quality=evaluate(name,prediction,truth,counts)
        results.extend(rows);class_tables.append(table);local_rows.append(local);profiles[name]=profile;qc[name]=quality
        sources[name]=path
    out.mkdir(parents=True,exist_ok=True)
    all_metrics=pd.DataFrame(results)
    main_table=all_metrics[all_metrics.method.isin(['DeepSpatial','SpatialZ'])]
    all_classes=pd.concat(class_tables,ignore_index=True)
    main_table.to_csv(out/'comparison.csv',index=False)
    all_metrics[~all_metrics.method.isin(['DeepSpatial','SpatialZ'])].to_csv(out/'variants.csv',index=False)
    all_classes.to_csv(out/'cell_type_metrics.csv',index=False)
    pd.DataFrame(local_rows).to_csv(out/'spatial_nn_expression.csv',index=False)
    for name,profile in profiles.items():
        profile.to_csv(out/f'{name}_gene_profiles.csv',index=False)
    sources['DeepSpatial_run']=ds/'holdout_1/run_parameters.json'
    sources['SpatialZ_reference_metrics']=sz/'output/merfish_holdout_009/metrics.json'
    sources['DeepSpatial_reference_summary']=ds/'fold_summary.csv'
    sources['evaluator']=Path(__file__).resolve()
    audit=dict(status='ANALYZED',truth_identity=identity,scope='one shared held-out slice; no training rerun',
               sources={key:dict(path=str(path),sha256=fingerprint(path)) for key,path in sources.items()},
               training_rare_classes=rare_classes, two_input_rare_classes=pair_rare,
               qc=qc,SpatialZ_generation_parameters=run['parameters'],DeepSpatial_run=ds_run)
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    cols=['method','MSE','MAE','PCC','rare_recall_50um','rare_recall_100um','predicted_n']
    raw=main_table[main_table.scale=='provided_X'][cols]
    log=main_table[main_table.scale=='log1p'][cols]
    rare=all_classes[all_classes.method.isin(['DeepSpatial','SpatialZ']) & all_classes.is_rare]
    table_rare=rare.pivot(index='cell_class',columns='method',values=['predicted_n','recall_50um','recall_100um'])
    table_rare.columns=['_'.join(col) for col in table_rare.columns]
    table_rare.to_csv(out/'rare_recall_comparison.csv')
    original_sz=json.loads((sz/'output/merfish_holdout_009/metrics.json').read_text())['comparisons']['generated_0.09']
    got=raw[raw.method=='SpatialZ'].iloc[0]
    if not (np.isclose(got.PCC,original_sz['gene_mean_pearson']) and
            np.isclose(got.MAE,original_sz['gene_mean_mae']) and
            np.isclose(got.MSE,original_sz['gene_mean_rmse']**2)):
        raise AssertionError('Unified raw SpatialZ expression metrics disagree with its supplied report.')
    original_ds=pd.read_csv(ds/'fold_summary.csv').query('heldout_id==1').iloc[0]
    got_log=log[log.method=='DeepSpatial'].iloc[0]
    for key,old_key in [('PCC','global_mean_log_expression_pcc'),
                        ('rare_recall_50um','rare_macro_coverage_recall_50um'),
                        ('rare_recall_100um','rare_macro_coverage_recall_100um')]:
        if not np.isclose(got_log[key],original_ds[old_key]):
            raise AssertionError(f'DeepSpatial unified metric differs from original: {key}')
    os.environ.setdefault('MPLCONFIGDIR',str(out/'.matplotlib'))
    plot_comparison(main_table,all_classes,out)
    d=raw.set_index('method').loc['DeepSpatial']
    s=raw.set_index('method').loc['SpatialZ']
    missing={name:all_classes.loc[(all_classes.method==name) & all_classes.missing_in_prediction,
                                  'cell_class'].tolist() for name in ('DeepSpatial','SpatialZ')}
    report=f'''# DeepSpatial 与 SpatialZ 统一评估

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-10-18
- Verification Status: ANALYZED
- Version Label: same_slice_comparison_v1

本次只比较相同的留出切片 0.09，对应 DeepSpatial 的 holdout_1。两份真值的表达矩阵、坐标、标签、基因和细胞 ID 逐项完全一致；SpatialZ 输入及真值的 SHA256 与运行记录一致。DeepSpatial 的另外两轮不与这一个 SpatialZ 结果混合求均值。

## 本次结果

在输入 X 尺度上，SpatialZ 相对 DeepSpatial 的 MSE、MAE、PCC 差值分别为 {s.MSE-d.MSE:+.6f}、{s.MAE-d.MAE:+.6f}、{s.PCC-d.PCC:+.6f}。误差指标的负差值表示 SpatialZ 更低，PCC 的正差值表示 SpatialZ 更高。

稀有类 50 µm 宏平均同类召回率分别为 DeepSpatial {d.rare_recall_50um:.2%}、SpatialZ {s.rare_recall_50um:.2%}；100 µm 分别为 {d.rare_recall_100um:.2%}、{s.rare_recall_100um:.2%}。SpatialZ 减去 DeepSpatial 的差值分别为 {(s.rare_recall_50um-d.rare_recall_50um)*100:+.2f}、{(s.rare_recall_100um-d.rare_recall_100um)*100:+.2f} 个百分点。这是单张切片的描述性差异，不作显著性判断。

真值共有 {len(np.unique(truth['labels']))} 个大类，其中 {len(rare_classes)} 个为本次固定的稀有类。DeepSpatial 完全缺失的真值类别：{missing['DeepSpatial']}；SpatialZ 完全缺失的真值类别：{missing['SpatialZ']}。非零生成数量也可能对应零空间覆盖，必须同时查看逐类召回，不能用一个宏平均概括所有稀有类型。

## 主表 使用 X 提供的表达尺度

{markdown_table(raw)}

MSE、MAE 越低越好，PCC 与同类覆盖召回率越高越好。召回率列为 0–1 的比例，乘 100 得百分数。MSE 与 MAE 的单位依赖输入表达尺度，不能跨尺度直接比较。

![同一切片五项指标对比]({out/'comparison.png'})

## 相同预处理 log1p 后的补充结果

{markdown_table(log)}

## 统一口径

- 表达指标比较 155 个基因的平均表达向量：先在细胞维度求每个基因的均值，再计算 MSE、MAE、Pearson PCC。PCC 就是皮尔逊相关系数。主表不进行额外归一化；补充表先逐细胞执行 log1p(max(X,0)) 再取均值，与原 DeepSpatial 周报一致。它们不是逐细胞恢复准确率。
- 两种方法细胞数不同且无一一配对，未将任意矩阵行强行相减。另在 spatial_nn_expression.csv 提供真实细胞到 XY 最近预测细胞的 log1p 表达误差；该空间关联不使用真实细胞类型配对，允许多对一，PCC 跨全部匹配的细胞与基因元素展开计算，仅作局部诊断。
- 稀有类固定由 DeepSpatial 本轮四张训练切片中的占比 <1% 定义，共六类，两种方法使用同一列表；每类分别计算 50/100 µm XY 范围内是否存在同类预测，再对六类等权平均。真值存在但预测完全缺失的类，其召回率为 0，不能从宏平均分母删除。只用 SpatialZ 两张输入定义稀有类，得到的类别列表也相同。
- 两种方法沿用同一原始 XY 坐标系，没有根据留出真值拟合新的配准或缩放。SpatialZ 输出为 0.09 的二维虚拟切片，DeepSpatial 使用目标 Z±5 µm 的既有薄层；两者都按 XY 进行空间评价。点数使用原生输出，并记录在主表，未按真值数调整生成或事后补点。

## 稀有类逐类对比

{markdown_table(table_rare.reset_index())}

![稀有类逐类同类召回]({out/'rare_recall_by_class.png'})

## 可比性和解释边界

1. 这是相同目标、相同真值和评估代码下的现有输出比较。DeepSpatial 使用四张训练切片，SpatialZ 仅使用目标两侧的两张；输入信息量和生成机制不完全相同，不能写成严格控制训练数据的消融比较。
2. 只有一个共同留出切片，不能据此宣称普遍优越或显著提升。全局平均表达可能掩盖局部误差，须同时查看逐类召回和附加的局部诊断。六个稀有类中部分真值仅 3–5 个，结果有很强的小样本敏感性。
3. SpatialZ 原版与联合采样修改版严格区分。原版主表使用 merfish_holdout_009/generated_009.h5ad；若传入 --include-variants，联合采样 gene/cell 各四个表达种子另列 variants.csv。这些重复固定了几何坐标和类别，只改变表达抽样，不能作为空间恢复的四次独立重复，也不能据此估计 SpatialZ 原版全流程的波动。
4. merfish_first_pair/midpoint.h5ad 使用 0.04 与 0.09 作为输入，不是本次 0.09 留出结果，故排除。未把真实切片拼回生成点集来抬高召回率。Region、neuron_class 等来源注释不作为独立预测身份依据。
5. DeepSpatial 本轮有 7 个负表达元素：主表使用其提供的值，log1p 表中按原评估裁剪为 0。两种方法保存 X 均无 log1p 预处理标记；输入数据本身已处理/配准，原始上游流程没有重新估计。仅从已有记录无法独立排除上游处理曾接触被留出切片的可能。

## 核对范围

复算的 SpatialZ 原尺度 PCC、MAE、MSE 与附带报告一致（MSE = 原 RMSE²）；DeepSpatial 的 log1p PCC 和两项稀有类召回与原表一致。分析不修改原始数据，也未重新执行模型训练或生成。完整路径、校验和和参数见 audit.json。

统计解释检查覆盖 11/11 项：整体与分层差异、从均值推断个体、单数据选择限制、碰撞变量（无因果调整，不适用）、稀有基率、回归均值（无前后干预，不适用）、仅对已保存运行作结论、多指标选择、后验尺度/配对选择、相关与因果、反向因果（无因果方向分析，不适用）。本次不提供 p 值或推断性置信区间；所有结论限于指定保存结果。

## 输出

- comparison.csv：两种原方法的两种表达尺度和同一稀有类召回。
- cell_type_metrics.csv、rare_recall_comparison.csv：逐类数量与覆盖，缺失类保留为 0。
- spatial_nn_expression.csv：XY 最近邻关联后的表达局部诊断。
- variants.csv：联合采样修改版本的独立附表。
- 各方法 gene_profiles.csv：每个基因的真值与预测平均表达，可追溯全部表达指标。
- comparison.png、rare_recall_by_class.png：汇报图。
- audit.json：文件指纹、数据身份和原配置核对。
'''
    (out/'统一对比报告.md').write_text(report,encoding='utf-8')
    print('Provided X scale\n'+raw.to_string(index=False))
    print('\nlog1p scale\n'+log.to_string(index=False))
    print('\nRare class comparison\n'+table_rare.to_string())
    print(f'\nSaved to {out}')


if __name__=='__main__':
    main()
