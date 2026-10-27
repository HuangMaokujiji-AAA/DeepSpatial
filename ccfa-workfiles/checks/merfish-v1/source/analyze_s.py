"""Read-only paired audit of completed experiment S and its original V1 weights."""
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / 'HM_DeepSpital/outputs/s_holdouts_1_2_3'
V1_ROOT = ROOT / 'HM_DeepSpital/outputs/v1_holdouts_1_2_3'
DEST = RUN / 'analysis'
DEST.mkdir(exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(DEST / '.matplotlib')

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from analyze_v1 import dense, md_table, niche_composition, spatial_expression
from analyze_three_holdouts import parent_copy, extra_metrics

BASE, NEW, COPY = 'V1 reg=0.05', 'Experiment S', 'S matched source copy'


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def verify_sampling(volume,folder,original,train_ids):
    budget=pd.read_csv(folder/'sampling_budget.csv')
    embedded=volume.uns['depth_type_sampling']['budget']
    pd.testing.assert_frame_equal(budget,embedded,check_dtype=False,check_index_type=False,rtol=1e-10,atol=1e-10)
    np.testing.assert_array_equal(np.floor((volume.obs.z_coord+5)/10).astype(int),volume.obs.depth_bin_index)
    np.testing.assert_allclose(volume.obs.depth_bin_center_um,volume.obs.depth_bin_index*10)
    np.testing.assert_allclose(volume.obs.z_norm,volume.obs.z_coord/200)
    train_z=np.asarray(train_ids)*50
    obs=volume.obs.copy()
    positions=np.searchsorted(train_z,obs.source_z.to_numpy())
    fwd=obs.source_direction.astype(str).to_numpy()=='forward'
    gap=positions-np.where(fwd,0,1)
    assert ((gap>=0)&(gap<len(train_z)-1)).all()
    obs['gap_start_um'],obs['gap_end_um']=train_z[gap],train_z[gap+1]
    keys=['gap_start_um','gap_end_um','depth_bin_index','source_direction','source_cell_class']
    actual=obs.groupby(keys,observed=True).size()
    expected=budget.set_index(keys)
    np.testing.assert_array_equal(actual.reindex(expected.index,fill_value=0),expected.allocated_n)
    assert int(budget.allocated_n.sum())==volume.n_obs
    for bounds,frame in budget.groupby(['gap_start_um','gap_end_um']):
        left,right=(original[int(z/50)] for z in bounds)
        total=max(1,int((left.n_obs+right.n_obs)/2*(bounds[1]-bounds[0])/10))
        assert int(frame.allocated_n.sum())==total
        np.testing.assert_allclose(frame.expected_n.sum(),total,atol=1e-8)
    grouped=budget.groupby(['gap_start_um','gap_end_um','depth_bin_index','source_cell_class'])[['allocated_n','conditional_expected_n']].sum()
    assert (abs(grouped.allocated_n-grouped.conditional_expected_n)<=1+1e-9).all()
    for z in train_z:
        source=original[int(z/50)]
        mask=obs.source_z.to_numpy()==z
        ids=obs.loc[mask,'source_cell_index'].to_numpy()
        assert list(source.obs_names[ids])==list(obs.loc[mask,'source_cell_id'].astype(str))
        np.testing.assert_array_equal(source.obs.cell_class.astype(str).to_numpy()[ids],obs.loc[mask,'source_cell_class'].astype(str))
    saved_counts=pd.read_csv(folder/'sampling_counts.csv')
    keys=['depth_bin_index','depth_bin_center_um','source_direction','source_cell_class','cell_class']
    saved=saved_counts.set_index(keys).generated_n.sort_index()
    computed=volume.obs.groupby(keys,observed=True).size().sort_index()
    np.testing.assert_array_equal(saved.to_numpy(),computed.to_numpy())
    assert list(saved.index)==list(computed.index)
    assert int(volume.uns['hm_reconstruction']['generated_cells'])==volume.n_obs
    return budget


def main():
    notebook=json.loads((ROOT/'merfish_leave_one_slice_out.ipynb').read_text(encoding='utf-8'))
    code=next(''.join(c['source']) for c in notebook['cells'] if c.get('id')=='metric-helpers')
    scope=dict(np=np,pd=pd,ad=ad,sp=sp,cKDTree=cKDTree,LABEL_KEY='cell_class',RARE_TRAIN_FRACTION=.01,COVERAGE_RADII_UM=(25,50,100))
    exec(compile(code,'notebook:metric-helpers','exec'),scope)
    evaluate=scope['evaluate_holdout']
    original={i:ad.read_h5ad(next((ROOT/'data/merfish_mouse_hypothalamus').rglob(f'merfish_{i}.h5ad'))) for i in range(5)}
    summaries,tables,maps,checks,counts,audits,traces,anchors,programs=[],[],[],[],[],[],[],[],[]
    plots={}
    for i in (1,2,3):
        folder=RUN/f'holdout_{i}'
        baseline_dir=V1_ROOT/f'holdout_{i}'
        cfg=json.loads((folder/'run_config.json').read_text(encoding='utf-8'))
        baseline_cfg=json.loads((baseline_dir/'run_config.json').read_text(encoding='utf-8'))
        for key in ('seed','input_glob','slice_z_um','data','model','training','run'):
            assert cfg[key]==baseline_cfg[key],key
        rec=cfg['reconstruction'].copy()
        for key in ('source_sampling','device','depth_bin_width_um','depth_grid_origin_um'): rec.pop(key)
        assert rec==baseline_cfg['reconstruction']
        assert cfg['reconstruction']['depth_bin_width_um']==10 and cfg['reconstruction']['depth_grid_origin_um']==0
        meta=cfg['experiment_s']
        assert meta['checkpoint_weights']=='EMA' and meta['training_performed'] is False and meta['audit_only'] is False
        assert meta['sampling_seed']==42 and cfg['run']['heldout_indices']==[i]
        checkpoint_matches=sha256(baseline_dir/'model/model.ckpt')==meta['checkpoint_sha256']
        assert checkpoint_matches
        truth=original[i]
        train_ids=[j for j in range(5) if j!=i]
        train=pd.concat([original[j].obs.cell_class.astype(str) for j in train_ids]).value_counts()
        volume=ad.read_h5ad(folder/'generated.h5ad')
        baseline_volume=ad.read_h5ad(baseline_dir/'generated.h5ad')
        assert volume.n_obs==baseline_volume.n_obs
        assert truth.var_names.equals(volume.var_names) and volume.var_names.equals(baseline_volume.var_names)
        assert i*50 not in set(volume.obs.source_z)
        assert not volume.obs.is_observed.any()
        np.testing.assert_allclose(volume.obsm['spatial_3d'][:,:2],volume.obsm['spatial'])
        np.testing.assert_allclose(volume.obsm['spatial_3d'][:,2],volume.obs.z_coord)
        budget=verify_sampling(volume,folder,original,train_ids)
        pred=volume[abs(volume.obs.z_coord-i*50)<=5].copy()
        baseline=baseline_volume[abs(baseline_volume.obs.z_coord-i*50)<=5].copy()
        copy_pred=parent_copy(pred,original)
        runs={BASE:baseline,NEW:pred,COPY:copy_pred}
        plots[i]=(truth,baseline,pred)
        niche=niche_composition(truth,runs)
        spatial_rows=[dict(heldout_id=i,**r) for width in (100,200) for r in spatial_expression(truth,runs,width)]
        maps.extend(spatial_rows)
        spatial_frame=pd.DataFrame(spatial_rows)
        local_tables={}
        local_summaries={}
        for name,a in runs.items():
            table,summary=evaluate(truth,a,train)
            summary.update(heldout_id=i,run=name,**extra_metrics(truth,a,table),**niche[name])
            for width in (100,200):
                data=spatial_frame[(spatial_frame.run==name)&(spatial_frame.bin_width_um==width)]
                summary[f'spatial_expression_median_pcc_{width}um']=float(data.spatial_expression_pcc.median())
                summary[f'spatial_expression_shared_bins_{width}um']=int(data.shared_bins.iloc[0])
            rare=table[table.is_rare_train]
            valid=rare.predicted_n>0
            summary['rare_micro_precision_100um']=float(np.average(rare.loc[valid,'coverage_precision_100um'],weights=rare.loc[valid,'predicted_n']))
            local_tables[name]=table
            local_summaries[name]=summary
            tables.append(table.assign(run=name,heldout_id=i))
            x=np.log1p(np.maximum(dense(a),0))
            tx=np.log1p(np.maximum(dense(truth),0))
            program_ids=truth.var_names.get_indexer(['Mbp','Opalin','Ermn','Gjc3'])
            for label in table.cell_class:
                tm=truth.obs.cell_class.astype(str).to_numpy()==label
                pm=a.obs.cell_class.astype(str).to_numpy()==label
                ts,ps=tx[tm][:,program_ids].mean(1),x[pm][:,program_ids].mean(1)
                programs.append(dict(heldout_id=i,run=name,cell_class=label,truth_n=int(tm.sum()),predicted_n=int(pm.sum()),
                                     truth_program_mean=float(ts.mean()),predicted_program_mean=float(ps.mean()) if len(ps) else np.nan))
                if label in set(rare.cell_class):
                    d=cKDTree(a.obsm['spatial'][pm]).query(truth.obsm['spatial'][tm])[0] if pm.any() else np.full(tm.sum(),np.inf)
                    for index,distance in zip(np.flatnonzero(tm),d):
                        anchors.append(dict(heldout_id=i,run=name,cell_class=label,truth_cell_id=str(truth.obs_names[index]),
                                            nearest_same_class_um=float(distance),within_50um=bool(distance<=50),within_100um=bool(distance<=100)))
        common=set(local_tables[BASE].loc[local_tables[BASE].predicted_n>0,'cell_class'])&set(local_tables[NEW].loc[local_tables[NEW].predicted_n>0,'cell_class'])
        for name,summary in local_summaries.items():
            subset=local_tables[name][local_tables[name].cell_class.isin(common)]
            summary['paired_present_class_n']=len(subset)
            summary['paired_present_class_expression_pcc']=float(subset.mean_log_expression_pcc.mean())
            summary['paired_present_rare_expression_pcc']=float(subset.loc[subset.is_rare_train,'mean_log_expression_pcc'].mean())
            summaries.append(summary)
        audit=pd.read_csv(folder/'sampling_audit.csv')
        audits.append(audit.assign(heldout_id=i))
        current_counts=budget[budget.depth_bin_index==i*5].groupby('source_cell_class')[['expected_n','allocated_n']].sum()
        for label in sorted(train.index):
            b=local_tables[BASE].set_index('cell_class').loc[label]
            s=local_tables[NEW].set_index('cell_class').loc[label]
            expectation=float(current_counts.loc[label,'expected_n']) if label in current_counts.index else 0.
            counts.append(dict(heldout_id=i,cell_class=label,truth_n=int(s.truth_n),v1_n=int(b.predicted_n),s_n=int(s.predicted_n),
                               expected_n=expectation,v1_count_error=float(b.predicted_n-expectation),s_count_error=float(s.predicted_n-expectation)))
        delta=np.linalg.norm(pred.obsm['spatial']-copy_pred.obsm['spatial'],axis=1)
        for label in sorted(train.index):
            mask=pred.obs.source_cell_class.astype(str).to_numpy()==label
            traces.append(dict(heldout_id=i,cell_class=label,predicted_n=int(mask.sum()),unique_parents=int(pred.obs.loc[mask,'parent_id'].nunique()),
                               parent_displacement_median_um=float(np.median(delta[mask])) if mask.any() else np.nan))
        checks.append(dict(heldout_id=i,volume_n=volume.n_obs,v1_slab_n=baseline.n_obs,s_slab_n=pred.n_obs,
                           checkpoint_hash_matches=True,training_parameters_match=True,volume_count_preserved=True,
                           sampling_budget_matches_obs=True,sampling_counts_matches_obs=True,source_ids_and_types_verified=True,
                           finite_expression=bool(np.isfinite(dense(volume)).all()),finite_coordinates=bool(np.isfinite(volume.obsm['spatial_3d']).all()),
                           negative_expression_entries=int((dense(volume)<0).sum()),
                           class_switches=int((volume.obs.cell_class.astype(str)!=volume.obs.source_cell_class.astype(str)).sum()),
                           zero_expression_slab_cells=int((dense(pred)==0).all(1).sum()),
                           parent_displacement_median_um=float(np.median(delta))))
        print('FOLD',i,json.dumps(local_summaries[NEW]),flush=True)
        del volume,baseline_volume
    summary=pd.DataFrame(summaries)
    table=pd.concat(tables,ignore_index=True)
    qc=pd.DataFrame(checks)
    count_table=pd.DataFrame(counts)
    means=summary.groupby('run').agg({k:['mean','std'] for k in [
        'missing_classes','cell_fraction_total_variation','tissue_histogram_tv','symmetric_mean_nn_um','xy_sd_ratio_mean',
        'global_mean_log_expression_pcc','paired_present_class_expression_pcc','paired_present_rare_expression_pcc',
        'rare_macro_coverage_recall_25um','rare_macro_coverage_recall_50um','rare_macro_coverage_recall_100um',
        'rare_micro_recall_100um','rare_micro_precision_100um','niche_js_divergence_mean','niche_prediction_empty_fraction',
        'spatial_expression_median_pcc_100um','spatial_expression_median_pcc_200um']})
    means.columns=['_'.join(c) for c in means.columns]
    means.reset_index().to_csv(DEST/'fold_mean_std.csv',index=False)
    for name,frame in [('fold_summary',summary),('cell_type_metrics',table),('quality_checks',qc),
                       ('expected_count_comparison',count_table),('sampling_audits',pd.concat(audits,ignore_index=True)),
                       ('spatial_expression_metrics',pd.DataFrame(maps)),('source_diagnostics',pd.DataFrame(traces)),
                       ('rare_cell_distances',pd.DataFrame(anchors)),('program_metrics',pd.DataFrame(programs))]:
        frame.to_csv(DEST/f'{name}.csv',index=False)
    make_plots(plots,summary,table,count_table)
    write_report(summary,table,qc,count_table,means,pd.concat(audits,ignore_index=True),pd.DataFrame(traces))
    print('MEANS',means.to_string(),flush=True)


def make_plots(data,summary,table,counts):
    plt.rcParams.update({'font.family':'Times New Roman','font.size':11,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for ax,(key,title,scale) in zip(axes.flat,[
        ('missing_classes','Missing cell classes (lower is better)',1),
        ('rare_macro_coverage_recall_100um','Rare-class macro recall @100 µm (%)',100),
        ('tissue_histogram_tv','Tissue density error (TV)',1),
        ('spatial_expression_median_pcc_100um','Spatial expression PCC (100 µm bins)',1),
    ]):
        for shift,name,color,label in [(-.18,BASE,'#357ba8','V1'),(.18,NEW,'#d15e32','S')]:
            values=summary[summary.run==name].sort_values('heldout_id')[key].to_numpy()*scale
            ax.bar(np.arange(3)+shift,values,width=.34,color=color,label=label)
            for x,value in zip(np.arange(3)+shift,values):
                ax.text(x,value,f'{value:.1f}' if scale==100 else (f'{value:.0f}' if key=='missing_classes' else f'{value:.3f}'),ha='center',va='bottom',fontsize=9)
        ax.set_xticks(range(3),['Holdout 1','Holdout 2','Holdout 3'])
        ax.set_title(title,fontsize=11)
        ax.spines[['right','top']].set_visible(False)
        ax.set_ylim(0,ax.get_ylim()[1]*1.18)
    axes[0,0].legend(frameon=False)
    for ext in ('png','svg'):fig.savefig(DEST/f'overall_comparison.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(9,12),layout='constrained')
    for row,i in enumerate((1,2,3)):
        truth,base,pred=data[i]
        xy=np.asarray(truth.obsm['spatial'])
        for col,label in enumerate(['OD Mature 3','OD Immature 2']):
            ax=axes[row,col]
            tm=truth.obs.cell_class.astype(str).to_numpy()==label
            ax.scatter(*xy.T,s=1,color='#c8cdd2',alpha=.35,rasterized=True)
            for point in xy[tm]:ax.add_patch(plt.Circle(point,100,color='#357ba8',alpha=.12,lw=0))
            ax.scatter(*xy[tm].T,s=60,facecolors='none',edgecolors='#357ba8',label=f'Truth n={tm.sum()}')
            for name,a,color,marker in [('V1',base,'#69966b','+'),('S',pred,'#d15e32','x')]:
                pm=a.obs.cell_class.astype(str).to_numpy()==label
                ax.scatter(*np.asarray(a.obsm['spatial'])[pm].T,s=45,color=color,marker=marker,label=f'{name} n={pm.sum()}')
            ax.set(title=f'Holdout {i}: {label}',xlim=(-1000,1000),ylim=(-1000,1000),xlabel='X (µm)',ylabel='Y (µm)',aspect='equal')
            ax.legend(fontsize=9,loc='upper left')
    for ext in ('png','svg'):fig.savefig(DEST/f'rare_cell_spatial.{ext}',dpi=170)
    plt.close(fig)


def write_report(summary,table,qc,counts,means,audits,traces):
    base=summary[summary.run==BASE].set_index('heldout_id')
    new=summary[summary.run==NEW].set_index('heldout_id')
    copy=summary[summary.run==COPY].set_index('heldout_id')
    rows=[]
    metrics=[('组织密度误差 TV ↓','tissue_histogram_tv','.4f'),
             ('类型比例总变差 ↓','cell_fraction_total_variation','.4f'),
             ('双向最近邻距离 µm ↓','symmetric_mean_nn_um','.2f'),
             ('稀有类宏平均覆盖 @50 µm ↑','rare_macro_coverage_recall_50um','.2%'),
             ('稀有类宏平均覆盖 @100 µm ↑','rare_macro_coverage_recall_100um','.2%'),
             ('稀有类细胞数加权覆盖 @100 µm ↑','rare_micro_recall_100um','.2%'),
             ('稀有类细胞数加权 precision @100 µm ↑','rare_micro_precision_100um','.2%'),
             ('全局均值表达 PCC ↑','global_mean_log_expression_pcc','.5f'),
             ('共同存在类别的均值表达 PCC ↑','paired_present_class_expression_pcc','.4f'),
             ('共同存在稀有类别的均值表达 PCC ↑','paired_present_rare_expression_pcc','.4f'),
             ('空间表达 PCC，100 µm 网格 ↑','spatial_expression_median_pcc_100um','.4f'),
             ('空间表达 PCC，200 µm 网格 ↑','spatial_expression_median_pcc_200um','.4f'),
             ('类型邻域组成 JS 散度 ↓','niche_js_divergence_mean','.4f')]
    for title,key,fmt in metrics:
        rows.append({'指标':title,'原 V1':format(base[key].mean(),fmt),'S':format(new[key].mean(),fmt),
                     'S 来源直接复制':format(copy[key].mean(),fmt)})
    folds=[]
    for i in (1,2,3):
        b,s=base.loc[i],new.loc[i]
        folds.append({'留出':str(i),'真值/V1/S数量':f'{int(s.truth_n)}/{int(b.predicted_n)}/{int(s.predicted_n)}',
                      '类别数：V1→S':f'{int(b.predicted_classes)}→{int(s.predicted_classes)}',
                      '稀有类100µm覆盖：V1→S':f'{b.rare_macro_coverage_recall_100um:.2%}→{s.rare_macro_coverage_recall_100um:.2%}'})
    rare_labels=table[(table.run==NEW)&(table.heldout_id==1)&table.is_rare_train].cell_class.tolist()
    rare_rows=[]
    for label in rare_labels:
        row={'稀有类':label}
        for i in (1,2,3):
            b=table[(table.run==BASE)&(table.heldout_id==i)&(table.cell_class==label)].iloc[0]
            s=table[(table.run==NEW)&(table.heldout_id==i)&(table.cell_class==label)].iloc[0]
            row[f'留出{i}：V1→S']=f'{b.coverage_recall_100um:.1%}→{s.coverage_recall_100um:.1%}'
        rare_rows.append(row)
    expected_rows=counts[counts.cell_class.isin(['OD Mature 3','OD Immature 2'])].copy()
    expected_rows=expected_rows[['heldout_id','cell_class','truth_n','expected_n','v1_n','s_n']]
    expected_rows['expected_n']=expected_rows.expected_n.map(lambda x:f'{x:.3f}')
    expected_rows.columns=['留出','类别','真值','可见数据推算期望','V1','S']
    rare_counts=counts[counts.cell_class.isin(rare_labels)]
    count_mae={name:float(abs(counts[f'{name}_count_error']).mean()) for name in ('v1','s')}
    rare_mae={name:float(abs(rare_counts[f'{name}_count_error']).mean()) for name in ('v1','s')}
    pooling={}
    for name,frame in [('V1',base),('S',new),('来源复制',copy)]:
        pooling[name]=(int(round((frame.rare_micro_recall_100um*frame.rare_true_n).sum())),int(frame.rare_true_n.sum()))
    report=f'''# 实验 S 三轮结果分析

分析日期：2026-10-27。输入：`HM_DeepSpital/outputs/s_holdouts_1_2_3`。对照为同一组 checkpoint 的 `v1_holdouts_1_2_3`，其训练 reg=.05。

## 判断

实验 S 达到了降低薄层类型计数随机偏差的直接目标，补上了留出 1 中原先整类缺失的 OD Mature 3。但本轮没有观察到稀有类空间恢复收益：六类宏平均 100 µm 覆盖由 20.30% 变为 19.90%，三轮均略降；OD Mature 3、OD Immature 2 三轮覆盖仍全部为 0。

应保留 S 作为来源/深度抽样控制与后续实验的稳定预算参照，不能据此声称已改善稀有类空间恢复。下一步主要瓶颈仍是可见端点支持不足和不可信局部运输。

## 实验身份与输出核验

- 三轮 S 的 checkpoint SHA-256 与原 V1 的 `holdout_i/model/model.ckpt` 文件逐一匹配；记录使用 EMA、sampling_seed=42、未重新训练。
- seed、训练数据设置、模型设置、训练设置、留出编号及输入文件记录与各自原 V1 相同。生成 thickness=10、steps=100、chunk_size=512、表达稀疏化和中间状态保存设置沿用原运行；只增加深度×类型分层调度、固定 10 µm 网格及设备记录。
- 每轮均仅有相应四张可见切片的 source_z，没有隐藏切片来源。已检查全体积 source_cell_id、source_cell_index、source_cell_class 与本地原始可见数据一致。
- 三轮整段生成总数分别为 {', '.join(str(int(x)) for x in qc.volume_n)}，与原 V1 完全一致；评价薄层数量随抽样发生变化，不按隐藏真值计数截取。
- 已从全部 obs 复算每区间×薄层×方向×类型计数，与 sampling_budget.csv 的 allocated_n 完全一致；预算 CSV 与 H5AD 内嵌预算一致，实际 sampling_counts.csv 与 H5AD 一致。
- 每段 expected_n 与 allocated_n 的总和均等于原设定生成数量；每层类型的实际整数预算与条件期望之差不超过 1。
- Z、depth_bin_index、depth_bin_center_um、z_norm、spatial_3d 相互一致。三轮表达和坐标均有限，无负表达、评价薄层无全零细胞。
- 三轮全部生成 cell_class 与 source_cell_class 相同。类别计数的变化来自来源/深度调度，不能解释成模型新识别出隐藏组织身份。

## 抽样目标：计数更稳定，但该期望不是隐藏真值

{md_table(expected_rows)}

对三轮共 45 个类型×切片组合，实际计数相对“可见切片推算期望”的平均绝对偏差由 {count_mae['v1']:.2f} 个降至 {count_mae['s']:.2f} 个；六个稀有类的 18 个组合则由 {rare_mae['v1']:.2f} 降至 {rare_mae['s']:.2f}。

这里比较的是算法应执行的期望预算，绝不是预测数量相对隐藏真值的准确率。例：留出 3 的 OD Mature 3 期望约 4.07、S 实际 4，而隐藏真值为 3；更加贴近期望不保证更加贴近真值。

留出 1 的 OD Mature 3 从 0 个补到 1 个，因此 S 三轮评价薄层都覆盖 15 个大类。来源两侧该类只有 0/2 个，期望约 .963，不应为了真值 3 个强制补 3 个。

原始 S 抽样审计每轮有 100 个仅来源调度的种子：该类留出 1 均值 .95、零计数 5%，数量只能为 0 或 1；留出 2 均为 3 个；留出 3 为 4 或 5 个。它支持预算低方差，但不是 100 次 ODE 空间准确率实验；本报告没有为 V1 另跑同规模的重复抽样对照，不能据此报告两者零计数频率的经验显著差异。

## 三轮空间与表达指标

{md_table(pd.DataFrame(rows))}

三轮等权平均，标准差见 `fold_mean_std.csv`。三轮是不同切片划分，每轮 S 仅一个完整推理种子，不是三个独立训练种子，也不是 100 次完整重建。

本报告的空间表达和邻域 JS 在 V1、S、S 来源复制共同有效的位置/网格上重新计算。因此其支持域不同于上一份包含原版 DeepSpatial 的报告，不能直接将跨报告的 PCC/JS 数字相减。例如当前 V1 空间表达 100 µm 中位 PCC 为 .3785，上一报告为 .3942，差异来自共同支持范围变化，并非 V1 文件被更改。

表达 PCC 对各类先取均值，再在基因维度相关；不是逐细胞准确率。为避免 V1 留出 1 缺少 OD Mature 3 造成分母不一致，“共同存在类别”使用 14/15/15 类，稀有类使用 5/6/6 类。覆盖召回仍将缺失类别计 0。CSV 中原协议的宏 precision 对没有预测的类未定义；主表使用细胞数加权 precision，避免直接比较五类和六类分母。

![主要指标](overall_comparison.png)

### 整体分布

组织密度 TV 从 .1711 降至 .1671，约改善 2.4%；类型比例 TV 从 .02564 降至 .02431。几何最近邻均值从 12.63 降至 12.53 µm，但留出 3 由 12.45 变为 12.68 µm。与先前降低熵正则带来的显著收缩缓解不同，本轮主要是小幅变化。

全局表达均值 PCC 为 .99575→.99572，基本不变。共同类别表达 PCC 有小幅提高；因为 checkpoint 没有变化，这反映来源组合与目标深度变化，不能解释为表达网络经过 S 学得更好。

空间表达 100 µm 网格 PCC 本轮三轮均下降，平均 .3785→.3580；200 µm 网格 .5261→.5239。类型邻域 JS .2358→.2348，基本持平。本轮没有显示局部分子空间分布的稳定增益。

## 稀有类覆盖：存在性改善，位置没有改善

{md_table(pd.DataFrame(folds))}

{md_table(pd.DataFrame(rare_rows))}

100 µm 宏平均覆盖下降约 .40 个百分点；50 µm 宏平均覆盖由 7.37% 降至 6.11%。细胞数加权三轮平均 100 µm 覆盖从 35.50% 降至 32.96%。合并三轮真实稀有细胞，覆盖计数从 {pooling['V1'][0]}/{pooling['V1'][1]}=36.02% 变为 {pooling['S'][0]}/{pooling['S'][1]}=33.33%；同期稀有类预测总数从 369 增至 394。

这说明问题不是简单的生成数量不够。覆盖定义允许多个真实点共享同一预测邻居，不能把这些计数解释为逐细胞识别数。

- OD Mature 3 的 S 预测数量为 1/3/4，三轮 100 µm 覆盖仍为 0。留出 1 新生成的单个该类细胞，与三个真实该类点的平均距离约 1313 µm，存在不等于定位成功。
- OD Immature 2 的 S 数量为 7/6/6，三轮覆盖仍为 0。留出 2 的同类最近邻平均距离从约 598 µm 降至 273 µm，但仍未进入 100 µm 阈值；留出 1/3 分别约 509/498 µm。
- Endothelial 2 在留出 2/3 上升，OD Mature 4 在留出 3 从 0 升为 12.5%；OD Mature 1 三轮均下降，Pericytes 留出 3 由 65.4% 降至 50.0%。当前不能描述为各稀有类一致获益。
- OD Immature 2 的表达 PCC .534/.656/.755→.603/.760/.801，分子均值有所改善，但空间覆盖仍为 0。来源细胞组合改变可以提高这种类均值统计，不能替代定位证据。

![极稀有类别空间位置](rare_cell_spatial.png)

图中蓝色空心圈是真实细胞，圆盘为 100 µm 范围；绿色加号为原 V1，橙叉为 S。两类三轮仍未覆盖真实位置。

## 原因与下一步

S 只改变目标深度和来源预算，没有改变训练配对、学得的速度场或空间支持。S 的全体细胞 parent 位移中位数约 41.5/47.3/44.7 µm，与 V1 接近；OD Immature 2 的位移仍约 409/221/184 µm，OD Mature 3 为 385/307/130 µm。稀有类的较大位移和可见端点缺少近距离支持的问题仍然存在。

固定 S 的来源、类型和目标深度，直接保留来源 XY/表达的诊断对照，100 µm 稀有类宏平均覆盖为 29.31%/17.10%/36.72%，平均 27.71%，高于 S 的 19.90%。S 仅在留出 2 优于该对照。该对照不是完整三维重建模型，但再次表明当前运输会破坏部分可见来源中已有的局部结构。

建议：

1. 保留 S 作为后续比较中的固定低方差调度。当前证据支持预算实现正确，不支持将 S 宣称为空间准确率改进。
2. 优先推进局部配对实验 G：记录每类物理候选距离和有效支持；禁止均衡抽样把极低质量的远距离类块重新放大；无可信对应时显式标记 unsupported，而非强配远处目标。
3. G-real/G-zero/G-shuffle 共享 S 的固定来源调度和相同配对索引，分离生态位条件增益。S 结果不提供生态位模块有效性的新增证据。
4. 如果需要判断这轮小幅升降是否稳定，可在同一权重下对 V1 和 S 各做多个匹配推理种子，并比较均值、方差；无需重新训练。现有 100 种子审计只覆盖计数，没有覆盖 XY 预测，不能拿来报告空间显著性。
5. 所有参数和选择规则仍应在训练侧内部验证或独立保留样本上确定；当前三张隐藏切片均已被查看，不再作为全新未知测试。

## 可复算文件

- `fold_summary.csv` / `fold_mean_std.csv`：三轮整体指标与切片间标准差。
- `cell_type_metrics.csv` / `rare_cell_distances.csv`：逐类指标与真实稀有点的最近同类距离。
- `expected_count_comparison.csv` / `sampling_audits.csv`：可见期望与本轮计数、100 种子来源审计。
- `quality_checks.csv`：权重身份、配置、预算、来源和数值核验。
- `source_diagnostics.csv` / `spatial_expression_metrics.csv` / `program_metrics.csv`：来源位移与分子分布。
- 源码：`../../../../ccfa-workfiles/checks/merfish-v1/source/analyze_s.py`，复用同目录原有分析函数。全部模型和生成 H5AD 保持原样。
'''
    assert '\ufffd' not in report
    (DEST/'report.md').write_text(report,encoding='utf-8')
    assert (DEST/'report.md').read_text(encoding='utf-8')==report


if __name__=='__main__':
    main()
