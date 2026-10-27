"""Audit three completed V1 folds against saved original and previous V1 runs."""
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / 'HM_DeepSpital/outputs/v1_holdouts_1_2_3'
DEST = RUN / 'analysis'
DEST.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(DEST / '.matplotlib'))

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from analyze_v1 import dense, pcc, md_table, niche_composition, spatial_expression

BASE = 'DeepSpatial'
NEW = 'V1 reg=0.05'
OLD = 'V1 reg=0.8'
COPY = 'Matched source copy'


def parent_copy(pred, original):
    x, xy, labels = np.zeros(pred.shape), np.zeros((pred.n_obs,2)), np.empty(pred.n_obs,object)
    for z in pred.obs.source_z.unique():
        mask = pred.obs.source_z.to_numpy() == z
        src = original[int(round(z/50))]
        ids = pred.obs.loc[mask,'source_cell_index'].to_numpy()
        assert list(src.obs_names[ids]) == list(pred.obs.loc[mask,'source_cell_id'].astype(str))
        x[mask],xy[mask] = dense(src)[ids],src.obsm['spatial'][ids]
        labels[mask] = src.obs.cell_class.astype(str).to_numpy()[ids]
    np.testing.assert_array_equal(labels,pred.obs.source_cell_class.astype(str))
    return ad.AnnData(X=x,obs=pd.DataFrame({'cell_class':labels},index=pred.obs_names),
                      var=pred.var.copy(),obsm={'spatial':xy})


def extra_metrics(truth,pred,table):
    rare = table[table.is_rare_train & (table.truth_n>0)]
    tx,px = np.asarray(truth.obsm['spatial']),np.asarray(pred.obsm['spatial'])
    edges = np.arange(-1000,1001,100)
    th = np.histogram2d(*tx.T,bins=(edges,edges))[0]/truth.n_obs
    ph = np.histogram2d(*px.T,bins=(edges,edges))[0]/pred.n_obs
    d = cKDTree(px).query(tx)[0]
    rt,rp = np.linalg.norm(tx,axis=1),np.linalg.norm(px,axis=1)
    return dict(cell_fraction_total_variation=float(abs(table.truth_fraction-table.predicted_fraction).sum()/2),
                tissue_histogram_tv=float((abs(th-ph).sum()+abs(th.sum()-ph.sum()))/2),
                outside_histogram_fraction=float(1-ph.sum()),
                truth_x_sd_um=float(tx[:,0].std()),truth_y_sd_um=float(tx[:,1].std()),
                x_sd_um=float(px[:,0].std()),y_sd_um=float(px[:,1].std()),
                xy_sd_ratio_mean=float(np.mean(px.std(0)/tx.std(0))),
                fraction_outside_600um=float((rp>600).mean()),
                truth_fraction_outside_600um=float((rt>600).mean()),
                truth_outer_coverage_50um=float((d[rt>600]<=50).mean()),
                rare_true_n=int(rare.truth_n.sum()),rare_predicted_n=int(rare.predicted_n.sum()),
                rare_micro_recall_100um=float(np.average(rare.coverage_recall_100um,weights=rare.truth_n)),
                present_class_expression_n=int(table.mean_log_expression_pcc.notna().sum()),
                present_class_expression_pcc=float(table.mean_log_expression_pcc.mean()),
                present_rare_expression_n=int(rare.mean_log_expression_pcc.notna().sum()),
                present_rare_expression_pcc=float(rare.mean_log_expression_pcc.mean()))


def main():
    nb=json.loads((ROOT/'merfish_leave_one_slice_out.ipynb').read_text(encoding='utf-8'))
    source=next(''.join(c['source']) for c in nb['cells'] if c.get('id')=='metric-helpers')
    scope=dict(np=np,pd=pd,ad=ad,sp=sp,cKDTree=cKDTree,LABEL_KEY='cell_class',
               RARE_TRAIN_FRACTION=.01,COVERAGE_RADII_UM=(25,50,100))
    exec(compile(source,'notebook:metric-helpers','exec'),scope)
    evaluate=scope['evaluate_holdout']
    original={i:ad.read_h5ad(next((ROOT/'data/merfish_mouse_hypothalamus').rglob(f'merfish_{i}.h5ad'))) for i in range(5)}
    summaries,tables,qcs,maps,losses,traces,anchors,programs,config_checks,depth_counts=[],[],[],[],[],[],[],[],[],[]
    plot_data={}
    old_cfg=json.loads((ROOT/'HM_DeepSpital/outputs/v1_holdout_2_run2/run_config.json').read_text())
    new_settings=None
    for i in (1,2,3):
        folder=RUN/f'holdout_{i}'
        cfg=json.loads((folder/'run_config.json').read_text(encoding='utf-8'))
        model_cfg=json.loads((folder/'model/config.json').read_text(encoding='utf-8'))
        train_ids=[j for j in range(5) if j!=i]
        assert cfg['run']['heldout_indices']==[i]
        assert sorted(Path(p).name for p in cfg['run']['visible_files']) == [f'merfish_{j}.h5ad' for j in train_ids]
        assert cfg['data']['uot_reg']==.05 and cfg['data']['uot_tau']==.05
        settings={k:v for k,v in cfg.items() if k!='run'}
        if new_settings is None: new_settings=settings
        assert new_settings==settings
        expected_old=json.loads(json.dumps(settings))
        expected_old['data']['uot_reg']=.8
        assert expected_old=={k:v for k,v in old_cfg.items() if k!='run'}
        truth=original[i]
        train=pd.concat([original[j].obs.cell_class.astype(str) for j in train_ids]).value_counts()
        xy=np.vstack([original[j].obsm['spatial'] for j in train_ids])
        np.testing.assert_allclose([model_cfg['spatial_stats'][k] for k in ('x_min','y_min','x_range','y_range')],
                                   [*xy.min(0),*(np.ptp(xy,axis=0)+1e-8)])
        assert model_cfg['categories']==sorted(train.index)
        assert model_cfg['gene_names']==list(truth.var_names)
        feat=model_cfg['niche_featurizer']
        program_x=np.concatenate([np.log1p(dense(original[j][:,feat['program_genes']])) for j in train_ids])
        np.testing.assert_allclose(program_x.mean(0),feat['gene_mean'],rtol=1e-6,atol=1e-7)
        np.testing.assert_allclose(program_x.std(0),feat['gene_scale'],rtol=1e-6,atol=1e-7)
        vol=ad.read_h5ad(folder/'generated.h5ad')
        depth=pd.crosstab(pd.cut(vol.obs.z_coord,bins=np.arange(0,201,10),right=False),vol.obs.cell_class)
        depth_counts.append(depth.reset_index().assign(heldout_id=i))
        pred=vol[np.abs(vol.obs.z_coord-i*50)<=5].copy()
        assert set(pred.obs.source_z)=={(i-1)*50.,(i+1)*50.}
        assert i*50. not in set(vol.obs.source_z)
        assert not vol.obs.is_observed.any()
        np.testing.assert_allclose(vol.obsm['spatial_3d'][:,:2],vol.obsm['spatial'])
        np.testing.assert_allclose(vol.obsm['spatial_3d'][:,2],vol.obs.z_coord)
        baseline_dir=ROOT/f'output/merfish_holdout/holdout_{i}'
        truth_saved=ad.read_h5ad(baseline_dir/'heldout_truth.h5ad')
        assert truth_saved.obs_names.equals(truth.obs_names)
        np.testing.assert_array_equal(dense(truth_saved),dense(truth))
        np.testing.assert_array_equal(truth_saved.obsm['spatial'],truth.obsm['spatial'])
        np.testing.assert_array_equal(truth_saved.obs.cell_class.astype(str),truth.obs.cell_class.astype(str))
        baseline=ad.read_h5ad(baseline_dir/'predicted_holdout.h5ad')
        copy_pred=parent_copy(pred,original)
        runs={BASE:baseline,NEW:pred,COPY:copy_pred}
        same_schedule=None
        if i==2:
            oldvol=ad.read_h5ad(ROOT/'HM_DeepSpital/outputs/v1_holdout_2_run2/generated.h5ad')
            old=oldvol[np.abs(oldvol.obs.z_coord-100)<=5].copy()
            runs[OLD]=old
            columns=['source_z','source_cell_id','source_cell_index','source_direction','parent_id']
            same_schedule=all(np.array_equal(pred.obs[c].astype(str),old.obs[c].astype(str)) for c in columns)
            same_schedule=bool(same_schedule and np.array_equal(pred.obs.z_coord,old.obs.z_coord))
            del oldvol
        local_tables={}
        local_summaries={}
        niche=niche_composition(truth,runs)
        spatial_rows=[]
        for width in (100,200):
            spatial_rows.extend(dict(heldout_id=i,**r) for r in spatial_expression(truth,runs,width))
        maps.extend(spatial_rows)
        spatial_frame=pd.DataFrame(spatial_rows)
        for name,a in runs.items():
            assert truth.var_names.equals(a.var_names)
            table,summary=evaluate(truth,a,train)
            summary.update(heldout_id=i,run=name,**extra_metrics(truth,a,table),**niche[name])
            for width in (100,200):
                subset=spatial_frame[(spatial_frame.run==name)&(spatial_frame.bin_width_um==width)]
                summary[f'spatial_expression_median_pcc_{width}um']=float(subset.spatial_expression_pcc.median())
                summary[f'spatial_expression_shared_bins_{width}um']=int(subset.shared_bins.iloc[0])
            if name==BASE:
                saved=json.loads((baseline_dir/'metrics.json').read_text())
                for key in saved.keys()&summary.keys():
                    if isinstance(saved[key],(int,float)):
                        np.testing.assert_allclose(summary[key],saved[key],rtol=1e-7,atol=1e-7,equal_nan=True)
            local_tables[name]=table
            local_summaries[name]=summary
            tables.append(table.assign(heldout_id=i,run=name))
            for label in table.loc[table.is_rare_train,'cell_class']:
                tm=truth.obs.cell_class.astype(str).to_numpy()==label
                pm=a.obs.cell_class.astype(str).to_numpy()==label
                d=cKDTree(a.obsm['spatial'][pm]).query(truth.obsm['spatial'][tm])[0] if pm.any() else np.full(tm.sum(),np.inf)
                for index,distance in zip(np.flatnonzero(tm),d):
                    anchors.append(dict(heldout_id=i,run=name,cell_class=label,truth_cell_id=str(truth.obs_names[index]),
                                        nearest_same_class_um=float(distance),within_50um=bool(distance<=50),within_100um=bool(distance<=100)))
            tx,px=np.log1p(np.maximum(dense(truth),0)),np.log1p(np.maximum(dense(a),0))
            ids=truth.var_names.get_indexer(['Mbp','Opalin','Ermn','Gjc3'])
            for label in sorted(train.index):
                tm=truth.obs.cell_class.astype(str).to_numpy()==label
                pm=a.obs.cell_class.astype(str).to_numpy()==label
                ts,ps=tx[tm][:,ids].mean(1),px[pm][:,ids].mean(1)
                programs.append(dict(heldout_id=i,run=name,cell_class=label,truth_n=int(tm.sum()),predicted_n=int(pm.sum()),
                                     truth_mean=float(ts.mean()),prediction_mean=float(ps.mean()) if len(ps) else np.nan,
                                     absolute_mean_error=float(abs(ps.mean()-ts.mean())) if len(ps) else np.nan))
        common=set(local_tables[BASE].loc[local_tables[BASE].predicted_n>0,'cell_class'])&set(local_tables[NEW].loc[local_tables[NEW].predicted_n>0,'cell_class'])
        for name,summary in local_summaries.items():
            table=local_tables[name]
            paired=table[table.cell_class.isin(common)]
            summary['paired_present_class_n']=len(paired)
            summary['paired_present_class_expression_pcc']=float(paired.mean_log_expression_pcc.mean())
            summary['paired_present_rare_expression_pcc']=float(paired.loc[paired.is_rare_train,'mean_log_expression_pcc'].mean())
            summaries.append(summary)
        delta=np.linalg.norm(pred.obsm['spatial']-copy_pred.obsm['spatial'],axis=1)
        for label in sorted(train.index):
            mask=pred.obs.source_cell_class.astype(str).to_numpy()==label
            traces.append(dict(heldout_id=i,cell_class=label,slab_n=int(mask.sum()),unique_slab_parents=int(pred.obs.loc[mask,'parent_id'].nunique()),
                               volume_n=int((vol.obs.cell_class.astype(str)==label).sum()),
                               slab_displacement_median_um=float(np.median(delta[mask])) if mask.any() else np.nan))
        log=pd.read_csv(folder/'checkpoints/logs/version_0/metrics.csv').dropna(subset=['loss_epoch'])
        assert list(log.epoch.astype(int))==list(range(100))
        losses.append(log.assign(heldout_id=i))
        qcs.append(dict(heldout_id=i,volume_n=vol.n_obs,slab_n=pred.n_obs,finite_expression=bool(np.isfinite(dense(vol)).all()),
                        finite_coordinates=bool(np.isfinite(vol.obsm['spatial_3d']).all()),
                        negative_expression_volume=int((dense(vol)<0).sum()),negative_expression_slab=int((dense(pred)<0).sum()),
                        zero_expression_slab_cells=int((dense(pred)==0).all(1).sum()),
                        volume_class_switches=int((vol.obs.cell_class.astype(str)!=vol.obs.source_cell_class.astype(str)).sum()),
                        median_parent_displacement_um=float(np.median(delta)),
                        old_v1_generation_schedule_identical=same_schedule,epochs=len(log),last_loss=float(log.loss_epoch.iloc[-1])))
        config_checks.append(dict(heldout_id=i,training_ids=train_ids,seed=cfg['seed'],uot_reg=cfg['data']['uot_reg'],uot_tau=cfg['data']['uot_tau'],
                                  identical_settings_across_folds=True,only_change_from_previous_v1='data.uot_reg: 0.8 -> 0.05',
                                  training_only_normalization_and_niche_statistics_verified=True))
        plot_data[i]=(truth,baseline,pred)
        print('FOLD',i,json.dumps(local_summaries[NEW]),flush=True)
        del vol
    summary=pd.DataFrame(summaries)
    table=pd.concat(tables,ignore_index=True)
    qc=pd.DataFrame(qcs)
    mean=summary[summary.run.isin([BASE,NEW,COPY])].groupby('run').agg(
        {k:['mean','std'] for k in ['rare_macro_coverage_recall_50um','rare_macro_coverage_recall_100um',
                                   'rare_macro_coverage_precision_100um','rare_micro_recall_100um',
                                   'global_mean_log_expression_pcc','paired_present_class_expression_pcc',
                                   'paired_present_rare_expression_pcc','tissue_histogram_tv','symmetric_mean_nn_um',
                                   'niche_js_divergence_mean','niche_prediction_empty_fraction',
                                   'spatial_expression_median_pcc_100um','spatial_expression_median_pcc_200um','xy_sd_ratio_mean']})
    mean.columns=['_'.join(c) for c in mean.columns]
    mean.reset_index().to_csv(DEST/'fold_mean_std.csv',index=False)
    for name,frame in [('fold_summary',summary),('cell_type_metrics',table),('quality_checks',qc),
                       ('spatial_expression_metrics',pd.DataFrame(maps)),('training_epoch_losses',pd.concat(losses,ignore_index=True)),
                       ('source_diagnostics',pd.DataFrame(traces)),('rare_cell_distances',pd.DataFrame(anchors)),
                       ('program_metrics',pd.DataFrame(programs)),('depth_cell_counts',pd.concat(depth_counts,ignore_index=True))]:
        frame.to_csv(DEST/f'{name}.csv',index=False)
    (DEST/'configuration_checks.json').write_text(json.dumps(config_checks,ensure_ascii=False,indent=2),encoding='utf-8')
    make_plots(plot_data,summary,table)
    write_report(summary,table,qc,mean,pd.DataFrame(traces))
    print('MEANS',mean.to_string(),flush=True)


def make_plots(data,summary,table):
    plt.rcParams.update({'font.family':'Times New Roman','font.size':11,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for ax,(key,title,scale) in zip(axes.flat,[
        ('tissue_histogram_tv','Tissue density error (TV; lower is better)',1),
        ('spatial_expression_median_pcc_100um','Spatial expression PCC (100 µm; higher is better)',1),
        ('rare_macro_coverage_recall_100um','Rare-class macro recall @100 µm (%)',100),
        ('xy_sd_ratio_mean','XY standard deviation / truth (target = 1)',1),
    ]):
        for shift,name,color,label in [(-.18,BASE,'#357ba8','Original'),(.18,NEW,'#d15e32','V1 reg=.05')]:
            values=summary[summary.run==name].sort_values('heldout_id')[key].to_numpy()*scale
            ax.bar(np.arange(3)+shift,values,width=.34,color=color,label=label)
            for x,value in zip(np.arange(3)+shift,values):
                ax.text(x,value,f'{value:.1f}' if scale==100 else f'{value:.3f}',ha='center',va='bottom',fontsize=9)
        ax.set_xticks(range(3),['Holdout 1','Holdout 2','Holdout 3'])
        ax.set_title(title,fontsize=11)
        ax.spines[['right','top']].set_visible(False)
        if key=='xy_sd_ratio_mean': ax.axhline(1,color='gray',lw=1,ls='--')
        ax.set_ylim(0,ax.get_ylim()[1]*1.16)
    axes[0,0].legend(frameon=False)
    for ext in ('png','svg'): fig.savefig(DEST/f'overall_comparison.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(12,11),layout='constrained')
    edges=np.arange(-1000,1001,100)
    images=[]
    for i in (1,2,3):
        images.append([np.histogram2d(*np.asarray(a.obsm['spatial']).T,bins=(edges,edges))[0]/a.n_obs*100 for a in data[i]])
    maximum=max(a.max() for row in images for a in row)
    for row,i in enumerate((1,2,3)):
        for col,(name,a) in enumerate(zip(['Truth',BASE,NEW],data[i])):
            ax=axes[row,col]
            im=ax.imshow(images[row][col].T,origin='lower',extent=(-1000,1000,-1000,1000),vmin=0,vmax=maximum,cmap='Blues')
            ax.set(title=f'Holdout {i}: {name}',xlabel='X (µm)',ylabel='Y (µm)')
    fig.colorbar(im,ax=axes,label='Cells per 100 × 100 µm bin (%)',shrink=.75)
    for ext in ('png','svg'): fig.savefig(DEST/f'tissue_density_comparison.{ext}',dpi=170)
    plt.close(fig)
    labels=table[(table.heldout_id==1)&(table.run==NEW)&table.is_rare_train].cell_class.tolist()
    fig,axes=plt.subplots(1,3,figsize=(13,4.8),layout='constrained')
    for ax,i in zip(axes,(1,2,3)):
        matrix=np.column_stack([table[(table.heldout_id==i)&(table.run==name)].set_index('cell_class').loc[labels,'coverage_recall_100um'] for name in (BASE,NEW)])*100
        im=ax.imshow(matrix,vmin=0,vmax=100,cmap='Blues',aspect='auto')
        for r in range(len(labels)):
            for c in range(2): ax.text(c,r,f'{matrix[r,c]:.1f}%',ha='center',va='center',color='white' if matrix[r,c]>60 else '#152638')
        ax.set_xticks([0,1],['Original','V1 reg=.05'])
        ax.set_yticks(range(len(labels)),labels)
        ax.set_title(f'Holdout {i}: same-type recall @100 µm')
        ax.tick_params(length=0)
    fig.colorbar(im,ax=axes,label='True cells covered (%)',shrink=.8)
    for ext in ('png','svg'): fig.savefig(DEST/f'rare_cell_coverage.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(9,12),layout='constrained')
    for row,i in enumerate((1,2,3)):
        truth,base,pred=data[i]
        for col,label in enumerate(['OD Mature 3','OD Immature 2']):
            ax=axes[row,col]
            xy=np.asarray(truth.obsm['spatial'])
            tm=truth.obs.cell_class.astype(str).to_numpy()==label
            ax.scatter(*xy.T,s=1,color='#c8cdd2',alpha=.35,rasterized=True)
            for point in xy[tm]: ax.add_patch(plt.Circle(point,100,color='#357ba8',alpha=.12,lw=0))
            ax.scatter(*xy[tm].T,s=60,facecolors='none',edgecolors='#357ba8',label=f'Truth n={tm.sum()}')
            for name,a,color,marker in [('Original',base,'#69966b','+'),('V1',pred,'#d15e32','x')]:
                pm=a.obs.cell_class.astype(str).to_numpy()==label
                ax.scatter(*np.asarray(a.obsm['spatial'])[pm].T,s=45,color=color,marker=marker,label=f'{name} n={pm.sum()}')
            ax.set(title=f'Holdout {i}: {label}',xlim=(-1000,1000),ylim=(-1000,1000),xlabel='X (µm)',ylabel='Y (µm)',aspect='equal')
            ax.legend(fontsize=9,loc='upper left')
    for ext in ('png','svg'): fig.savefig(DEST/f'rare_cell_spatial.{ext}',dpi=170)
    plt.close(fig)


def write_report(summary,table,qc,mean,traces):
    new=summary[summary.run==NEW].set_index('heldout_id')
    base=summary[summary.run==BASE].set_index('heldout_id')
    source=summary[summary.run==COPY].set_index('heldout_id')
    rows=[]
    metrics=[('组织密度误差 TV ↓','tissue_histogram_tv','.4f'),
             ('类别比例总变差 ↓','cell_fraction_total_variation','.4f'),
             ('双向平均最近邻距离 µm ↓','symmetric_mean_nn_um','.2f'),
             ('空间表达 PCC 中位数，100 µm ↑','spatial_expression_median_pcc_100um','.4f'),
             ('空间表达 PCC 中位数，200 µm ↑','spatial_expression_median_pcc_200um','.4f'),
             ('局部类型组成 JS 散度 ↓','niche_js_divergence_mean','.4f'),
             ('真实位置周围 50 µm 空预测邻域比例 ↓','niche_prediction_empty_fraction','.2%'),
             ('稀有类宏平均召回 @50 µm ↑','rare_macro_coverage_recall_50um','.2%'),
             ('稀有类宏平均召回 @100 µm ↑','rare_macro_coverage_recall_100um','.2%'),
             ('稀有类细胞数加权召回 @100 µm ↑','rare_micro_recall_100um','.2%'),
             ('全局均值表达 PCC ↑','global_mean_log_expression_pcc','.4f'),
             ('共同存在类别的表达 PCC 宏平均 ↑','paired_present_class_expression_pcc','.4f'),
             ('共同存在稀有类别的表达 PCC 宏平均 ↑','paired_present_rare_expression_pcc','.4f'),
             ('XY 标准差 / 真值，目标 1','xy_sd_ratio_mean','.4f')]
    for title,key,fmt in metrics:
        rows.append({'指标':title,'原版':format(base[key].mean(),fmt),'V1 reg=.05':format(new[key].mean(),fmt),
                     '匹配来源直接复制':format(source[key].mean(),fmt)})
    fold_rows=[]
    for i in (1,2,3):
        b,v=base.loc[i],new.loc[i]
        fold_rows.append({'留出':str(i),'真实 / 原版 / V1 数量':f'{int(v.truth_n)} / {int(b.predicted_n)} / {int(v.predicted_n)}',
                          'V1 类别数':str(int(v.predicted_classes)),
                          '密度 TV：原版 → V1':f'{b.tissue_histogram_tv:.3f} → {v.tissue_histogram_tv:.3f}',
                          '稀有类100µm召回：原版 → V1':f'{b.rare_macro_coverage_recall_100um:.2%} → {v.rare_macro_coverage_recall_100um:.2%}'})
    labels=table[(table.heldout_id==1)&(table.run==NEW)&table.is_rare_train].cell_class.tolist()
    rare_rows=[]
    for label in labels:
        row={'稀有类':label}
        for i in (1,2,3):
            b=table[(table.run==BASE)&(table.heldout_id==i)&(table.cell_class==label)].iloc[0]
            v=table[(table.run==NEW)&(table.heldout_id==i)&(table.cell_class==label)].iloc[0]
            row[f'留出{i}：原版 → V1']=f'{b.coverage_recall_100um:.1%} → {v.coverage_recall_100um:.1%}'
        rare_rows.append(row)
    old=summary[summary.run==OLD].iloc[0]
    current=new.loc[2]
    same_rows=[]
    for title,key,fmt in [('组织密度 TV ↓','tissue_histogram_tv','.4f'),
                          ('最近邻距离 µm ↓','symmetric_mean_nn_um','.2f'),
                          ('空间表达 PCC，100 µm ↑','spatial_expression_median_pcc_100um','.4f'),
                          ('类型邻域 JS ↓','niche_js_divergence_mean','.4f'),
                          ('稀有类100µm宏平均召回 ↑','rare_macro_coverage_recall_100um','.2%')]:
        same_rows.append({'留出2指标':title,'上一轮 V1 reg=.8':format(old[key],fmt),'本轮 V1 reg=.05':format(current[key],fmt)})
    report=f'''# V1 三轮逐片留出实验分析

分析日期：2026-10-27。结果目录：`HM_DeepSpital/outputs/v1_holdouts_1_2_3`。

## 结论

降低 UOT 熵正则后，三轮整体组织几何、局部空间表达和类型邻域组成均优于现有原版运行，严重的中心收缩得到明显缓解。但最稀有类别仍未恢复：OD Mature 3、OD Immature 2 三轮 100 µm 同类覆盖率全部为 0，且留出 1 的 OD Mature 3 在评价薄层整类缺失。

稀有类 100 µm 宏平均召回率为 20.30%，与原版 20.55% 基本持平；细胞数加权召回率则提高，收益主要体现在 OD Mature 1、Pericytes 等相对较多的稀有类上。当前应表述为“整体空间保真度改善，极稀有类别恢复仍不足”。

## 实验与文件核验

- 分别留出 1/2/3，训练切片分别为 0/2/3/4、0/1/3/4、0/1/2/4；每轮独立建模，seed=42，100 epochs。它们是三种切片划分，非三个随机种子。
- 三轮生成体积细胞数依次为 {', '.join(str(int(v)) for v in qc.volume_n)}。评价固定在留出 z=50/100/150 µm 的 ±5 µm 薄层，与原版协议一致；不根据隐藏真值细胞数筛选预测。
- 确认三轮配置完全相同，均为 uot_reg=.05、uot_tau=.05、balanced_fraction=.2、source_sampling=stratified、real 生态位条件和双向训练。19 维条件、hidden=256、depth=6、8 heads、batch=512、50,000 基础配对、lambda_g=.1/lambda_c=10、thickness=10、dopri5/100 输出时间点保持原设置。
- 相对上一轮 V1 reg=.8，已保存配置的唯一变化是 data.uot_reg。最终生成采用末轮模型的 EMA。个别最佳 loss checkpoint 位于 epoch 98，不影响日志完整覆盖 0–99 和末轮最终模型的保存。
- 各轮坐标归一化、类别、基因顺序及四基因程序的均值/尺度均与相应可见训练数据核对通过；留出切片未进入 source_z。
- 原版隐藏真值文件的细胞顺序、表达、标签和 XY 与原始切片一致；原版既有指标重新计算后在 1e-7 容差内一致。
- 三轮全体积表达和坐标均为有限数值，负表达元素均为 0；评价薄层没有全零表达细胞。生成 cell_class 均与对应来源类别相同。

## 三轮平均指标

{md_table(pd.DataFrame(rows))}

这里的平均对三轮切片等权。各轮标准差见 `fold_mean_std.csv`，它描述切片划分差异，不能当作多种子显著性。

表达 PCC 是跨基因的均值向量相关性，不是逐细胞预测准确率。留出 1 有一个类别无预测，其 PCC 未定义；为了公平比较，“共同存在类别”在原版与 V1 中使用同一类别集合（14/15/15 类，稀有类为 5/6/6 类）。缺失类别在覆盖召回率中仍计 0，不会被剔除。CSV 中的宏平均覆盖 precision 沿用原协议，仅在有预测的类别上计算；留出 1 的 V1 分母为 5 个稀有类，不能把它当作六类均有预测时的直接对等比较。

宏平均覆盖对六个稀有类等权；细胞数加权覆盖对真实细胞加权。若把三轮稀有真实细胞合并，原版为 118/372=31.72%，V1 为 134/372=36.02%；这是覆盖计数，不是细胞识别数，多个真实点可以共享同一预测邻居。

“匹配来源直接复制”固定新 V1 每一个生成来源、数量、深度和类别，保留来源 XY/表达，不做运输；未使用隐藏真值构造。它是诊断对照，不是完整三维重建方法。

![主要指标](overall_comparison.png)

## 各轮表现

{md_table(pd.DataFrame(fold_rows))}

三轮密度误差都下降，三轮空间表达都提高；稀有类覆盖在留出 1 下降、留出 2 小幅上升、留出 3 上升，不能称为跨切片稳定提高。

## 空间收缩已明显缓解

平均密度 TV 从 0.3018 降为 0.1711，约下降 43.3%；双向平均最近邻距离由 15.51 降为 12.63 µm，约下降 18.5%。XY 标准差相对真值的比例由平均 .774 升至 .945。新 V1 仍有约 5.5% 的平均尺度缺口，不能说完全恢复。

新 V1 三轮 X/Y 标准差分别约 473/470、476/458、469/478 µm；对应真值约 503/478、502/500、501/507 µm。每轮生成相对 parent 的位移中位数约 42.0、46.9、45.2 µm。

![三轮组织密度](tissue_density_comparison.png)

各图采用同一空间范围、100×100 µm 网格和色标，按组内细胞数归一化。当前组织范围明显比原版接近真值。

## 留出 2 与上一轮 V1：支持配对熵导致收缩的判断

两次留出 2 的生成数均为 5,632；已逐细胞确认 source_z、source_cell_id、source_cell_index、source_direction、parent_id 和目标 z_coord 完全相同。因而改进不能解释为换了一批来源或目标深度。

{md_table(pd.DataFrame(same_rows))}

原先 X/Y 标准差约 358/344 µm，本轮约 476/458 µm；parent 位移中位数由 239.4 降至 46.9 µm。结合先前配对中点诊断和本轮实际重建，证据支持高熵分散配对是空间收缩的主要原因。此比较尚未包含多随机种子和服务器代码哈希核验，不把它表述为全部误差来源的唯一证明。

空间表达和邻域 JS 均在每轮所有对照共同的有效位置上重算，因此本报告与此前报告的相关性/JS 数值可能略有差异；共同支持范围见 CSV。密度 TV 和同类覆盖使用完整评价薄层。

## 稀有类仍是主要缺口

{md_table(pd.DataFrame(rare_rows))}

OD Mature 1 三轮覆盖均上升；Pericytes 在留出 2/3 明显提高。Endothelial 2 的留出 1/2 下降，OD Mature 4 未有一致改善。OD Immature 2 的类均值表达 PCC 三轮为 .534/.656/.755，均低于原版 .686/.710/.793。

![稀有类覆盖](rare_cell_coverage.png)

![关键稀有类位置](rare_cell_spatial.png)

稀有类真值在训练集占比 <1%，名单只根据可见训练切片定义。OD Mature 3 三轮真值共 8 个，本轮预测薄层共 8 个，但分布为 0/5/3，真值为 3/2/3，所有真实点 100 µm 内均无同类预测。OD Immature 2 真值共 17 个，预测也共 17 个（6/6/5），仍全部零覆盖。总数量接近不等于位置恢复。

### 原因一：薄层中的整类缺失来自深度抽样

留出 1 的全体积中有 45 个 OD Mature 3；夹住隐藏层的 z=0–100 区间中有 10 个。但这些细胞的生成 z 约为 33.96、64.00、76.96、78.42、78.71、86.49、88.52、88.75、88.81、95.73 µm，没有落在 z=45–55 的评价薄层。该类并非从全体积完全消失。

当前分层抽样按整个区间/方向的类别预算选来源，然后与随机目标深度组合，不能保证每个薄层有该稀有类；全体积生成的类别又没有离开来源类别。因此这是薄层来源/深度分配导致的缺失，不能归因于评价时标签预测失败。

### 原因二：整体位移已变小，极稀有类仍被长距离运输

OD Immature 2 三轮 parent 位移中位数约 418/202/343 µm；OD Mature 3 在有预测的留出 2/3 中约为 186/178 µm。相比全体 42–47 µm 的中位位移，这些稀有类依然明显异常。

这与“同类候选很少，即使熵正则降低，也只能在很远的端点间选择”的解释一致，尚需逐类配对距离与生态位相容性进一步核实。只降低一个全局正则参数没有解决稀有类局部匹配。

### 原因三：运输相对保留来源仍未形成稳定稀有类收益

匹配来源直接复制的三轮 100 µm 稀有类宏平均覆盖为 27.50%/14.14%/34.08%，平均 25.24%，高于 V1 的 20.30%。V1 只在留出 2 优于此对照，留出 1/3 均较差。密度 TV 方面，来源复制平均 .1450，V1 .1711。

它不能替代完整体积重建，但说明当前运输仍会破坏部分可见来源中已有的局部结构。来源复制对 OD Mature 3 三轮覆盖也均为 0，OD Immature 2 在前两轮也为 0，提示极少数细胞的真实中间位置可能缺少充分的两侧观测支持。

## 下一步建议

1. **保留 reg=.05 作为后续参照。** 本轮确认了几何收益，回到 .8 缺乏依据；当前还不能单独声称生态位模块有效，因为对比原版同时改变了多个模块及配对正则。
2. **优先处理深度×类型的联合来源抽样。** 在预先定义的深度薄层内按可见切片插值的类别期望比例分配预算，再按方向和 parent 均衡抽样；保持总密度和期望类别比例，不根据隐藏真值设置配额，不强制零期望类型出现。单独验证是否减少留出 1 这类随机漏采。
3. **针对稀有类限制不可信远距离匹配。** 按类别记录配对距离、唯一 parent 和有效候选数；测试物理尺度软成本、局部候选图及生态位相容性成本。没有可信局部匹配时允许 unmatched/低置信度分支，不为了保住类预算而强配远端点。
4. **做 real/zero/shuffle 同容量对照。** 固定 reg、抽样调度和训练端点，检查生态位条件对稀有类、空间表达和邻域组成的独立贡献。当前结果不足以把改善归功于该条件。
5. **在冻结候选后补多种子和未参与选择的测试。** 三轮均 seed=42；holdout 2 已用于开发诊断，整个数据集目前也已被查看，不能把现有三轮当作全新的独立最终测试。继续调参应使用训练侧内部验证或另行预留样本。

不优先增加网络规模、单纯增加稀有类总数或只做 V2 表达修正；目前有明确的深度分配和稀有类位置问题，应先解决这两条路径。

## 输出与复算

`fold_summary.csv`、`fold_mean_std.csv` 为整体指标；`cell_type_metrics.csv` 为逐类指标；`rare_cell_distances.csv` 为每个真实稀有细胞的同类最近邻；`source_diagnostics.csv` 为生成来源和位移；`depth_cell_counts.csv` 为每 10 µm 的类别数；`spatial_expression_metrics.csv`、`program_metrics.csv` 为分子分布；`quality_checks.csv`、`configuration_checks.json`、`training_epoch_losses.csv` 为核验与日志。

源码：`../../../../ccfa-workfiles/checks/merfish-v1/source/analyze_three_holdouts.py`，复用同目录 `analyze_v1.py` 的指标辅助函数。原始模型、配置和 H5AD 均未修改。
'''
    assert '\ufffd' not in report
    (DEST/'report.md').write_text(report,encoding='utf-8')
    assert (DEST/'report.md').read_text(encoding='utf-8')==report


if __name__=='__main__':
    main()
