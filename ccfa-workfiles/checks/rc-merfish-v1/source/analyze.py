"""Read-only audit and post-hoc diagnosis of the completed RC MERFISH experiment.

Writes derived evidence into this analysis directory; never changes run outputs.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parents[1]
RUN=ROOT/'RC_DeepSpatial/outputs/merfish_v1'
sys.path.insert(0,str(ROOT))
from compare_spatialz import read_h5ad,vector

ARMS=['baseline','control','R0','A','B','AB','ABC']
F1='model_class_rare_macro_one_to_one_f1_50um'
UF1='model_class_ultra_rare_macro_one_to_one_f1_50um'
PF1='expression_probe_rare_macro_one_to_one_f1_50um'
CF1='model_class_common_macro_one_to_one_f1_50um'
KEYS=['global_provided_X_mse','global_log1p_mse','global_log1p_pcc',F1,UF1,PF1,CF1,'density_tv']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def extra(path):
    with h5py.File(path,'r') as f:
        return {k:vector(f['obs'][k]) for k in ('source_cell_index','source_direction','source_cell_class',
                                               'target_cell_index','z_coord','generation_mode')}


def match_table(truth,xy,labels,state):
    rows=[]
    total=sum(state['counts'])
    for c,count in zip(state['classes'],state['counts']):
        a,b=truth['xy'][truth['labels']==c],xy[labels==c]
        if len(a) and len(b):
            edges=cKDTree(a).sparse_distance_matrix(cKDTree(b),50,output_type='coo_matrix')
            graph=sparse.csr_matrix((np.ones(len(edges.data)),(edges.row,edges.col)),shape=(len(a),len(b)))
            n=int((maximum_bipartite_matching(graph,perm_type='column')>=0).sum())
        else:n=0
        rows.append(dict(cell_class=c,train_n=count,is_rare=count/total<=.05,is_ultra_rare=count/total<=.01,
                         truth_n=len(a),predicted_n=len(b),f1_50um=2*n/(len(a)+len(b)) if len(a)+len(b) else np.nan))
    return pd.DataFrame(rows)


def macro(table,group):
    mask=(table.truth_n>0)&table[group]
    return float(table.loc[mask,'f1_50um'].mean())


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((RUN/'experiment.json').read_text())
    full=pd.read_csv(RUN/'all_runs.csv')
    assert len(full)==21 and set(full.arm)==set(ARMS)
    assert full.groupby('heldout').size().to_dict()=={1:7,2:7,3:7}
    assert set(full.seed)=={42}
    audit=[];source_status=[];per_type=[];sampling=[];marker=[];cost=[];counter=[];movement=[];anchor=[];recomputed=[]
    inputs={int(p.stem.rsplit('_',1)[1]):p for p in (ROOT/'data/merfish_mouse_hypothalamus').rglob('merfish_[0-4].h5ad')}
    for name,expected in cfg['source_sha256'].items():
        path=ROOT/'RC_DeepSpatial'/name
        exact=sha(path)==expected
        normalized=hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==expected
        assert exact or normalized,(name,'source differs beyond CRLF')
        source_status.append(dict(file=name,exact=exact,lf_normalized=normalized))
    for h in (1,2,3):
        fold=RUN/f'holdout_{h}/seed_42'
        state=json.loads((fold/'shared/state.json').read_text())
        manifest=json.loads((fold/'manifest.json').read_text())
        assert manifest['config']==cfg['config'] and manifest['source_sha256']==cfg['source_sha256']
        for i,item in manifest['visible_inputs'].items():
            assert sha(inputs[int(i)])==item['sha256']
        stamp=json.loads((fold/'shared/evaluation_truth.json').read_text())
        assert sha(inputs[h])==stamp['sha256']
        truth=read_h5ad(inputs[h]);left=read_h5ad(inputs[h-1]);right=read_h5ad(inputs[h+1])
        common_schedule=None
        for arm in ARMS:
            base=fold/arm
            trained=json.loads((base/'trained.json').read_text());generated=json.loads((base/'generated.json').read_text())
            assert trained['epochs']==100
            hashes={k:sha(base/name)==generated[k] for k,name in [('gap_sha256','predicted_gap.h5ad'),('slab_sha256','predicted_holdout.h5ad')]}
            hashes['checkpoint_sha256']=sha(base/'last.pt')==trained['checkpoint_sha256']
            assert all(hashes.values()),(h,arm,hashes)
            learning=pd.read_csv(base/'training.csv');assert learning.epoch.tolist()==list(range(1,101))
            metrics=json.loads((base/'metrics.json').read_text())
            saved=full[(full.heldout==h)&(full.arm==arm)].iloc[0]
            for k,v in metrics.items():
                if isinstance(v,(int,float)) and v is not None:assert np.isclose(saved[k],v,rtol=1e-12,atol=1e-12),(h,arm,k)
            if arm!='baseline':
                if common_schedule is None:common_schedule=generated['schedule_sha256']
                assert common_schedule==generated['schedule_sha256']
            if arm not in ('baseline','control'):
                control=json.loads((fold/'control/trained.json').read_text())
                assert trained['base_checkpoint_sha256']==control['checkpoint_sha256']
            pred=read_h5ad(base/'predicted_holdout.h5ad');obs=extra(base/'predicted_holdout.h5ad')
            assert np.array_equal(pred['genes'],truth['genes']) and len(pred['x'])==metrics['predicted_n']
            table=match_table(truth,pred['xy'],pred['labels'],state)
            f1=macro(table,'is_rare');uf1=macro(table,'is_ultra_rare')
            assert np.isclose(f1,metrics[F1],atol=1e-12)
            assert np.isclose(uf1,metrics[UF1],atol=1e-12)
            gt=truth['x'].astype(np.float32);gp=pred['x'].astype(np.float32)
            raw_mse=float(np.mean((gt.mean(0).astype(float)-gp.mean(0).astype(float))**2))
            assert np.isclose(raw_mse,metrics['global_provided_X_mse'],atol=1e-12)
            recomputed.append(dict(heldout=h,arm=arm,rare_f1=f1,ultra_f1=uf1,raw_mse=raw_mse))
            audit.append(dict(heldout=h,arm=arm,epochs=len(learning),hashes_ok=all(hashes.values()),
                              source_class_disagreements=int((pred['labels']!=obs['source_cell_class']).sum()),
                              source_anchor_n=int((obs['generation_mode']=='source_anchor').sum())))
            per_type.append(pd.read_csv(base/'per_type.csv').assign(heldout=h,arm=arm))
            sampling.append(pd.read_csv(base/'trajectory_sampling.csv').assign(heldout=h,arm=arm))
            marker.append(pd.read_csv(base/'marker_sparsification.csv').assign(heldout=h,arm=arm))
            cost.append(dict(heldout=h,arm=arm,training_seconds=trained['elapsed_seconds'],generation_seconds=generated['elapsed_seconds'],
                             trainable_parameters=trained['trainable_parameters']))
            forward=obs['source_direction']=='forward';parent=obs['source_cell_index'].astype(int)
            src=np.empty_like(pred['xy']);dest=np.empty_like(src)
            for mask,s,d in ((forward,left,right),(~forward,right,left)):
                src[mask]=s['xy'][parent[mask]]
                dest[mask]=d['xy'][np.maximum(obs['target_cell_index'][mask],0)]
            dis=np.linalg.norm(pred['xy']-src,axis=1)
            movement.append(dict(heldout=h,arm=arm,mean_um=dis.mean(),median_um=np.median(dis),p95_um=np.quantile(dis,.95)))
            for c in state['classes']:
                mask=pred['labels']==c
                anchor.append(dict(heldout=h,arm=arm,cell_class=c,generated_n=int(mask.sum()),
                                   source_anchor_n=int(((obs['generation_mode']=='source_anchor')&mask).sum())))
            if arm=='control':
                ratio=np.where(forward,(obs['z_coord']-50*(h-1))/100,(50*(h+1)-obs['z_coord'])/100)
                ot=src+(dest-src)*ratio[:,None];ot[obs['target_cell_index']<0]=src[obs['target_cell_index']<0]
                for name,xy in [('source_xy_only',src),('control',pred['xy']),('sampled_ot_linear',ot)]:
                    ctable=match_table(truth,xy,pred['labels'],state)
                    counter.append(dict(heldout=h,variant=name,rare_f1_50um=macro(ctable,'is_rare'),ultra_f1_50um=macro(ctable,'is_ultra_rare')))
    for name,records in [('audit',audit),('source_audit',source_status),('recomputed_metrics',recomputed),
                         ('cost',cost),('source_movement',movement),('anchor_counts',anchor),('spatial_counterfactuals',counter)]:
        pd.DataFrame(records).to_csv(OUT/f'{name}.csv',index=False)
    ptype=pd.concat(per_type,ignore_index=True);ptype.to_csv(OUT/'per_type.csv',index=False)
    samples=pd.concat(sampling,ignore_index=True);samples.to_csv(OUT/'sampling.csv',index=False)
    markers=pd.concat(marker,ignore_index=True);markers['drop']=markers.positive_before-markers.positive_after
    markers.to_csv(OUT/'marker_diagnostics.csv',index=False)
    means=full.groupby('arm')[KEYS].mean().reindex(ARMS);means.to_csv(OUT/'main_means.csv')
    deltas=[]
    for reference in ('baseline','control','R0','A','AB'):
        for arm in ARMS:
            if arm==reference:continue
            for h in (1,2,3):
                a=full[(full.heldout==h)&(full.arm==arm)].iloc[0]
                b=full[(full.heldout==h)&(full.arm==reference)].iloc[0]
                row=dict(heldout=h,arm=arm,reference=reference)
                for key in KEYS:
                    row[key]=a[key]-b[key]
                    if 'mse' in key:row[key+'_relative_percent']=100*(a[key]/b[key]-1)
                deltas.append(row)
    pd.DataFrame(deltas).to_csv(OUT/'paired_deltas.csv',index=False)
    cls=ptype[(ptype.label_view=='model_class')&ptype.is_rare]
    heat=cls.groupby(['cell_class','arm']).one_to_one_f1_50um.mean().unstack().reindex(columns=ARMS)*100
    heat.to_csv(OUT/'rare_class_f1_percent.csv')
    plots(full,means,heat,inputs)
    additional(inputs)
    print('Verified 21/21 runs: 100 epochs, input/result/checkpoint hashes, numeric summaries, rare F1 and raw MSE.')
    print('Counterfactual means:\n',pd.DataFrame(counter).groupby('variant')[['rare_f1_50um','ultra_f1_50um']].mean().to_string())
    print('Class changes:',sum(x['source_class_disagreements'] for x in audit))
    print('Total training hours:',sum(x['training_seconds'] for x in cost)/3600)


def additional(inputs):
    """Post-hoc label-permutation reference and actual endpoint displacement."""
    null_rows=[];pair_rows=[]
    for h in (1,2,3):
        fold=RUN/f'holdout_{h}/seed_42'
        state=json.loads((fold/'shared/state.json').read_text())
        truth=read_h5ad(inputs[h]);pred=read_h5ad(fold/'control/predicted_holdout.h5ad')
        observed=match_table(truth,pred['xy'],pred['labels'],state)
        rare=observed[observed.is_rare&(observed.truth_n>0)]
        rng=np.random.default_rng(20261101+h)
        null={c:[] for c in rare.cell_class}
        for _ in range(100):
            labels=rng.permutation(pred['labels'])
            for c in null:
                a,b=truth['xy'][truth['labels']==c],pred['xy'][labels==c]
                edges=cKDTree(a).sparse_distance_matrix(cKDTree(b),50,output_type='coo_matrix')
                graph=sparse.csr_matrix((np.ones(len(edges.data)),(edges.row,edges.col)),shape=(len(a),len(b)))
                n=int((maximum_bipartite_matching(graph,perm_type='column')>=0).sum())
                null[c].append(2*n/(len(a)+len(b)))
        for row in rare.itertuples():
            values=np.asarray(null[row.cell_class])
            null_rows.append(dict(heldout=h,cell_class=row.cell_class,truth_n=row.truth_n,predicted_n=row.predicted_n,
                                  observed_f1=row.f1_50um,null_mean=float(values.mean()),
                                  null_q025=float(np.quantile(values,.025)),null_q975=float(np.quantile(values,.975)),
                                  null_zero_fraction=float((values==0).mean()),is_ultra_rare=row.is_ultra_rare))
        for arm in ('baseline','control','A'):
            with np.load(fold/arm/'trajectory_indices.npz') as f:
                index={k:f[k] for k in f.files}
            distance=np.empty(len(index['cell0']))
            for src_i in np.unique(index['slice0']):
                for dst_i in np.unique(index['slice1'][index['slice0']==src_i]):
                    mask=(index['slice0']==src_i)&(index['slice1']==dst_i)
                    src,dst=read_h5ad(inputs[int(src_i)]),read_h5ad(inputs[int(dst_i)])
                    delta=dst['xy'][index['cell1'][mask]]-src['xy'][index['cell0'][mask]]
                    distance[mask]=np.linalg.norm(delta,axis=1)
            pair_rows.append(dict(heldout=h,arm=arm,mean_um=distance.mean(),median_um=np.median(distance),p95_um=np.quantile(distance,.95)))
    pd.DataFrame(null_rows).to_csv(OUT/'label_permutation_reference.csv',index=False)
    pd.DataFrame(pair_rows).to_csv(OUT/'training_endpoint_distances.csv',index=False)
    print('Extreme rare permutation reference:\n',pd.DataFrame(null_rows).query('cell_class in ["OD Immature 2","OD Mature 3"]')[['heldout','cell_class','observed_f1','null_mean','null_zero_fraction']].to_string(index=False))
    print('Training endpoint distances:\n',pd.DataFrame(pair_rows).groupby('arm')[['mean_um','median_um','p95_um']].mean().to_string())


def plots(full,means,heat,inputs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Times New Roman','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#59636d','#276f8a','#a0a6aa','#609a77','#ae78a0','#cc9a42','#c65a43']
    fig,axes=plt.subplots(2,2,figsize=(12,7.6),layout='constrained')
    for ax,key,title,scale in zip(axes.ravel(),['global_provided_X_mse',F1,UF1,'global_log1p_mse'],
                                 ['A  Global mean-profile MSE (provided X), lower is better',
                                  'B  Rare spatial F1 at 50 um, higher is better',
                                  'C  Ultra-rare spatial F1 at 50 um, higher is better',
                                  'D  Global mean-profile MSE (log1p), lower is better'],[1,100,100,1]):
        x=np.arange(len(ARMS));ax.bar(x,means[key]*scale,color=colors,alpha=.85,width=.7)
        for h,marker,offset in [(1,'o',-.12),(2,'s',0),(3,'^',.12)]:
            values=full[full.heldout==h].set_index('arm').reindex(ARMS)[key]*scale
            ax.scatter(x+offset,values,s=22,c='#202630',marker=marker,label=f'Fold {h}',zorder=3)
        ax.set_xticks(x,ARMS);ax.set_title(title,loc='left',fontsize=11,pad=11)
        ax.set_ylabel('F1 (%)' if scale==100 else 'MSE');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
    axes[0,0].legend(frameon=False,ncol=3,fontsize=9)
    fig.suptitle('MERFISH: three holdouts, one seed (42)\nBars are fold means; points are individual folds, not independent biological replicates',fontsize=14)
    fig.savefig(OUT/'overview.png',dpi=180);plt.close(fig)
    skill=Path(os.environ.get('CODEX_HOME',Path.home()/'.codex'))/'skills/ccf-visual-composer/resources/python/ccfa_plot_recipes.py'
    if skill.exists():
        spec=importlib.util.spec_from_file_location('ccfa_plot_recipes',skill);module=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=module;spec.loader.exec_module(module)
        (OUT/'rare_class_heatmap.svg').write_text(module.heatmap_matrix(
            heat.values.tolist(),heat.index.tolist(),ARMS,'Rare-cell spatial F1 at 50 um (%)',
            subtitle='Mean over 3 holdouts; source label view; includes zero-recovery classes',
            note='Rare is defined from visible training abundance <= 5%.',theme=module.Theme(width=1100,height=850)),encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(13,4.7),layout='constrained')
    label='OD Immature 2'
    for ax,h in zip(axes,(1,2,3)):
        truth=read_h5ad(inputs[h]);tm=truth['labels']==label
        ax.scatter(truth['xy'][:,0],truth['xy'][:,1],s=1,color='#d5d9dc',alpha=.3,rasterized=True)
        ax.scatter(*truth['xy'][tm].T,s=95,marker='*',color='#15191d',label='Measured truth',zorder=5)
        for arm,col,mark in [('control','#276f8a','o'),('ABC','#c65a43','x')]:
            p=read_h5ad(RUN/f'holdout_{h}/seed_42/{arm}/predicted_holdout.h5ad');pm=p['labels']==label
            ax.scatter(*p['xy'][pm].T,s=60,marker=mark,color=col,facecolors='none' if mark=='o' else col,label=arm,zorder=4)
        ax.set_title(f'Fold {h}: true {int(tm.sum())} cells');ax.set_aspect('equal');ax.set_xlabel('X (um)');ax.set_ylabel('Y (um)')
    axes[0].legend(frameon=False,fontsize=9)
    fig.suptitle('OD Immature 2: generated counts do not establish spatial recovery\nControl and ABC have zero one-to-one F1 within 50 um in all three holdouts',fontsize=13)
    fig.savefig(OUT/'rare_spatial_failure.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
