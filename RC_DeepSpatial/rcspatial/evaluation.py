"""Unpaired point-cloud evaluation, frozen expression probe and mechanism diagnostics."""
import numpy as np
import pandas as pd
import warnings
from scipy import sparse
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning

from .data import dense


def expression_metrics(a,b):
    a,b=np.asarray(a,float).ravel(),np.asarray(b,float).ravel()
    return dict(mse=float(np.mean((a-b)**2)),mae=float(np.mean(np.abs(a-b))),
                pcc=float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else np.nan)


def fit_probe(slices,state,seed):
    x=np.log1p(np.vstack([s['g'] for s in slices]))
    y=np.concatenate([s['y'] for s in slices])
    classifier=(LogisticRegression(max_iter=3000,C=1.,class_weight='balanced',random_state=seed)
                if len(np.unique(y))>1 else DummyClassifier(strategy='most_frequent'))
    probe=make_pipeline(StandardScaler(),classifier)
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        probe.fit(x,y)
    # Diagnostic markers from training only; never treated as independent biological truth.
    markers={}
    for c in range(len(state['classes'])):
        mask=y==c
        if mask.any() and (~mask).any():
            contrast=x[mask].mean(0)-x[~mask].mean(0)
            markers[c]=np.argsort(-contrast,kind='stable')[:min(5,len(contrast))].tolist()
        else:
            markers[c]=[]
    return probe,markers


def distances(a,b):
    return cKDTree(b).query(a)[0] if len(a) and len(b) else np.full(len(a),np.inf)


def type_metrics(tx,px,ty,py,tg,pg,state,config,label_view):
    cfg=config['evaluation'];counts=dict(zip(state['classes'],state['counts']))
    rows=[]
    for c in sorted(set(counts)|set(ty)|set(py)):
        tm,pm=ty==c,py==c
        nt,npred=int(tm.sum()),int(pm.sum())
        a,b=tx[tm],px[pm]
        t2p,p2t=distances(a,b),distances(b,a)
        ntrain=int(counts.get(c,0));fraction=ntrain/sum(counts.values())
        row=dict(label_view=label_view,cell_class=c,train_n=ntrain,train_fraction=fraction,
                 is_rare=0<fraction<=cfg['rare_fraction'],is_ultra_rare=0<fraction<=.01,
                 unseen_in_training=ntrain==0,truth_n=nt,predicted_n=npred,
                 missing=nt>0 and npred==0,false_presence=nt==0 and npred>0,
                 fraction_error=npred/max(1,len(py))-nt/len(ty))
        for radius in cfg['radii_um']:
            recall=float((t2p<=radius).mean()) if nt else np.nan
            precision=float((p2t<=radius).mean()) if npred else (0. if nt else np.nan)
            key=f'{radius:g}um'
            row[f'recall_{key}']=recall;row[f'precision_{key}']=precision
            row[f'f1_{key}']=2*precision*recall/(precision+recall) if precision+recall>0 else 0. if nt else np.nan
            edges=cKDTree(a).sparse_distance_matrix(cKDTree(b),radius,output_type='coo_matrix') if nt and npred else None
            if edges is not None:
                graph=sparse.csr_matrix((np.ones(len(edges.data)),(edges.row,edges.col)),shape=(nt,npred))
                matched=int((maximum_bipartite_matching(graph,perm_type='column')>=0).sum())
            else:
                matched=0
            row[f'one_to_one_f1_{key}']=2*matched/(nt+npred) if nt+npred else np.nan
        for name,value in expression_metrics(tg[tm].mean(0),pg[pm].mean(0)).items() if nt and npred else []:
            row[f'mean_log_expression_{name}']=value
        rows.append(row)
    table=pd.DataFrame(rows)
    summary=dict(missing_types=int(table.missing.sum()),false_present_types=int(table.false_presence.sum()),
                 fraction_tv=float(table.fraction_error.abs().sum()/2),
                 unseen_truth_cells=int(table.loc[table.unseen_in_training,'truth_n'].sum()))
    for group,mask in (('all',table.truth_n>0),('rare',table.is_rare&(table.truth_n>0)),
                       ('ultra_rare',table.is_ultra_rare&(table.truth_n>0)),
                       ('common',~table.is_rare&~table.unseen_in_training&(table.truth_n>0))):
        part=table[mask]
        summary[f'{group}_types']=len(part)
        summary[f'{group}_missing_types']=int(part.missing.sum())
        for column in table:
            if column.startswith(('recall_','precision_','f1_','one_to_one_','mean_log_expression_')):
                summary[f'{group}_macro_{column}']=float(part[column].mean()) if len(part) else np.nan
    return table,summary


def evaluate(truth,prediction,state,config,probe,markers):
    if truth.var_names.tolist()!=state['genes'] or prediction.var_names.tolist()!=state['genes']:
        raise ValueError('Evaluation requires the frozen training gene order.')
    if not truth.n_obs or not prediction.n_obs:
        raise ValueError('Truth/prediction slab is empty; cannot evaluate a different slab silently.')
    tx,px=np.asarray(truth.obsm[config['data']['spatial_key']]),np.asarray(prediction.obsm['spatial'])
    raw_t,raw_p=dense(truth),dense(prediction)
    if not np.isfinite(raw_t).all() or not np.isfinite(raw_p).all() or not np.isfinite(px).all():
        raise ValueError('Evaluation contains nonfinite values.')
    tg,pg=np.log1p(np.maximum(raw_t,0)),np.log1p(np.maximum(raw_p,0))
    ty=truth.obs[config['data']['label_key']].astype(str).to_numpy()
    model_y=prediction.obs.cell_class.astype(str).to_numpy()
    probe_y=np.asarray(state['classes'])[probe.predict(pg).astype(int)]
    summary=dict(truth_n=truth.n_obs,predicted_n=prediction.n_obs,
                 negative_fraction=float((raw_p<0).mean()),
                 symmetric_nn_um=float((distances(tx,px).mean()+distances(px,tx).mean())/2),
                 unique_parents=int(prediction.obs.parent_id.nunique()),
                 source_anchor_fraction=float((prediction.obs.generation_mode=='source_anchor').mean()))
    for scale,t,p in (('provided_X',raw_t,raw_p),('log1p',tg,pg)):
        summary.update({f'global_{scale}_{k}':v for k,v in expression_metrics(t.mean(0),p.mean(0)).items()})
    tables=[]
    for view,py in (('model_class',model_y),('expression_probe',probe_y)):
        table,metrics=type_metrics(tx,px,ty,py,tg,pg,state,config,view)
        tables.append(table);summary.update({view+'_'+k:v for k,v in metrics.items()})
    # Fixed physical bins: density TV includes empty/missing predicted regions.
    width=config['evaluation']['spatial_bin_um']
    bins=np.floor(np.vstack((tx,px))/width).astype(np.int64)
    _,ids=np.unique(bins,axis=0,return_inverse=True)
    nt=np.bincount(ids[:len(tx)],minlength=ids.max()+1)
    npred=np.bincount(ids[len(tx):],minlength=len(nt))
    summary['density_tv']=float(np.abs(nt/len(tx)-npred/len(px)).sum()/2)
    before=np.asarray(prediction.layers['expression_before_sparsification'])
    marker_rows=[]
    for c,name in enumerate(state['classes']):
        keep=model_y==name;cols=markers[c]
        marker_rows.append(dict(cell_class=name,generated_n=int(keep.sum()),
                                train_derived_marker_genes=';'.join(state['genes'][i] for i in cols),
                                positive_before=float((before[np.ix_(keep,cols)]>0).mean()) if keep.any() and cols else np.nan,
                                positive_after=float((raw_p[np.ix_(keep,cols)]>0).mean()) if keep.any() and cols else np.nan))
    return pd.concat(tables,ignore_index=True),summary,pd.DataFrame(marker_rows)
