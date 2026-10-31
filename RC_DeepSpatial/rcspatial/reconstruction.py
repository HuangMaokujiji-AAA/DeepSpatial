"""Visible-only generation with replayable natural-abundance source/target schedules."""
import anndata as ad
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torchdiffeq import odeint

from .vendor.sampling import depth_type_source_schedule
from .model import BaseFlow


def make_schedule(left, right, plan, config, seed, stratified=True):
    cfg=config['reconstruction']
    total=max(1,int((len(left['g'])+len(right['g']))/2*(right['z_um']-left['z_um'])/cfg['thickness_um']))
    rng=np.random.default_rng(seed)
    if stratified:
        out=depth_type_source_schedule(left['y'],right['y'],left['z_um'],right['z_um'],total,
                                       cfg['slab_width_um'],cfg['grid_origin_um'],rng)
    else:
        u=rng.random(total)
        n0,n1=len(left['g']),len(right['g'])
        # Stable inverse of the original linear cell-density CDF.
        t=u*(n0+n1)/(n0+np.sqrt(n0*n0+u*(n1*n1-n0*n0)))
        forward=rng.random(total)>t
        parent=np.empty(total,dtype=np.int64)
        parent[forward]=rng.integers(n0,size=forward.sum())
        parent[~forward]=rng.integers(n1,size=(~forward).sum())
        z=left['z_um']+t*(right['z_um']-left['z_um'])
        out=dict(t=t,is_forward=forward,source_index=parent,
                 source_class=np.where(forward,left['y'][np.minimum(parent,n0-1)],right['y'][np.minimum(parent,n1-1)]),
                 depth_bin_index=np.floor((z-cfg['grid_origin_um']+cfg['slab_width_um']/2)/cfg['slab_width_um']).astype(np.int64),
                 budget=[],bin_width_um=cfg['slab_width_um'],grid_origin_um=cfg['grid_origin_um'])
    target=np.full(total,-1,dtype=np.int64)
    support_count=np.zeros(total,dtype=np.int64)
    # Conditional targets use the original local plan in EVERY residual arm;
    # A changes training marginals, not inference candidates or their budget.
    for forward in (True,False):
        matrix=(plan if forward else plan.T).tocsr()
        children=np.flatnonzero(out['is_forward']==forward)
        for src in np.unique(out['source_index'][children]):
            ids=children[out['source_index'][children]==src]
            start,end=matrix.indptr[src:src+2]
            weights=matrix.data[start:end]
            if weights.sum()>0:
                target[ids]=rng.choice(matrix.indices[start:end],len(ids),p=weights/weights.sum())
                support_count[ids]=end-start
    out.update(target_index=target,support_count=support_count)
    return out


def save_schedule(path, schedule):
    arrays={k:v for k,v in schedule.items() if isinstance(v,np.ndarray)}
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream,**arrays)
    os.replace(tmp,path)


def load_schedule(path):
    with np.load(path,allow_pickle=False) as archive:
        return {k:archive[k].copy() for k in archive.files}


@torch.no_grad()
def reconstruct(model,left,right,schedule,state,config,device,anchor_unsupported=True):
    """Solve per unique (source,target) condition; no hidden cell counts or expression."""
    model=model.to(device).eval()
    cfg=config['reconstruction']
    total=len(schedule['t'])
    gd,cd=len(state['genes']),len(state['classes'])
    x=np.empty((total,2),np.float32);g=np.empty((total,gd),np.float32)
    continuous=np.empty((total,cd),np.float32);labels=np.empty(total,np.int64)
    times=torch.linspace(0,1,cfg['steps'],device=device)
    for forward in (True,False):
        src,dst=(left,right) if forward else (right,left)
        ids=np.flatnonzero(schedule['is_forward']==forward)
        parents=schedule['source_index'][ids]
        targets=schedule['target_index'][ids]
        anchored=(targets<0) if anchor_unsupported else np.zeros(len(ids),bool)
        aid,ap=ids[anchored],parents[anchored]
        x[aid],g[aid]=src['x'][ap],src['g'][ap]
        labels[aid]=src['y'][ap]
        continuous[aid]=np.eye(cd,dtype=np.float32)[labels[aid]]
        ids,parents,targets=ids[~anchored],parents[~anchored],targets[~anchored]
        if not len(ids):
            continue
        # Baseline and no-context arms may compute redundant identical paths;
        # deduplicating both IDs keeps exact schedules aligned across all arms.
        grouping=targets if not isinstance(model,BaseFlow) and model.context and model.dual else np.full_like(targets,-1)
        conditions,inverse=np.unique(np.column_stack((parents,grouping)),axis=0,return_inverse=True)
        for start in range(0,len(conditions),cfg['chunk_size']):
            current=conditions[start:start+cfg['chunk_size']]
            p,j=current.T
            j=np.maximum(j,0)
            tensor=lambda values: torch.as_tensor(values,dtype=torch.float32,device=device)
            source_c=tensor(np.eye(cd,dtype=np.float32)[src['y'][p]])
            initial=torch.cat((tensor(src['x'][p]),tensor(src['g'][p]),source_c),-1)
            h0,h1=tensor(src['h'][p]),tensor(dst['h'][j])
            delta=torch.full((len(p),1),dst['z']-src['z'],device=device)
            def field(t,s):
                out=model(xt=s[:,:2],gt=s[:,2:2+gd],ct=s[:,2+gd:],
                          t=t.expand(len(p)),zt=src['z']+t*delta,delta_z=delta,
                          h0=h0,h1=h1,source_c=source_c,
                          support=torch.ones(len(p),device=device))
                return torch.cat(out,-1)
            trajectory=odeint(field,initial,times,method=cfg['solver'],atol=cfg['atol'],rtol=cfg['rtol'])
            if not torch.isfinite(trajectory).all():
                raise FloatingPointError('Nonfinite ODE trajectory; no completed generation is recorded.')
            mask=(inverse>=start)&(inverse<start+len(p))
            child=ids[mask]
            column=torch.as_tensor(inverse[mask]-start,device=device)
            local_t=schedule['t'][child] if forward else 1-schedule['t'][child]
            position=torch.as_tensor(local_t*(cfg['steps']-1),device=device)
            low=position.long();high=torch.clamp(low+1,max=cfg['steps']-1)
            w=(position-low).float().unsqueeze(-1)
            sampled=(1-w)*trajectory[low,column]+w*trajectory[high,column]
            sampled=sampled.cpu().numpy()
            x[child],g[child],continuous[child]=sampled[:,:2],sampled[:,2:2+gd],sampled[:,2+gd:]
            nearest=position.round().long().clamp(max=cfg['steps']-1)
            labels[child]=trajectory[nearest,column,2+gd:].argmax(-1).cpu().numpy()
    before=g.copy()
    f=schedule['is_forward'];p=schedule['source_index']
    nnz=np.empty(total,int)
    nnz[f]=(left['g'][p[f]]>0).sum(1);nnz[~f]=(right['g'][p[~f]]>0).sum(1)
    if cfg['sparsify']:
        order=np.argsort(-g,axis=1,kind='stable')
        mask=np.arange(gd)[None,:]<nnz[:,None]
        keep=np.zeros_like(mask);np.put_along_axis(keep,order,mask,axis=1)
        g[~keep]=0
    source_code=np.empty(total,int);source_ids=np.empty(total,object);source_z=np.empty(total)
    source_code[f]=left['y'][p[f]];source_code[~f]=right['y'][p[~f]]
    source_ids[f]=left['ids'][p[f]];source_ids[~f]=right['ids'][p[~f]]
    source_z[f]=left['z_um'];source_z[~f]=right['z_um']
    physical=x.astype(float)*state['xy_range']+state['xy_min']
    z=left['z_um']+schedule['t']*(right['z_um']-left['z_um'])
    # Restore unsupported anchors EXACTLY in physical coordinates and expression.
    unsupported=(schedule['target_index']<0)&anchor_unsupported
    for side,s in ((True,left),(False,right)):
        m=unsupported&(f==side)
        physical[m]=s['physical'][p[m]]
        g[m]=s['g'][p[m]];before[m]=g[m]
    obs=pd.DataFrame(dict(cell_class=np.asarray(state['classes'])[labels],
                          source_cell_class=np.asarray(state['classes'])[source_code],source_cell_id=source_ids,
                          source_cell_index=p,source_z=source_z,source_direction=np.where(f,'forward','backward'),
                          parent_id=[f'{zz:.12g}:{ii}' for zz,ii in zip(source_z,source_ids)],
                          target_cell_index=schedule['target_index'],candidate_count=schedule['support_count'],
                          depth_bin_index=schedule['depth_bin_index'],z_coord=z,
                          generation_mode=np.where(unsupported,'source_anchor','ode'),is_observed=False,
                          annotation_origin='model_class_state',source_annotation_origin='observed_source'),
                     index=[f'generated_{i}' for i in range(total)])
    result=ad.AnnData(X=sparse.csr_matrix(g),obs=obs,var=pd.DataFrame(index=state['genes']),
                      obsm={'spatial':physical,'spatial_3d':np.column_stack((physical,z)),
                            'cell_class_continuous':continuous},
                      layers={'expression_before_sparsification':before})
    result.uns['rc_reconstruction']=dict(generated_n=total,source_anchor_n=int(unsupported.sum()),
                                          input_X_transform='identity',type_state_is_calibrated_probability=False)
    return result
