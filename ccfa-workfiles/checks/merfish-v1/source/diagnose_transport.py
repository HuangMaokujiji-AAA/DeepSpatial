"""Recompute full observed-slice UOT plans; diagnose targets without retraining."""
from pathlib import Path
import gc
import importlib.util
import json
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import ot
import scipy.sparse as sp
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / 'HM_DeepSpital/outputs/v1_holdout_2_run2'
DEST = RUN / 'analysis'


def weighted_metrics(q, xy0, xy1, distance, labels0, labels1, categories, name):
    q = q / q.sum()
    row, col = q.sum(1), q.sum(0)
    m0, m1 = row @ xy0, col @ xy1
    var0 = row @ (xy0-m0)**2
    var1 = col @ (xy1-m1)**2
    cross = np.array([xy0[:,k] @ q @ xy1[:,k] for k in range(2)]) - m0*m1
    midvar = (var0+var1+2*cross)/4
    entropy = -np.where(q>0,q*np.log(np.maximum(q,1e-300)),0).sum(1)/row+np.log(row)
    # Entropy of p(target|source), averaged under the actual source marginal.
    effective = np.exp(entropy)
    assert np.isfinite(effective).all() and effective.max() <= q.shape[1]*(1+1e-8)
    result = dict(variant=name,mean_pair_distance_um=float(np.sum(q*distance)),
                  fraction_over_100um=float(q[distance>100].sum()),
                  fraction_over_300um=float(q[distance>300].sum()),
                  fraction_over_500um=float(q[distance>500].sum()),
                  x_endpoint_correlation=float(cross[0]/np.sqrt(var0[0]*var1[0])),
                  y_endpoint_correlation=float(cross[1]/np.sqrt(var0[1]*var1[1])),
                  source_x_sd_um=float(np.sqrt(var0[0])),source_y_sd_um=float(np.sqrt(var0[1])),
                  target_x_sd_um=float(np.sqrt(var1[0])),target_y_sd_um=float(np.sqrt(var1[1])),
                  midpoint_x_sd_um=float(np.sqrt(midvar[0])),midpoint_y_sd_um=float(np.sqrt(midvar[1])),
                  mean_effective_targets=float(row@effective),
                  fraction_cross_type=float(q[labels0[:,None]!=labels1[None,:]].sum()))
    details=[]
    for k,label in enumerate(categories):
        mask0,mask1=labels0==k,labels1==k
        block=q[np.ix_(mask0,mask1)]
        if not block.size or not block.sum(): continue
        block/=block.sum()
        d=distance[np.ix_(mask0,mask1)]
        details.append(dict(variant=name,cell_class=label,source_n=int(mask0.sum()),target_n=int(mask1.sum()),
                            source_training_fraction=float(row[mask0].sum()),
                            mean_pair_distance_um=float(np.sum(block*d)),
                            fraction_over_300um=float(block[d>300].sum())))
    return result,details


def balanced_distribution(pi, labels0, labels1, fraction):
    q=(1-fraction)*pi/pi.sum()
    valid=[]
    for label in np.intersect1d(labels0,labels1):
        rows,cols=np.flatnonzero(labels0==label),np.flatnonzero(labels1==label)
        mass=pi[np.ix_(rows,cols)].sum()
        if mass>0: valid.append((rows,cols,mass))
    for rows,cols,mass in valid:
        idx=np.ix_(rows,cols)
        q[idx]+=fraction/len(valid)*pi[idx]/mass
    return q


def main():
    config=json.loads((RUN/'model/config.json').read_text(encoding='utf-8'))
    categories=config['categories']
    data={i:ad.read_h5ad(next((ROOT/'data/merfish_mouse_hypothalamus').rglob(f'merfish_{i}.h5ad'))) for i in (1,3)}
    xs,gs,cs,labels=[],[],[],[]
    for i in (1,3):
        a=data[i]
        xy=np.asarray(a.obsm['spatial']).copy()
        for k,axis in enumerate(('x','y')):
            xy[:,k]=(xy[:,k]-config['spatial_stats'][axis+'_min'])/config['spatial_stats'][axis+'_range']
        xs.append(xy.astype(np.float32))
        gs.append((a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)).astype(np.float32))
        labels.append(pd.Categorical(a.obs.cell_class.astype(str),categories=categories).codes)
        cs.append(np.eye(len(categories),dtype=np.float32)[labels[-1]])
    spec=importlib.util.spec_from_file_location('local_uot',ROOT/'HM_DeepSpital/deepspatial/data_utils/uot_solver.py')
    solver=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solver)
    cost=solver.compute_cost_matrix(xs[0],gs[0],cs[0],xs[1],gs[1],cs[1],.5)
    xy0,xy1=[np.asarray(data[i].obsm['spatial']) for i in (1,3)]
    distance=cdist(xy0,xy1)
    aa,bb=np.ones(len(xy0))/len(xy0),np.ones(len(xy1))/len(xy1)
    summaries,details=[],[]
    saved_mass=config['trajectory_sampling'][1]['transport_total_mass']
    metadata=dict(pot_version=ot.__version__,pair=[1,3],source_cells=len(xy0),target_cells=len(xy1),
                  saved_transport_total_mass=saved_mass,scope='full visible endpoint slices; pairing diagnostics, not retrained accuracy')
    for name,reg,tau in [('current',.8,.05),('lower_entropy',.05,.05),
                          ('lower_entropy_stronger_marginal',.05,.5)]:
        pi,log=ot.unbalanced.sinkhorn_unbalanced(aa,bb,cost,reg=reg,reg_m=tau,
                                                 reg_type='kl',numItermax=100,stopThr=1e-3,log=True)
        mass=float(pi.sum())
        if name=='current':
            metadata['recomputed_transport_total_mass']=mass
            metadata['relative_mass_difference']=abs(mass-saved_mass)/saved_mass
            metadata['kl_matches_saved_mass']=bool(np.isclose(mass,saved_mass,rtol=1e-5))
            if not metadata['kl_matches_saved_mass']:
                warnings.warn('Local recomputation mass differs from saved server mass; interpret as configuration diagnostic.')
            r,dd=weighted_metrics(pi,xy0,xy1,distance,*labels,categories,'original_UOT_without_balancing')
            summaries.append(r); details.extend(dd)
        q=balanced_distribution(pi,*labels,.2)
        r,dd=weighted_metrics(q,xy0,xy1,distance,*labels,categories,name)
        r.update(reg=reg,reg_m=tau,transport_total_mass=mass,
                 sinkhorn_last_error=float(log['err'][-1]))
        summaries.append(r); details.extend(dd)
        print(json.dumps(r),flush=True)
        del pi,q
        gc.collect()
    pd.DataFrame(summaries).to_csv(DEST/'pairing_diagnostics.csv',index=False)
    pd.DataFrame(details).to_csv(DEST/'pairing_class_diagnostics.csv',index=False)
    (DEST/'pairing_provenance.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    actual=[]
    for st in config['trajectory_sampling']:
        for j,label in enumerate(categories):
            actual.append(dict(slice_pair=str(st['slice_pair']),cell_class=label,
                               sampled_source_n=st['source_type_counts'][j],sampled_pairs=st['sampled_pairs'],
                               actual_training_fraction=st['source_type_counts'][j]/st['sampled_pairs'],
                               normalized_original_uot_mass=st['source_type_transport_mass'][j]/st['transport_total_mass']))
    pd.DataFrame(actual).to_csv(DEST/'actual_training_class_distribution.csv',index=False)
    volume=ad.read_h5ad(RUN/'generated.h5ad')
    depth=[]
    for lo in range(0,200,10):
        v=volume[(volume.obs.z_coord>=lo)&(volume.obs.z_coord<lo+10)]
        xy=np.asarray(v.obsm['spatial'])
        depth.append(dict(z_start=lo,z_end=lo+10,n=v.n_obs,
                          x_sd_um=float(xy[:,0].std()),y_sd_um=float(xy[:,1].std())))
    pd.DataFrame(depth).to_csv(DEST/'depth_geometry.csv',index=False)
    slab=volume[np.abs(volume.obs.z_coord-100)<=5]
    directions=[]
    for direction,z,i in [('forward',50,1),('backward',150,3)]:
        mask=slab.obs.source_direction.astype(str).to_numpy()==direction
        source=np.asarray(data[i].obsm['spatial'])[slab.obs.loc[mask,'source_cell_index'].to_numpy()]
        dest=np.asarray(slab.obsm['spatial'])[mask]
        displacement=dest-source
        directions.append(dict(direction=direction,n=int(mask.sum()),
                               inward_fraction=float((np.sum(displacement*(-source),axis=1)>0).mean()),
                               displacement_median_um=float(np.median(np.linalg.norm(displacement,axis=1))),
                               x_sd_um=float(dest[:,0].std()),y_sd_um=float(dest[:,1].std())))
    pd.DataFrame(directions).to_csv(DEST/'direction_diagnostics.csv',index=False)
    print('Provenance',json.dumps(metadata),flush=True)


if __name__=='__main__':
    main()
