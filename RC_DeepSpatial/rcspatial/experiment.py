"""Auditable MERFISH leave-one-slice-out experiment orchestration."""
from copy import deepcopy
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

import anndata as ad
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
import torch

from .data import discover,prepare,context_view,validate
from .pairing import build_plans,audit_plans,support_diagnostics,Trajectories
from .model import make_model
from .training import fit,seed_all
from .reconstruction import make_schedule,save_schedule,load_schedule,reconstruct
from .evaluation import fit_probe,evaluate

PROJECT=Path(__file__).resolve().parents[1]
ARMS={
    'baseline':dict(base=True,A=False,B=False,C=False,raw=True),
    'control':dict(base=True,A=False,B=False,C=False),
    'R0':dict(A=False,B=False,C=False),
    'A':dict(A=True,B=False,C=False),
    'B':dict(A=False,B=True,C=False),
    'AB':dict(A=True,B=True,C=False),
    'ABC':dict(A=True,B=True,C=True),
    'ABC_zero':dict(A=True,B=True,C=True,context='zero'),
    'ABC_shuffle':dict(A=True,B=True,C=True,context='shuffle'),
    'ABC_single':dict(A=True,B=True,C=True,dual=False),
    'ABC_no_geometry':dict(A=True,B=True,C=True,context='no_geometry'),
    'ABC_no_anchor':dict(A=True,B=True,C=True,no_anchor=True),
    'AB_risk_only':dict(A=True,B=True,C=False,kappa=0.),
    'AB_marginal_only':dict(A=True,B=True,C=False,risk=0.),
}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def source_hashes():
    return {str(p.relative_to(PROJECT)).replace('\\','/'):digest(p)
            for root in ('rcspatial','examples') for p in sorted((PROJECT/root).rglob('*.py'))}


def clean_json(value):
    if isinstance(value,dict):return {str(k):clean_json(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean_json(v) for v in value]
    if isinstance(value,np.ndarray):return clean_json(value.tolist())
    if isinstance(value,np.generic):return clean_json(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def write_json(path,value):
    path=Path(path);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(clean_json(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(tmp,path)


def check_or_create(folder,identity,resume):
    folder=Path(folder)
    if folder.exists():
        if not resume:raise FileExistsError(f'{folder} exists; choose a new output root or --resume.')
        if not (folder/'manifest.json').is_file() or json.loads((folder/'manifest.json').read_text(encoding='utf-8'))!=identity:
            raise ValueError(f'Configuration, source or visible inputs changed: {folder}')
    else:
        folder.mkdir(parents=True)
        write_json(folder/'manifest.json',identity)


def load_or_build_plans(folder,slices,config,kind):
    paths=[folder/f'{kind}_gap_{i}.npz' for i in range(len(slices)-1)]
    stamp=folder/f'{kind}_cache.json'
    if all(p.exists() for p in paths) and stamp.exists():
        hashes=json.loads(stamp.read_text(encoding='utf-8'))
        if {p.name:digest(p) for p in paths}!=hashes:
            raise ValueError(f'{kind} coupling cache changed after creation.')
        return [sparse.load_npz(p) for p in paths]
    if any(p.exists() for p in paths):
        # Earlier partial creation is deterministic; overwrite task-owned caches only.
        print(f'Rebuilding incomplete {kind} coupling cache.',flush=True)
    plans=build_plans(slices,config,kind)
    for p,matrix in zip(paths,plans):sparse.save_npz(p,matrix)
    write_json(stamp,{p.name:digest(p) for p in paths})
    return plans


def checkpoint_model(folder,state,config,arm,device):
    saved=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
    model=make_model(state,config,arm)
    model.load_state_dict(saved['ema'])
    return model.to(device).eval(),saved


def run(config,output,holdouts,seeds,arms,stage='all',device='auto',resume=False,excluded=(),role='development'):
    if any(a not in ARMS for a in arms):raise ValueError('Unknown arm requested.')
    if stage not in ('audit','base','train','generate','evaluate','all'):raise ValueError('Unknown stage.')
    if role not in ('development','validation','test'):raise ValueError('Unknown evaluation role.')
    if any(not 0<=i<len(config['slice_z_um']) for i in excluded):raise ValueError('Excluded slice index out of range.')
    if (config['training']['epochs']<1 or config['training']['batch_size']<1
            or config['reconstruction']['steps']<2 or config['reconstruction']['chunk_size']<1
            or config['reconstruction']['thickness_um']<=0 or config['reconstruction']['slab_width_um']<=0
            or config['evaluation']['spatial_bin_um']<=0):
        raise ValueError('Require positive training/generation budgets and at least two ODE steps.')
    if config['training']['num_workers']!=0:
        raise ValueError('This resumable in-memory runner uses num_workers=0; use separate GPU processes for folds.')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else torch.device(device)
    output=Path(output).resolve()
    if not output.is_relative_to(PROJECT):raise ValueError('Output must stay inside RC_DeepSpatial.')
    paths=discover(config,PROJECT)
    sources=source_hashes()
    config=deepcopy(config)
    output.mkdir(parents=True,exist_ok=True)
    root_identity=dict(config=config,source_sha256=sources,excluded=list(excluded),role=role)
    root_stamp=output/'experiment.json'
    if root_stamp.exists():
        if json.loads(root_stamp.read_text(encoding='utf-8'))!=root_identity:
            raise ValueError('Output root already belongs to different sources/configuration/role; use a new root.')
    else:
        write_json(root_stamp,root_identity)
    if len(set(holdouts))!=len(holdouts) or len(set(seeds))!=len(seeds):raise ValueError('Duplicate fold or seed.')
    if role=='validation' and not excluded:raise ValueError('Validation runs must declare excluded outer-test indices.')
    records=[]
    for heldout in holdouts:
        if heldout in excluded:raise ValueError('An evaluation target cannot also be an excluded outer-test input.')
        visible=[i for i in range(len(paths)) if i not in {heldout,*excluded}]
        identity=dict(config=config,source_sha256=sources,visible_inputs={str(i):dict(file=paths[i].name,sha256=digest(paths[i])) for i in visible},
                      heldout=int(heldout),excluded=list(excluded),role=role)
        # No hidden file bytes or contents are opened for training identity.
        for seed in seeds:
            folder=output/f'holdout_{heldout}'/f'seed_{seed}'
            check_or_create(folder,dict(identity,seed=int(seed)),resume)
            shared=folder/'shared';shared.mkdir(exist_ok=True)
            slices,state=prepare(paths,config,heldout,excluded)
            write_json(shared/'state.json',state)
            pd.DataFrame(dict(cell_class=state['classes'],train_n=state['counts'])).to_csv(shared/'train_counts.csv',index=False)
            local=load_or_build_plans(shared,slices,config,'local')
            audit_plans(slices,local,state,config['A']['kappa']).to_csv(shared/'transport_local.csv',index=False)
            support_diagnostics(slices,local,state).to_csv(shared/'matching_support.csv',index=False)
            left=max((s for s in slices if s['index']<heldout),key=lambda s:s['index'])
            right=min((s for s in slices if s['index']>heldout),key=lambda s:s['index'])
            gap=next(i for i in range(len(slices)-1) if slices[i]['index']==left['index'])
            raw=None
            if 'baseline' in arms or stage=='base':
                raw=load_or_build_plans(shared,slices,config,'raw')
                audit_plans(slices,raw,state,config['A']['kappa']).to_csv(shared/'transport_raw.csv',index=False)
            for name,stratified,plans in (('control',True,local),('baseline',False,raw)):
                if plans is None:continue
                schedule_path=shared/f'schedule_{name}.npz'
                schedule_stamp=shared/f'schedule_{name}.json'
                if schedule_path.exists() and schedule_stamp.exists():
                    if digest(schedule_path)!=json.loads(schedule_stamp.read_text(encoding='utf-8'))['sha256']:
                        raise ValueError('A shared inference schedule changed after creation.')
                else:
                    sched=make_schedule(left,right,plans[gap],config,seed+1701,stratified)
                    save_schedule(schedule_path,sched)
                    write_json(schedule_stamp,dict(sha256=digest(schedule_path)))
                    if sched['budget']:pd.DataFrame(sched['budget']).to_csv(shared/'sampling_budget.csv',index=False)
            if not (shared/'probe.joblib').exists():
                probe,markers=fit_probe(slices,state,seed)
                joblib.dump(probe,shared/'probe.joblib');write_json(shared/'markers.json',markers)
            environment=dict(python=platform.python_version(),device=str(device),cuda=torch.version.cuda,
                             gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None)
            for package in ('torch','numpy','pandas','scipy','anndata','POT','timm','torchdiffeq','scikit-learn'):
                environment[package]=importlib.metadata.version(package)
            write_json(shared/'environment.json',environment)
            if stage=='audit':
                records.append(dict(heldout=heldout,seed=seed,status='audit_only'));continue
            requested=list(arms)
            if stage=='base':requested=['baseline','control']
            elif stage in ('train','all') and any(not ARMS[a].get('base') for a in requested):
                requested=['control',*[a for a in requested if a!='control']]
            for name in requested:
                arm=ARMS[name];arm_config=deepcopy(config)
                if arm.get('no_anchor'):
                    arm_config['trust']['anchor_lambda']={m:0. for m in ('x','g','c')}
                current=folder/name
                arm_identity=dict(identity,seed=int(seed),arm=name,arm_options=arm)
                check_or_create(current,arm_identity,resume)
                israw=arm.get('raw',False)
                plans=raw if israw else local
                if stage in ('all','train','base'):
                    if not (current/'trained.json').exists():
                        view=context_view(slices,arm.get('context','real'),seed)
                        kappa=arm.get('kappa',config['A']['kappa']) if arm['A'] else 0.
                        risk=arm.get('risk',config['A']['balanced_risk']) if arm['A'] else 0.
                        dataset=Trajectories(view,plans,state,config['pairing']['n_samples'],seed,
                                             kappa=kappa,bidirectional=not israw,balanced_risk=risk,
                                             weight_cap=config['A']['weight_cap'])
                        dataset.diagnostics.to_csv(current/'trajectory_sampling.csv',index=False)
                        np.savez_compressed(current/'trajectory_indices.npz',**{k:dataset.tensors[k].numpy()
                                            for k in ('slice0','slice1','cell0','cell1','weight')})
                        base_state=None
                        if not arm.get('base'):
                            source=folder/'control'/'last.pt'
                            if not (folder/'control'/'trained.json').exists():raise FileNotFoundError('Train control backbone first.')
                            base_state=torch.load(source,map_location='cpu',weights_only=False)['ema']
                            arm_identity['base_checkpoint_sha256']=digest(source)
                        seed_all(seed)
                        model=make_model(state,arm_config,arm,base_state)
                        # Context permutations are applied to anchor examples too.
                        anchors=Trajectories(view,local,state,config['pairing']['n_samples'],seed+510,
                                              bidirectional=True) if arm['C'] else None
                        started=time.perf_counter()
                        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
                        try:
                            ema=fit(model,dataset,arm_config,current,seed,device,anchors,resume,metadata=arm_identity)
                            write_json(current/'trained.json',dict(epochs=config['training']['epochs'],
                                       elapsed_seconds=time.perf_counter()-started,
                                       trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                                       total_parameters=sum(p.numel() for p in model.parameters()),trajectory_ess=dataset.ess,
                                       peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
                                       base_checkpoint_sha256=arm_identity.get('base_checkpoint_sha256'),checkpoint_sha256=digest(current/'last.pt')))
                        except Exception as exc:
                            write_json(current/'failed.json',dict(stage='train',error=repr(exc)));raise
                        del model,ema,dataset,anchors
                    else:
                        saved=json.loads((current/'trained.json').read_text(encoding='utf-8'))
                        if digest(current/'last.pt')!=saved['checkpoint_sha256']:raise ValueError('Completed checkpoint was changed.')
                if stage in ('all','generate') and not (current/'generated.json').exists():
                    if not (current/'trained.json').exists():raise FileNotFoundError(f'Train {name} before generation.')
                    model,_=checkpoint_model(current,state,arm_config,arm,device)
                    view=context_view([left,right],arm.get('context','real'),seed)
                    schedule_path=shared/f'schedule_{"baseline" if israw else "control"}.npz'
                    schedule=load_schedule(schedule_path)
                    started=time.perf_counter()
                    result=reconstruct(model,*view,schedule,state,arm_config,device,anchor_unsupported=not israw)
                    result.uns['rc_run_json']=json.dumps(dict(arm=name,heldout=heldout,seed=seed,role=role))
                    result.write_h5ad(current/'predicted_gap.h5ad',compression='gzip')
                    center=float(config['slice_z_um'][heldout]);width=config['reconstruction']['slab_width_um']
                    selected=np.abs(result.obs.z_coord.to_numpy()-center)<=width/2
                    slab=result[selected].copy()
                    if not slab.n_obs:raise ValueError('Fixed held-out slab has zero cells; no test-derived count correction.')
                    slab.write_h5ad(current/'predicted_holdout.h5ad',compression='gzip')
                    slab.obs.cell_class.astype(str).value_counts().rename_axis('cell_class').rename('generated_n').to_csv(current/'generated_type_counts.csv')
                    write_json(current/'generated.json',dict(schedule_sha256=digest(schedule_path),
                               gap_n=result.n_obs,slab_n=slab.n_obs,elapsed_seconds=time.perf_counter()-started,
                               gap_sha256=digest(current/'predicted_gap.h5ad'),slab_sha256=digest(current/'predicted_holdout.h5ad')))
                    del result,slab,model
                gc.collect()
                if device.type=='cuda':torch.cuda.empty_cache()
            # Hidden expression/labels are FIRST opened here, after ALL requested training/generation.
            if stage in ('all','evaluate'):
                probe=joblib.load(shared/'probe.joblib')
                markers={int(k):v for k,v in json.loads((shared/'markers.json').read_text(encoding='utf-8')).items()}
                truth=ad.read_h5ad(paths[heldout]);validate(truth,config['data'],state['genes'])
                write_json(shared/'evaluation_truth.json',dict(file=paths[heldout].name,sha256=digest(paths[heldout]),role=role))
                for name in requested:
                    current=folder/name
                    if (current/'evaluated.json').exists():continue
                    if not (current/'generated.json').exists():raise FileNotFoundError(f'Generate {name} before evaluation.')
                    generated=json.loads((current/'generated.json').read_text(encoding='utf-8'))
                    if digest(current/'predicted_holdout.h5ad')!=generated['slab_sha256']:raise ValueError('Prediction changed after generation.')
                    prediction=ad.read_h5ad(current/'predicted_holdout.h5ad')
                    table,metrics,marker_table=evaluate(truth,prediction,state,config,probe,markers)
                    table.to_csv(current/'per_type.csv',index=False);marker_table.to_csv(current/'marker_sparsification.csv',index=False)
                    metrics.update(arm=name,heldout=heldout,seed=seed,role=role)
                    write_json(current/'metrics.json',metrics)
                    write_json(current/'evaluated.json',dict(truth_sha256=digest(paths[heldout]),prediction_sha256=generated['slab_sha256']))
                    records.append(metrics)
            del slices
    return records


def generation_drift(output):
    """Compare the same scheduled generated children to control, not to true cell identities."""
    rows=[]
    for control_path in Path(output).glob('holdout_*/seed_*/control/predicted_holdout.h5ad'):
        fold=control_path.parent.parent
        state=json.loads((fold/'shared/state.json').read_text(encoding='utf-8'))
        control=ad.read_h5ad(control_path)
        reference=json.loads((control_path.parent/'generated.json').read_text(encoding='utf-8'))
        for path in fold.glob('*/predicted_holdout.h5ad'):
            if path==control_path:continue
            stamp=json.loads((path.parent/'generated.json').read_text(encoding='utf-8'))
            if stamp['schedule_sha256']!=reference['schedule_sha256']:continue
            pred=ad.read_h5ad(path)
            identity=['source_cell_id','source_z','source_direction','target_cell_index','z_coord']
            if not pred.obs[identity].equals(control.obs[identity]):
                raise ValueError('Identical schedule hash but generated child identities differ.')
            dx=np.linalg.norm(pred.obsm['spatial']-control.obsm['spatial'],axis=1)
            dg=((pred.X.toarray()-control.X.toarray())/state['gene_scale'])
            dg=np.sqrt(np.mean(dg**2,axis=1))
            labels=pred.obs.source_cell_class.astype(str).to_numpy()
            changed=pred.obs.cell_class.astype(str).to_numpy()!=control.obs.cell_class.astype(str).to_numpy()
            for name in state['classes']:
                mask=labels==name
                if not mask.any():continue
                rows.append(dict(fold=fold.parent.name,seed=fold.name,arm=path.parent.name,source_cell_class=name,
                                 n=int(mask.sum()),mean_xy_drift_um=float(dx[mask].mean()),
                                 p95_xy_drift_um=float(np.quantile(dx[mask],.95)),
                                 gene_scaled_rms=float(dg[mask].mean()),class_change_fraction=float(changed[mask].mean())))
    pd.DataFrame(rows).to_csv(Path(output)/'drift_to_control.csv',index=False)


def compare(output,allow_incomplete=False):
    """Paired seed/fold deltas, descriptive summaries; never auto-select on held-out scores."""
    output=Path(output)
    root=json.loads((output/'experiment.json').read_text(encoding='utf-8'))
    for path in output.glob('holdout_*/seed_*/manifest.json'):
        manifest=json.loads(path.read_text(encoding='utf-8'))
        if any(manifest[k]!=root[k] for k in ('config','source_sha256','role','excluded')):
            raise ValueError('Refusing to compare mixed source/configuration/role runs.')
    folders=[p.parent for p in output.glob('holdout_*/seed_*/*/manifest.json')]
    completeness=pd.DataFrame([dict(folder=str(p.relative_to(output)),
                              trained=(p/'trained.json').exists(),generated=(p/'generated.json').exists(),
                              evaluated=(p/'evaluated.json').exists()) for p in folders])
    if len(completeness):
        completeness.to_csv(output/'completion.csv',index=False)
        if not completeness[['trained','generated','evaluated']].all().all():
            if allow_incomplete:
                print('Other folds/arms are incomplete; completion.csv updated, aggregate summary deferred.',flush=True)
                return None
            raise ValueError('Incomplete arms found; inspect completion.csv. Partial runs are not summarized as full experiments.')
    rows=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(output.glob('holdout_*/seed_*/*/metrics.json'))]
    if not rows:raise ValueError('No evaluated runs found.')
    table=pd.DataFrame(rows)
    table.to_csv(output/'all_runs.csv',index=False)
    numeric=[c for c in table.select_dtypes(include=np.number) if c not in ('heldout','seed')]
    deltas=[]
    for reference in ('baseline','control','R0','AB'):
        base=table[table.arm==reference].set_index(['heldout','seed','role'])
        for row in rows:
            key=(row['heldout'],row['seed'],row['role'])
            if key not in base.index or row['arm']==reference:continue
            delta=dict(arm=row['arm'],reference=reference,heldout=key[0],seed=key[1],role=key[2])
            for c in numeric:
                a,b=row.get(c),base.loc[key,c]
                delta[c]=float(a-b) if a is not None and pd.notna(b) else np.nan
            deltas.append(delta)
    pd.DataFrame(deltas).to_csv(output/'paired_deltas.csv',index=False)
    fold=table.groupby(['arm','role','heldout'],as_index=False)[numeric].mean()
    fold.to_csv(output/'seed_averaged_folds.csv',index=False)
    fold.groupby(['arm','role'])[numeric].agg(['mean','std']).to_csv(output/'descriptive_summary.csv')
    generation_drift(output)
    return table
