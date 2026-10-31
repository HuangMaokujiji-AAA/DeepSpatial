"""Synthetic engineering checks; these do not measure MERFISH biological performance."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import torch

from rcspatial.data import prepare,context_view
from rcspatial.pairing import directed_distribution,build_plans,Trajectories
from rcspatial.model import BaseFlow,ResidualFlow
from rcspatial.training import linear_batch,fm_loss,fit,seed_all,atomic_torch_save
from rcspatial.experiment import run,compare,digest,checkpoint_model,ARMS
from rcspatial.reconstruction import load_schedule,reconstruct

torch.set_num_threads(1)


def fixture(root):
    cfg=json.loads((PROJECT/'configs/merfish.json').read_text(encoding='utf-8'))
    cfg['input_glob']=str(root/'merfish_[0-4].h5ad')
    cfg['model'].update(hidden_size=32,depth=1,num_heads=4,patch_size=2)
    cfg['residual']['hidden_size']=16
    cfg['pairing']['n_samples']=32
    cfg['training'].update(epochs=1,batch_size=16)
    cfg['reconstruction'].update(steps=4,chunk_size=16,solver='rk4',thickness_um=50.)
    cfg['evaluation']['rare_fraction']=.2
    cfg['test_purpose']='synthetic_engineering_only'
    rng=np.random.default_rng(7)
    for k in range(5):
        labels=['common']*10+(['isolated']*2 if k==0 else ['rare']*2)
        x=rng.poisson(3,(12,4)).astype(np.float32)
        x[-2:,0]+=5
        xy=np.column_stack((np.arange(12)*4+k,np.arange(12)%3*3)).astype(float)
        a=ad.AnnData(X=x,obs=pd.DataFrame({'cell_class':labels},index=[f's{k}_{i}' for i in range(12)]),
                     var=pd.DataFrame(index=['g0','g1','g2','g3']),obsm={'spatial':xy})
        a.write_h5ad(root/f'merfish_{k}.h5ad')
    return cfg,[root/f'merfish_{i}.h5ad' for i in range(5)]


class PipelineTests(unittest.TestCase):
    def test_directed_marginal_and_forbidden_edges(self):
        p=sparse.csr_matrix([[9.,1.,0.],[0.,1.,1.],[0.,0.,0.]])
        q=directed_distribution(p,1.)
        np.testing.assert_allclose(np.asarray(q.sum(1)).ravel(),[.5,.5,0])
        np.testing.assert_allclose(directed_distribution(p,0).toarray(),p.toarray()/p.sum())
        self.assertTrue(np.all(q.toarray()[p.toarray()==0]==0))
        self.assertFalse(np.allclose(np.asarray(q.sum(0)).ravel(),np.full(3,1/3)))
        np.testing.assert_allclose(directed_distribution(p,.4).toarray(),.6*p.toarray()/p.sum()+.4*q.toarray())
        with self.assertRaises(ValueError):directed_distribution(p,-1)

    def test_zero_residual_frozen_backbone_and_modality_caps(self):
        with tempfile.TemporaryDirectory(dir=PROJECT) as tmp:
            cfg,paths=fixture(Path(tmp));slices,state=prepare(paths,cfg,1)
            p=build_plans(slices,cfg,'local')
            ds=Trajectories(slices,p,state,32,3,kappa=.5,balanced_risk=.2)
            batch={k:v[:8] for k,v in ds.tensors.items()}
            inputs,target,weights=linear_batch(batch,torch.device('cpu'),torch.full((8,),.3))
            base=BaseFlow(state,cfg).eval()
            model=ResidualFlow(state,cfg,ARMS['ABC'],base.state_dict()).eval()
            expected=base(**inputs);actual=model(**inputs)
            for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
            loss,_=fm_loss(actual,target,weights,cfg);loss.backward()
            self.assertTrue(all(p.grad is None for p in model.base.parameters()))
            self.assertIsNotNone(model.head[-1].weight.grad)
            raw=tuple(torch.full_like(u,10000.) for u in target)
            bounded=model.controller(raw,inputs['source_c'])
            gates=model.controller.class_gates[inputs['source_c'].argmax(-1)]
            for normed,cap in zip(model.controller.normalized(bounded),model.controller.caps):
                self.assertTrue(torch.all(normed.square().mean(-1).sqrt()<=cap*gates+1e-6))
            # MSE objective is the original linear velocity-FM objective.
            manual=sum(f*((a-b).square().mean(1)*weights).mean()
                       for f,a,b in zip((1.,cfg['training']['lambda_g'],cfg['training']['lambda_c']),actual,target))
            torch.testing.assert_close(loss,manual)

    def test_context_and_outer_slice_isolation(self):
        with tempfile.TemporaryDirectory(dir=PROJECT) as tmp:
            cfg,paths=fixture(Path(tmp))
            original=ad.read_h5ad;opened=[]
            def reader(path,*args,**kwargs):
                opened.append(Path(path));return original(path,*args,**kwargs)
            with patch('rcspatial.data.ad.read_h5ad',side_effect=reader):
                slices,state=prepare(paths,cfg,1,excluded=[2])
            self.assertNotIn(paths[1],opened);self.assertNotIn(paths[2],opened)
            self.assertEqual(state['train_indices'],[0,3,4])
            shuffled=context_view(slices,'shuffle',11)
            for before,after in zip(slices,shuffled):
                for c in np.unique(before['y']):
                    mask=before['y']==c
                    np.testing.assert_allclose(np.sort(before['h'][mask],axis=0),np.sort(after['h'][mask],axis=0))
            self.assertTrue(all(not s['h'].any() for s in context_view(slices,'zero',11)))

    def test_end_to_end_save_reload_resume_and_evaluation_order(self):
        with tempfile.TemporaryDirectory(dir=PROJECT) as tmp:
            root=Path(tmp);cfg,paths=fixture(root);out=root/'run'
            arms=['baseline','control','R0','ABC','ABC_shuffle']
            reader=ad.read_h5ad
            def guarded(path,*args,**kwargs):
                if Path(path)==paths[1]:
                    self.assertTrue(all((out/'holdout_1/seed_0'/arm/'generated.json').exists() for arm in arms))
                return reader(path,*args,**kwargs)
            with patch('rcspatial.experiment.ad.read_h5ad',side_effect=guarded):
                records=run(cfg,out,[1],[0],arms,device='cpu')
            self.assertEqual(len(records),len(arms))
            table=compare(out);self.assertEqual(len(table),len(arms))
            fold=out/'holdout_1/seed_0'
            control=json.loads((fold/'control/generated.json').read_text())['schedule_sha256']
            for arm in ('R0','ABC','ABC_shuffle'):
                self.assertEqual(json.loads((fold/arm/'generated.json').read_text())['schedule_sha256'],control)
            state=json.loads((fold/'shared/state.json').read_text())
            model,_=checkpoint_model(fold/'ABC',state,cfg,ARMS['ABC'],torch.device('cpu'))
            slices,_=prepare(paths,cfg,1)
            schedule=load_schedule(fold/'shared/schedule_control.npz')
            result=reconstruct(model,slices[0],slices[1],schedule,state,cfg,torch.device('cpu'))
            saved=ad.read_h5ad(fold/'ABC/predicted_gap.h5ad')
            np.testing.assert_allclose(result.X.toarray(),saved.X.toarray(),rtol=0,atol=0)
            anchored=result.obs.generation_mode=='source_anchor'
            self.assertGreater(int(anchored.sum()),0)
            for i in np.flatnonzero(anchored):
                s=slices[0] if result.obs.iloc[i].source_direction=='forward' else slices[1]
                parent=int(result.obs.iloc[i].source_cell_index)
                np.testing.assert_array_equal(result.obsm['spatial'][i],s['physical'][parent])
                np.testing.assert_array_equal(result.X[i].toarray()[0],s['g'][parent])
            checkpoint=fold/'ABC/last.pt';fingerprint=digest(checkpoint);mtime=checkpoint.stat().st_mtime_ns
            run(cfg,out,[1],[0],arms,device='cpu',resume=True)
            self.assertEqual(digest(checkpoint),fingerprint);self.assertEqual(checkpoint.stat().st_mtime_ns,mtime)
            changed=deepcopy(cfg);changed['A']['kappa']=.75
            with self.assertRaises(ValueError):run(changed,out,[1],[0],arms,stage='audit',device='cpu',resume=True)

    def test_interrupted_training_restores_optimizer_ema_and_random_stream(self):
        with tempfile.TemporaryDirectory(dir=PROJECT) as tmp:
            root=Path(tmp);cfg,paths=fixture(root);cfg['training']['epochs']=2
            slices,state=prepare(paths,cfg,1);plans=build_plans(slices,cfg,'local')
            ds=Trajectories(slices,plans,state,32,3,kappa=.5,balanced_risk=.2)
            seed_all(3);template=ResidualFlow(state,cfg,ARMS['ABC'])
            expected=fit(deepcopy(template),ds,cfg,root/'continuous',3,torch.device('cpu'),ds,metadata={'test':True})
            def stop_after_checkpoint(value,path):
                atomic_torch_save(value,path)
                raise RuntimeError('Simulated process interruption after an epoch checkpoint.')
            with patch('rcspatial.training.atomic_torch_save',side_effect=stop_after_checkpoint):
                with self.assertRaisesRegex(RuntimeError,'Simulated'):
                    fit(deepcopy(template),ds,cfg,root/'interrupted',3,torch.device('cpu'),ds,metadata={'test':True})
            actual=fit(deepcopy(template),ds,cfg,root/'interrupted',3,torch.device('cpu'),ds,resume=True,metadata={'test':True})
            for key,value in expected.state_dict().items():
                torch.testing.assert_close(actual.state_dict()[key],value,rtol=0,atol=0)


if __name__=='__main__':
    unittest.main(verbosity=2)
