"""Original linear velocity FM and AdamW/EMA, plus explicit A risk and C anchors."""
from copy import deepcopy
import csv
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .model import ResidualFlow


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_torch_save(value, path):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, temp)
    os.replace(temp, path)


def linear_batch(batch, device, times=None):
    b = {k: v.to(device) for k,v in batch.items()}
    if times is None:
        # Upstream Transport.sample consumes this unused draw before drawing time.
        torch.randn_like(b['x1'])
        times = torch.rand(len(b['x0']), device=device)
    t = times.reshape(-1,1)
    inputs = {key+'t': (1-t)*b[key+'0']+t*b[key+'1'] for key in ('x','g','c','z')}
    inputs.update(t=times, delta_z=b['z1']-b['z0'], h0=b['h0'], h1=b['h1'], source_c=b['c0'])
    target = tuple(b[key+'1']-b[key+'0'] for key in ('x','g','c'))
    return inputs, target, b['weight']


def fm_loss(prediction, target, weights, config):
    errors = torch.stack([(p-u).square().mean(-1) for p,u in zip(prediction,target)], -1)
    factors = errors.new_tensor([1.,config['training']['lambda_g'],config['training']['lambda_c']])
    return ((errors*factors).sum(-1)*weights).mean(), errors.mean(0)


def anchor_penalty(model, inputs, config):
    """Natural-distribution endpoints/paths and detached one-step nearby states.

    The nearby state is a first-order probe, not a claimed complete teacher rollout.
    """
    output, _, delta = model(**inputs, return_parts=True)
    def penalty(values):
        normalized = model.controller.normalized(values)
        return torch.stack([d.square().mean() for d in normalized])
    terms = penalty(delta)
    dt = config['trust']['anchor_dt']
    nearby = dict(inputs)
    next_t = torch.clamp(inputs['t']+dt, max=1.)
    actual_dt = (next_t-inputs['t']).reshape(-1,1)
    for key, vel in zip(('xt','gt','ct'), output):
        nearby[key] = (inputs[key]+actual_dt*vel).detach()
    nearby['zt'] = inputs['zt']+actual_dt*inputs['delta_z']
    nearby['t'] = next_t
    _, _, near_delta = model(**nearby, return_parts=True)
    terms = .5*(terms+penalty(near_delta))
    lambdas = terms.new_tensor([config['trust']['anchor_lambda'][m] for m in ('x','g','c')])
    return (terms*lambdas).sum(), terms


def fit(model, dataset, config, folder, seed, device, anchor_dataset=None, resume=False, metadata=None):
    """Save last optimizer/EMA/RNG state each epoch; completed outputs are handled by runner."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    model = model.to(device)
    ema = deepcopy(model).requires_grad_(False).eval()
    cfg = config['training']
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    generator = torch.Generator().manual_seed(seed)
    anchor_generator = torch.Generator().manual_seed(seed+111)
    start, history, type_history = 0, [], []
    checkpoint = folder/'last.pt'
    if checkpoint.exists():
        if not resume:
            raise FileExistsError(f'{checkpoint} exists; use --resume for the identical run.')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['metadata'] != metadata:
            raise ValueError('Resume metadata/config/source/input identity changed.')
        model.load_state_dict(saved['model'])
        ema.load_state_dict(saved['ema'])
        optimizer.load_state_dict(saved['optimizer'])
        start, history = saved['epoch'], saved['history']
        type_history = saved['type_history']
        generator.set_state(saved['loader_rng'])
        anchor_generator.set_state(saved['anchor_rng'])
        torch.set_rng_state(saved['torch_rng'])
        if device.type == 'cuda' and saved['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
    loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, generator=generator,
                        num_workers=cfg['num_workers'], pin_memory=device.type=='cuda')
    started = time.perf_counter()
    for epoch in range(start, cfg['epochs']):
        model.train()
        totals = np.zeros(9)
        nclasses=dataset.tensors['c0'].shape[1]
        type_sums=np.zeros((nclasses,3));type_counts=np.zeros(nclasses,dtype=int)
        for batch in loader:
            inputs, target, weights = linear_batch(batch, device)
            prediction = model(**inputs)
            loss, modalities = fm_loss(prediction,target,weights,config)
            errors=torch.stack([(p-u).square().mean(-1) for p,u in zip(prediction,target)],-1).detach().cpu().numpy()
            classes=inputs['source_c'].argmax(-1).cpu().numpy()
            np.add.at(type_sums,classes,errors);np.add.at(type_counts,classes,1)
            anchor, drift = loss.new_zeros(()), loss.new_zeros(3)
            if isinstance(model, ResidualFlow) and model.controller.enabled and anchor_dataset is not None:
                ids = torch.randint(len(anchor_dataset), (len(weights),), generator=anchor_generator)
                anchor_batch = {k:v[ids] for k,v in anchor_dataset.tensors.items()}
                anchor_times = torch.rand(len(weights),generator=anchor_generator).to(device)
                anchor_inputs, _, _ = linear_batch(anchor_batch, device,anchor_times)
                anchor, drift = anchor_penalty(model, anchor_inputs, config)
            objective = loss+anchor
            if not torch.isfinite(objective):
                raise FloatingPointError('Nonfinite FM/anchor loss; run is incomplete.')
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if any(not torch.isfinite(g).all() for g in grads):
                raise FloatingPointError('Nonfinite gradient; run is incomplete.')
            grad_norm = torch.sqrt(sum(g.square().sum() for g in grads))
            optimizer.step()
            with torch.no_grad():
                for ep,p in zip(ema.parameters(),model.parameters()):
                    ep.lerp_(p, 1-cfg['ema_decay'])
            n = len(weights)
            totals += n*np.array([float(objective.detach()),float(loss.detach()),float(anchor.detach()),
                                  *modalities.detach().cpu().tolist(),*drift.detach().cpu().tolist()])
        totals /= len(dataset)
        row = dict(epoch=epoch+1, objective=totals[0], fm=totals[1], anchor=totals[2],
                   loss_x=totals[3],loss_g=totals[4],loss_c=totals[5],
                   drift_x=totals[6],drift_g=totals[7],drift_c=totals[8],
                   last_batch_grad_norm=float(grad_norm),elapsed_seconds=time.perf_counter()-started)
        history.append(row)
        for c in range(nclasses):
            values=type_sums[c]/type_counts[c] if type_counts[c] else np.full(3,np.nan)
            type_history.append(dict(epoch=epoch+1,class_code=c,sampled_n=int(type_counts[c]),
                                     loss_x=float(values[0]),loss_g=float(values[1]),loss_c=float(values[2])))
        atomic_torch_save(dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),
                               epoch=epoch+1,history=history,loader_rng=generator.get_state(),
                               type_history=type_history,
                               anchor_rng=anchor_generator.get_state(),
                               torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
                               metadata=metadata),checkpoint)
        with (folder/'training.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(history)
        with (folder/'training_by_type.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(type_history[0]));writer.writeheader();writer.writerows(type_history)
        print(f'{folder.name}: epoch {epoch+1}/{cfg["epochs"]} FM={row["fm"]:.5g} anchor={row["anchor"]:.5g}',flush=True)
    return ema.eval()
