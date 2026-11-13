"""Velocity FM, C marker rollouts and independent natural-risk D anchors."""
from copy import deepcopy
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import GapBatches, interpolate


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(path, value):
    temporary = Path(path).with_suffix('.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def fit(model, dataset, config, folder, state, device, metadata, marker_tasks=None, anchors=None, resume=False):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    seed = config['seed']; seed_all(seed)
    model = model.to(device)
    ema = deepcopy(model).requires_grad_(False).eval()
    cfg = config['training']
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    start, step, history, marker_history = 0, 0, [], []
    anchor_rng = np.random.default_rng(seed+938)
    times_rng = torch.Generator().manual_seed(seed+319)
    path = folder/'last.pt'
    if path.exists():
        if not resume:
            raise FileExistsError(f'Use --resume for the identical run: {path}')
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if saved['metadata'] != metadata:
            raise ValueError('Resume metadata changed.')
        model.load_state_dict(saved['model']); ema.load_state_dict(saved['ema']); optimizer.load_state_dict(saved['optimizer'])
        start, step, history, marker_history = saved['epoch'], saved['step'], saved['history'], saved['marker_history']
        torch.set_rng_state(saved['torch_rng']); np.random.set_state(saved['numpy_rng']); random.setstate(saved['python_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        anchor_rng.bit_generator.state = saved['anchor_rng']; times_rng.set_state(saved['times_rng'])
        if marker_tasks is not None:
            marker_tasks.rng.bit_generator.state = saved['marker_rng']
    started = time.perf_counter()
    for epoch in range(start, cfg['epochs']):
        model.train(); totals = np.zeros(7); examples = 0
        loader = DataLoader(dataset, batch_sampler=GapBatches(dataset, cfg['batch_size'], seed, epoch), num_workers=0,
                            pin_memory=device.type == 'cuda', generator=torch.Generator().manual_seed(seed+epoch))
        for batch in loader:
            n = len(batch['weight'])
            # Independent time stream keeps FM draws aligned across C/D ablations.
            times = torch.rand(n, generator=times_rng).to(device)
            inputs, target, weights = interpolate(batch, state, device, times)
            prediction = model(**inputs)
            modality = torch.stack([(p-u).square().mean(-1) for p, u in zip(prediction, target)], -1)
            factors = modality.new_tensor([1., cfg['lambda_g'], cfg['lambda_c']])
            fm = ((modality*factors).sum(-1)*weights).mean()
            anchor = fm*0
            if anchors is not None and model.controller.enabled:
                ids = anchor_rng.integers(len(anchors), size=min(n, cfg['batch_size']))
                items = [anchors[int(i)] for i in ids]
                abatch = {k: torch.as_tensor(np.stack([r[k] for r in items])) for k in items[0]}
                at = torch.as_tensor(anchor_rng.random(len(ids)), dtype=torch.float32, device=device)
                ai, _, _ = interpolate(abatch, state, device, at)
                _, _, residual = model(**ai, return_parts=True)
                anchor = config['D']['anchor_lambda']*sum(v.square().mean() for v in model.controller.normalized(residual))
            marker = fm*0
            if marker_tasks is not None and step % config['C']['every_steps'] == 0:
                marker, log = marker_tasks.loss(model, device)
                marker_history.append(dict(epoch=epoch+1, step=step, **log))
            objective = fm+anchor+marker
            if not torch.isfinite(objective):
                raise FloatingPointError('Nonfinite training objective.')
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for ep, p in zip(ema.parameters(), model.parameters()):
                    ep.lerp_(p, 1-cfg['ema_decay'])
            totals += n*np.array([float(objective.detach()), float(fm.detach()), float(marker.detach()), float(anchor.detach()),
                                   *modality.mean(0).detach().cpu().tolist()])
            examples += n; step += 1
        totals /= examples
        history.append(dict(epoch=epoch+1, objective=totals[0], fm=totals[1], marker=totals[2], anchor=totals[3],
                            loss_x=totals[4], loss_g=totals[5], loss_c=totals[6], last_gradient_norm=float(norm),
                            seconds_this_invocation=time.perf_counter()-started))
        save_checkpoint(path, dict(model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                                   epoch=epoch+1, step=step, metadata=metadata, history=history, marker_history=marker_history,
                                   torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                                   cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
                                   anchor_rng=anchor_rng.bit_generator.state, times_rng=times_rng.get_state(),
                                   marker_rng=marker_tasks.rng.bit_generator.state if marker_tasks is not None else None))
        pd.DataFrame(history).to_csv(folder/'training.csv', index=False)
        pd.DataFrame(marker_history).to_csv(folder/'marker_training.csv', index=False)
        print(f'{folder.name}: {epoch+1}/{cfg["epochs"]} FM={totals[1]:.5g} C={totals[2]:.5g} D={totals[3]:.5g}', flush=True)
    # Also repair logs if interruption happened after the final atomic checkpoint.
    pd.DataFrame(history).to_csv(folder/'training.csv', index=False)
    pd.DataFrame(marker_history).to_csv(folder/'marker_training.csv', index=False)
    return ema
