"""Target depth is known; hidden cell counts, positions and labels are never queried."""
import hashlib
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torchdiffeq import odeint

from .vendor.sampling import systematic_counts


def schedule(left, right, z_um, state, config, stratified=True):
    fraction = (z_um-left['z_um'])/(right['z_um']-left['z_um'])
    if not 0 < fraction < 1:
        raise ValueError('Generate strictly between measured training endpoints.')
    n0, n1 = len(left['y']), len(right['y'])
    total = max(1, int(round((1-fraction)*n0+fraction*n1)))
    extra = int.from_bytes(hashlib.sha256(f'{left["name"]}|{right["name"]}|{z_um:.9f}'.encode()).digest()[:4], 'little')
    rng = np.random.default_rng(np.random.SeedSequence([config['seed'], extra]))
    if stratified:
        cd = len(state['classes'])
        p0, p1 = np.bincount(left['y'], minlength=cd)/n0, np.bincount(right['y'], minlength=cd)/n1
        joint = np.column_stack(((1-fraction)*p0, fraction*p1))
        budgets = systematic_counts(joint.ravel(), total, rng).reshape(cd, 2)
        forward, parents = [], []
        for c in range(cd):
            for d, source in enumerate((left, right)):
                count = int(budgets[c, d])
                if not count:
                    continue
                pool = rng.permutation(np.flatnonzero(source['y'] == c))
                repeats, remainder = divmod(count, len(pool))
                parents.append(np.concatenate((np.tile(pool, repeats), pool[:remainder])))
                forward.append(np.full(count, d == 0))
        parents, forward = np.concatenate(parents), np.concatenate(forward)
        order = rng.permutation(total); parents, forward = parents[order], forward[order]
    else:
        forward = rng.random(total) < 1-fraction
        parents = np.empty(total, np.int64)
        parents[forward] = rng.integers(n0, size=int(forward.sum()))
        parents[~forward] = rng.integers(n1, size=int((~forward).sum()))
    return dict(forward=forward, parent=parents, fraction=float(fraction), z_um=float(z_um))


def add_targets(sched, left, right, transports, kind, seed):
    rng = np.random.default_rng(seed+1701)
    targets = np.full(len(sched['parent']), -1, np.int64)
    for forward in (True, False):
        mask = sched['forward'] == forward
        targets[mask] = transports.targets(left['name'], right['name'], forward, sched['parent'][mask], kind, rng)
    return dict(sched, target=targets)


@torch.no_grad()
def reconstruct(model, left, right, sched, state, config, device, baseline=False):
    model = model.to(device).eval()
    cfg = config['reconstruction']; total = len(sched['parent'])
    gd, cd = len(state['genes']), len(state['classes'])
    x, g, continuous = np.empty((total, 2), np.float32), np.empty((total, gd), np.float32), np.empty((total, cd), np.float32)
    supported = sched['target'] >= 0
    for forward in (True, False):
        src, dst = (left, right) if forward else (right, left)
        children = np.flatnonzero(sched['forward'] == forward)
        p, j = sched['parent'][children], sched['target'][children]
        anchor = (j < 0) & (not baseline)
        aid, ap = children[anchor], p[anchor]
        x[aid], g[aid] = src['x'][ap], src['g'][ap]
        continuous[aid] = np.eye(cd, dtype=np.float32)[src['y'][ap]]
        children, p, j = children[~anchor], p[~anchor], j[~anchor]
        if not len(children):
            continue
        context = getattr(model, 'context', False)
        groups, inverse = np.unique(np.column_stack((p, j if context else np.full_like(j, -1))), axis=0, return_inverse=True)
        end_t = sched['fraction'] if forward else 1-sched['fraction']
        times = torch.linspace(0, end_t, cfg['steps'], device=device)
        for start in range(0, len(groups), cfg['chunk_size']):
            part = groups[start:start+cfg['chunk_size']]; parents, targets = part.T
            targets = np.maximum(targets, 0)
            tensor = lambda values: torch.as_tensor(np.asarray(values).copy(), dtype=torch.float32, device=device)
            source_c = tensor(np.eye(cd, dtype=np.float32)[src['y'][parents]])
            initial = torch.cat((tensor(src['x'][parents]), tensor(src['g'][parents]), source_c), -1)
            h0, h1 = tensor(src['h'][parents]), tensor(dst['h'][targets])
            delta = torch.full((len(part), 1), dst['z']-src['z'], device=device)
            def field(t, values):
                out = model(xt=values[:, :2], gt=values[:, 2:2+gd], ct=values[:, 2+gd:],
                            t=t.expand(len(part)), zt=src['z']+t*delta, delta_z=delta,
                            h0=h0, h1=h1, source_c=source_c)
                return torch.cat(out, -1)
            result = odeint(field, initial, times, method=cfg['solver'], atol=cfg['atol'], rtol=cfg['rtol'])[-1]
            if not torch.isfinite(result).all():
                raise FloatingPointError('Nonfinite generated state.')
            values = result.cpu().numpy()
            mask = (inverse >= start) & (inverse < start+len(part))
            selection = values[inverse[mask]-start]
            ids = children[mask]
            x[ids], g[ids], continuous[ids] = selection[:, :2], selection[:, 2:2+gd], selection[:, 2+gd:]
    before = g.copy()
    parents, forward = sched['parent'], sched['forward']
    if cfg['sparsify']:
        nnz = np.empty(total, int)
        nnz[forward] = (left['g'][parents[forward]] > 0).sum(1)
        nnz[~forward] = (right['g'][parents[~forward]] > 0).sum(1)
        for start in range(0, total, 256):
            chunk = g[start:start+256]
            order = np.argsort(-chunk, axis=1, kind='stable')
            keep = np.zeros(chunk.shape, bool)
            np.put_along_axis(keep, order, np.arange(gd)[None, :] < nnz[start:start+len(chunk), None], axis=1)
            chunk[~keep] = 0
    physical = x.astype(float)*state['xy_range']+state['xy_min']
    source_y = np.empty(total, int); source_id = np.empty(total, object); source_z = np.empty(total)
    for side, source in ((True, left), (False, right)):
        mask = forward == side
        source_y[mask], source_id[mask], source_z[mask] = source['y'][parents[mask]], source['ids'][parents[mask]], source['z_um']
        anchor = mask & ~supported & (not baseline)
        physical[anchor] = source['physical'][parents[anchor]]
        g[anchor] = source['g'][parents[anchor]]; before[anchor] = g[anchor]
    names = np.asarray(state['classes'])
    obs = pd.DataFrame(dict(cell_class=names[continuous.argmax(-1)], source_cell_class=names[source_y],
                            source_cell_id=source_id, source_z=source_z, source_cell_index=parents,
                            source_direction=np.where(forward, 'forward', 'backward'), target_cell_index=sched['target'],
                            z_um=sched['z_um'], generation_mode=np.where(~supported & (not baseline), 'source_anchor', 'ode'),
                            parent_id=[f'{z:.9g}:{i}' for z, i in zip(source_z, source_id)]),
                       index=[f'generated_{i}' for i in range(total)])
    return ad.AnnData(X=sparse.csr_matrix(g), obs=obs,
                      var=pd.DataFrame({'gene_symbol': state['symbols']}, index=state['genes']),
                      obsm={'spatial': physical, 'spatial_3d': np.column_stack((physical, np.full(total, sched['z_um']))),
                            'cell_class_continuous': continuous},
                      layers={'expression_before_sparsification': before},
                      uns={'generation': dict(target_depth_um=sched['z_um'], budget_source='linear_visible_endpoint_counts',
                                              coordinate_units='um', annotation_origin='model_continuous_class_state',
                                              type_state_is_calibrated_probability=False)})


def write_h5ad(path, result):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem+'.tmp.h5ad')
    result.write_h5ad(temporary, compression='gzip')
    os.replace(temporary, path)
