"""Finite trajectory banks contain indices, never duplicated whole-brain expression."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from .io import digest, read_json, write_json
from .transport import corrected


class PairDataset(Dataset):
    def __init__(self, store, transports, state, config, arm):
        self.store, self.state = store, state
        self.gaps = [(a['section'], b['section']) for a, b in zip(state['train'], state['train'][1:])]
        self.kind = 'raw' if arm.get('raw', False) else 'local'
        self.bidirectional = not arm.get('baseline', False)
        active_a = arm.get('A', False)
        options = dict(kind=self.kind, bidirectional=self.bidirectional, A=active_a)
        name = f'bank_{self.kind}_{int(self.bidirectional)}_{int(active_a)}'
        folder = Path(transports.root).parent/'banks'; folder.mkdir(exist_ok=True)
        path, stamp = folder/(name+'.npz'), folder/(name+'.json')
        if stamp.exists():
            record = read_json(stamp)
            if record['sha256'] != digest(path) or record['options'] != options:
                raise ValueError('Trajectory bank changed.')
            with np.load(path) as archive:
                self.indices = {k: archive[k] for k in archive.files}
            return
        rng = np.random.default_rng(config['seed'])
        sizes = np.array([store.rows[a]['n_cells']*store.rows[b]['n_cells'] for a, b in self.gaps], dtype=float)
        expected = config['pairing']['n_samples']*sizes/sizes.sum()
        budgets = np.floor(expected).astype(int)
        budgets[np.argsort(-(expected-budgets), kind='stable')[:config['pairing']['n_samples']-budgets.sum()]] += 1
        counts = np.asarray(state['counts']); rare = counts/counts.sum() <= config['evaluation']['rare_fraction']
        records, diagnostics = [], []
        for gap, ((lname, rname), budget) in enumerate(zip(self.gaps, budgets)):
            if budget == 0:
                continue
            left, right = store[lname], store[rname]
            if self.kind == 'raw':
                i, j = transports.raw(lname, rname, int(budget))
                reverse = rng.random(budget) < .5 if self.bidirectional else np.zeros(budget, bool)
                weights = np.ones(budget, np.float32)
                records.append(dict(gap=np.full(budget, gap, np.int32), i=i, j=j, reverse=reverse, weight=weights))
                continue
            plan = transports.local(lname, rname)
            if plan.sum() <= 0:
                diagnostics.append(dict(gap=gap, status='no_supported_edges', lost_budget=int(budget)))
                continue
            nr = int(rng.binomial(int(budget), .5)) if self.bidirectional else 0
            for reverse, count in ((False, int(budget)-nr), (True, nr)):
                if not count:
                    continue
                p = plan.T.tocsr() if reverse else plan
                src = right if reverse else left
                q = corrected(p, config['A']['kappa'] if active_a else 0).tocoo()
                cls = src['y'][q.row]
                class_mass = np.bincount(cls, weights=q.data, minlength=len(counts))
                tail = rare & (class_mass > 0)
                eta = config['A']['risk_fraction'] if active_a and tail.any() else 0.
                risk = np.full(len(q.data), 1-eta)
                if eta:
                    selected = tail[cls]
                    risk[selected] += eta/(tail.sum()*class_mass[cls[selected]])
                choice = rng.choice(len(q.data), count, p=q.data/q.data.sum())
                a, b = q.row[choice], q.col[choice]
                i, j = (b, a) if reverse else (a, b)
                weights = np.minimum(risk[choice], config['A']['weight_cap']).astype(np.float32)
                records.append(dict(gap=np.full(count, gap, np.int32), i=i, j=j, reverse=np.full(count, reverse), weight=weights))
                for c in np.unique(cls):
                    keep = src['y'][a] == c
                    diagnostics.append(dict(gap=gap, direction='backward' if reverse else 'forward', subclass=state['classes'][c],
                                            status='sampled', sampled_n=int(keep.sum()), unique_source_n=len(np.unique(a[keep])),
                                            unique_target_n=len(np.unique(b[keep])), clipped_n=int((risk[choice][keep] > config['A']['weight_cap']).sum()),
                                            class_proposal_mass=float(class_mass[c]), mean_weight=float(weights[keep].mean()) if keep.any() else np.nan))
        if not records:
            raise ValueError('No supported training trajectories.')
        self.indices = {key: np.concatenate([r[key] for r in records]) for key in records[0]}
        np.savez_compressed(path, **self.indices)
        pd.DataFrame(diagnostics).to_csv(folder/(name+'_sampling.csv'), index=False)
        write_json(stamp, dict(options=options, sha256=digest(path), n=len(self), gaps=self.gaps,
                              effective_weight_n=float(self.indices['weight'].sum()**2/(self.indices['weight']**2).sum())))

    def __len__(self):
        return len(self.indices['i'])

    def __getitem__(self, index):
        gap = int(self.indices['gap'][index])
        a, b = (self.store[name] for name in self.gaps[gap])
        i, j = int(self.indices['i'][index]), int(self.indices['j'][index])
        if self.indices['reverse'][index]:
            a, b, i, j = b, a, j, i
        result = {k+'0': np.asarray(a[k][i]).copy() for k in ('x', 'g', 'h')}
        result.update({k+'1': np.asarray(b[k][j]).copy() for k in ('x', 'g', 'h')})
        result.update(c0=int(a['y'][i]), c1=int(b['y'][j]), z0=np.float32(a['z']), z1=np.float32(b['z']),
                      weight=self.indices['weight'][index])
        return result


class GapBatches(Sampler):
    """Shuffle gaps and rows each epoch while keeping disk reads local per batch."""
    def __init__(self, dataset, batch_size, seed, epoch):
        self.dataset, self.batch_size, self.seed, self.epoch = dataset, batch_size, seed, epoch

    def __iter__(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, 9301]))
        for gap in rng.permutation(np.unique(self.dataset.indices['gap'])):
            indices = rng.permutation(np.flatnonzero(self.dataset.indices['gap'] == gap))
            for start in range(0, len(indices), self.batch_size):
                yield indices[start:start+self.batch_size].tolist()

    def __len__(self):
        return sum((int((self.dataset.indices['gap'] == g).sum())+self.batch_size-1)//self.batch_size
                   for g in np.unique(self.dataset.indices['gap']))


def interpolate(batch, state, device, times=None):
    b = {k: v.to(device) for k, v in batch.items()}
    c0 = torch.nn.functional.one_hot(b['c0'], len(state['classes'])).float()
    c1 = torch.nn.functional.one_hot(b['c1'], len(state['classes'])).float()
    t = torch.rand(len(c0), device=device) if times is None else times
    t2 = t[:, None]
    z0, z1 = b['z0'][:, None], b['z1'][:, None]
    inputs = dict(xt=(1-t2)*b['x0']+t2*b['x1'], gt=(1-t2)*b['g0']+t2*b['g1'],
                  ct=(1-t2)*c0+t2*c1, zt=(1-t2)*z0+t2*z1, delta_z=z1-z0,
                  t=t, h0=b['h0'], h1=b['h1'], source_c=c0)
    targets = b['x1']-b['x0'], b['g1']-b['g0'], c1-c0
    return inputs, targets, b['weight']
