"""Sparse physical-support UOT and disk-blocked original global UOT.

Global kernels are temporary, one gap at a time. Sampled training edges and
Sinkhorn factors survive; the full N0 x N1 coupling never resides in RAM.
"""
from collections import OrderedDict
import hashlib
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree, distance

from .io import digest, read_json, write_json


def corrected(plan, kappa):
    if not 0 <= kappa <= 1:
        raise ValueError('kappa must be in [0,1].')
    p = sparse.csr_matrix(plan, dtype=np.float64).copy()
    p.eliminate_zeros()
    if not np.isfinite(p.data).all() or (p.data < 0).any():
        raise ValueError('Invalid coupling mass.')
    if p.sum() <= 0:
        return p
    p /= p.sum()
    mass = np.asarray(p.sum(1)).ravel()
    valid = mass > 0
    inverse = np.divide(1/valid.sum(), mass, out=np.zeros_like(mass), where=valid)
    return ((1-kappa)*p + kappa*sparse.diags(inverse)@p).tocsr()


def sinkhorn(kernel, tau, reg, tolerance, iterations, block=256):
    """KL-reference UOT: K=a*b*exp(-C/reg), a=1/N, b=1/M.

    Works on CSR or memmap. Stopping uses relative scaling changes; failure is
    explicit rather than recording a nonconverged plan as a finished run.
    """
    n, m = kernel.shape
    u, v = np.ones(n), np.ones(m)
    exponent = tau/(tau+reg)
    active_rows = np.asarray(kernel.sum(1)).ravel() > 0
    active_cols = np.asarray(kernel.sum(0)).ravel() > 0
    if not active_rows.any():
        return u*0, v*0, dict(iterations=0, relative_error=0.)
    for iteration in range(iterations):
        old_u, old_v = u.copy(), v.copy()
        if sparse.issparse(kernel):
            kv = kernel@v
        else:
            kv = np.empty(n)
            for start in range(0, n, block):
                kv[start:start+block] = kernel[start:start+block]@v
        u = np.power(np.divide(1/n, kv, out=np.zeros(n), where=kv > 0), exponent)
        if sparse.issparse(kernel):
            ktu = kernel.T@u
        else:
            ktu = np.zeros(m)
            for start in range(0, n, block):
                ktu += u[start:start+block]@kernel[start:start+block]
        v = np.power(np.divide(1/m, ktu, out=np.zeros(m), where=ktu > 0), exponent)
        err = max(np.max(np.abs(u-old_u)/np.maximum(np.abs(old_u), 1e-100)),
                  np.max(np.abs(v-old_v)/np.maximum(np.abs(old_v), 1e-100)))
        if not np.isfinite(u).all() or not np.isfinite(v).all():
            raise FloatingPointError('Nonfinite UOT scaling. Inspect coordinate units/costs.')
        if err < tolerance:
            return u, v, dict(iterations=iteration+1, relative_error=float(err))
    raise RuntimeError(f'UOT did not converge in {iterations} iterations: relative error {err:g}')


def local_uot(left, right, config):
    cfg = config['pairing']
    factor = abs(right['z_um']-left['z_um'])/cfg['reference_gap_um']
    radius, scale = cfg['radius_um']*factor, cfg['spatial_scale_um']*factor
    rows, cols, distances = [], [], []
    for label in np.intersect1d(left['y'], right['y']):
        a, b = np.flatnonzero(left['y'] == label), np.flatnonzero(right['y'] == label)
        edges = cKDTree(left['physical'][a]).sparse_distance_matrix(cKDTree(right['physical'][b]), radius, output_type='coo_matrix')
        rows.append(a[edges.row]); cols.append(b[edges.col]); distances.append(edges.data)
    shape = len(left['y']), len(right['y'])
    if not rows or not sum(map(len, rows)):
        return sparse.csr_matrix(shape), dict(radius_um=radius, edges=0, iterations=0)
    row, col, d = np.concatenate(rows), np.concatenate(cols), np.concatenate(distances)
    norms = [np.maximum(np.linalg.norm(s['g'], axis=1), 1e-12) for s in (left, right)]
    cosine = np.empty(len(row))
    for start in range(0, len(row), 2048):
        a, b = row[start:start+2048], col[start:start+2048]
        cosine[start:start+2048] = np.clip(1-np.einsum('ij,ij->i', left['g'][a], right['g'][b])/(norms[0][a]*norms[1][b]), 0, 2)
    # Cosine is already bounded; no O(N*M*G) all-pairs normalization in local UOT.
    cost = cfg['alpha_spatial']*(d/scale)**2 + (1-cfg['alpha_spatial'])*cosine
    kernel = sparse.csr_matrix((np.exp(-cost/cfg['local_reg'])/(shape[0]*shape[1]), (row, col)), shape=shape)
    kernel.eliminate_zeros()
    u, v, log = sinkhorn(kernel, cfg['tau'], cfg['local_reg'], cfg['tolerance'], cfg['max_iter'])
    plan = (sparse.diags(u)@kernel@sparse.diags(v)).tocsr()
    plan.eliminate_zeros()
    return plan, dict(log, radius_um=radius, edges=plan.nnz)


def _raw_cost(left, right, start, stop, scales, alpha):
    xy = distance.cdist(left['x'][start:stop], right['x'])/max(scales['spatial_max'], 1e-9)
    a = np.asarray(left['g'][start:stop], np.float64)
    b = np.asarray(right['g'], np.float64)
    an, bn = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    cosine = 1-(a@b.T)/np.maximum(an[:, None]*bn[None, :], 1e-12)
    cosine = np.clip(cosine, 0, 2)/max(scales['gene_max'], 1e-9)
    mismatch = left['y'][start:stop, None] != right['y'][None, :]
    return alpha*xy+(1-alpha)*cosine+10.*mismatch


def global_uot_samples(left, right, config, folder, count, rng):
    """Original global cost, uniform marginals, KL UOT, finite sampled trajectory bank."""
    cfg = config['pairing']; folder = Path(folder)
    n, m = len(left['y']), len(right['y']); block = cfg['block_size']
    scratch = folder/'global_kernel.tmp.npy'
    required = n*m*8
    if shutil.disk_usage(folder).free < required + 256*1024**2:
        raise OSError(f'Global baseline needs {required/1024**3:.2f} GiB temporary disk for this gap.')
    scales = dict(spatial_max=0., gene_max=0.)
    norm_b = np.maximum(np.linalg.norm(right['g'], axis=1), 1e-12)
    for start in range(0, n, block):
        a = np.asarray(left['g'][start:start+block], np.float64)
        cosine = np.clip(1-(a@np.asarray(right['g'], np.float64).T)/np.maximum(np.linalg.norm(a, axis=1)[:, None]*norm_b[None, :], 1e-12), 0, 2)
        scales['gene_max'] = max(scales['gene_max'], float(cosine.max()))
        scales['spatial_max'] = max(scales['spatial_max'], float(distance.cdist(left['x'][start:start+block], right['x']).max()))
    kernel = np.lib.format.open_memmap(scratch, mode='w+', dtype=np.float64, shape=(n, m))
    try:
        for start in range(0, n, block):
            cost = _raw_cost(left, right, start, start+block, scales, cfg['alpha_spatial'])
            kernel[start:start+block] = np.exp(-cost/cfg['raw_reg'])/(n*m)
        kernel.flush()
        u, v, log = sinkhorn(kernel, cfg['tau'], cfg['raw_reg'], cfg['tolerance'], cfg['max_iter'], block)
        mass = np.empty(n)
        for start in range(0, n, block):
            mass[start:start+block] = u[start:start+block]*(kernel[start:start+block]@v)
        row = rng.choice(n, count, p=mass/mass.sum())
        col = np.empty(count, np.int64)
        for i in np.unique(row):
            mask = row == i
            w = np.asarray(kernel[i])*v
            col[mask] = rng.choice(m, int(mask.sum()), p=w/w.sum())
        np.savez_compressed(folder/'raw_factors.npz', u=u, v=v, **scales)
        return row, col, dict(log, **scales, scratch_bytes=required)
    finally:
        del kernel
        scratch.unlink(missing_ok=True)


class Transports:
    def __init__(self, cache, store, config):
        self.root, self.store, self.config = Path(cache)/'transport', store, config
        self.root.mkdir(exist_ok=True)
        self.loaded = OrderedDict()

    def folder(self, left, right, kind):
        folder = self.root/f'{left}__{right}'/kind
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def local(self, left, right):
        key = left, right
        if key in self.loaded:
            self.loaded.move_to_end(key)
            return self.loaded[key]
        folder = self.folder(left, right, 'local'); path = folder/'plan.npz'
        if (folder/'complete.json').exists():
            if read_json(folder/'complete.json')['sha256'] != digest(path):
                raise ValueError('Local transport cache changed.')
            plan = sparse.load_npz(path)
            self.loaded[key] = plan
            if len(self.loaded) > 8:
                self.loaded.popitem(last=False)
            return plan
        plan, log = local_uot(self.store[left], self.store[right], self.config)
        sparse.save_npz(path, plan)
        write_json(folder/'complete.json', dict(log, sha256=digest(path)))
        a, b = self.store[left], self.store[right]
        diagnostics = []
        for direction, src, p in (('forward', a, plan), ('backward', b, plan.T.tocsr())):
            mass = np.asarray(p.sum(1)).ravel()
            qm = np.asarray(corrected(p, self.config['A']['kappa']).sum(1)).ravel()
            for c in np.unique(src['y']):
                mask = src['y'] == c
                diagnostics.append(dict(direction=direction, subclass=self.store.state['classes'][int(c)], n=int(mask.sum()),
                                        supported_n=int((mass[mask] > 0).sum()), input_fraction=float(mask.mean()),
                                        uot_fraction=float(mass[mask].sum()/mass.sum()) if mass.sum() else 0., corrected_fraction=float(qm[mask].sum())))
        pd.DataFrame(diagnostics).to_csv(folder/'support.csv', index=False)
        self.loaded[key] = plan
        if len(self.loaded) > 8:
            self.loaded.popitem(last=False)
        return plan

    def raw(self, left, right, count):
        folder = self.folder(left, right, 'raw'); path = folder/'samples.npz'
        if (folder/'complete.json').exists():
            log = read_json(folder/'complete.json')
            if log['samples_sha256'] != digest(path) or log['factors_sha256'] != digest(folder/'raw_factors.npz') or log['sample_count'] != count:
                raise ValueError('Raw transport cache identity changed.')
        else:
            seed = self.config['seed'] + int.from_bytes(hashlib.sha256((left+'|'+right).encode()).digest()[:4], 'little')
            row, col, log = global_uot_samples(self.store[left], self.store[right], self.config, folder, count, np.random.default_rng(seed))
            np.savez_compressed(path, row=row, col=col)
            write_json(folder/'complete.json', dict(log, sample_count=count, samples_sha256=digest(path), factors_sha256=digest(folder/'raw_factors.npz')))
        with np.load(path) as data:
            return data['row'], data['col']

    def targets(self, left, right, forward, parents, kind, rng):
        """Sample counterpart only from observed training endpoints, including raw no-A."""
        a, b = self.store[left], self.store[right]
        result = np.full(len(parents), -1, np.int64)
        if kind == 'local':
            plan = self.local(left, right)
            matrix = plan if forward else plan.T.tocsr()
            for i in np.unique(parents):
                mask = parents == i
                start, stop = matrix.indptr[i:i+2]
                w = matrix.data[start:stop]
                if w.sum() > 0:
                    result[mask] = rng.choice(matrix.indices[start:stop], int(mask.sum()), p=w/w.sum())
        else:
            folder = self.folder(left, right, 'raw')
            with np.load(folder/'raw_factors.npz') as archive:
                values = {k: archive[k] for k in archive.files}
            cfg = self.config['pairing']
            src, dst = (a, b) if forward else (b, a)
            dual = values['v'] if forward else values['u']
            unique = np.unique(parents)
            for start in range(0, len(unique), cfg['block_size']):
                selected = unique[start:start+cfg['block_size']]
                subset = {key: src[key][selected] for key in ('x', 'g', 'y')}
                costs = _raw_cost(subset, dst, 0, len(selected), values, cfg['alpha_spatial'])
                for index, i in enumerate(selected):
                    w = np.exp(-costs[index]/cfg['raw_reg'])*dual
                    mask = parents == i
                    result[mask] = rng.choice(len(w), int(mask.sum()), p=w/w.sum())
        return result
