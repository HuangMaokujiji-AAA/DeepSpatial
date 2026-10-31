"""Module A: directed marginal correction, separate from class-risk weighting."""
import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.utils.data import Dataset
import warnings

from .vendor.local_matching import compute_local_uot
from .vendor.uot_solver import compute_uot_coupling


def directed_distribution(coupling, kappa):
    """Return q=(1-kappa)*pi/Z+kappa*uniform_supported_rows*K.

    Only this directed SOURCE marginal is calibrated, never both marginals.
    Unsupported rows retain zero mass. No epsilon creates forbidden edges.
    """
    if not np.isfinite(kappa) or not 0 <= kappa <= 1:
        raise ValueError('kappa must lie in [0,1].')
    p = sparse.csr_matrix(coupling, dtype=np.float64).copy()
    p.eliminate_zeros()
    if not np.isfinite(p.data).all() or (p.data < 0).any():
        raise ValueError('Coupling must be finite and nonnegative.')
    if p.sum() <= 0:
        return p
    p /= float(p.sum())
    mass = np.asarray(p.sum(1)).ravel()
    active = mass > 0
    inv = np.divide(1/active.sum(), mass, out=np.zeros_like(mass), where=active)
    q = (1-kappa)*p + kappa*sparse.diags(inv)@p
    return q.tocsr()


def build_plans(slices, config, kind):
    """Pairing is shared between arms; raw and physical-support controls differ explicitly."""
    cfg = config['pairing']
    plans = []
    classes = max(int(s['y'].max()) for s in slices)+1
    for left, right in zip(slices, slices[1:]):
        if kind == 'raw':
            # Same upstream objective/solver; float64 prevents exp(-10/.05)
            # underflow when a class has no counterpart in the other slice.
            eye = np.eye(classes, dtype=np.float64)
            with warnings.catch_warnings():
                warnings.filterwarnings('error',message='.*[Nn]umerical.*')
                warnings.filterwarnings('error',category=RuntimeWarning)
                p = sparse.csr_matrix(compute_uot_coupling(
                    left['x'].astype(np.float64), left['g'].astype(np.float64), eye[left['y']],
                    right['x'].astype(np.float64), right['g'].astype(np.float64), eye[right['y']],
                    cfg['alpha_spatial'], cfg['uot_reg'], cfg['uot_tau']))
        elif kind == 'local':
            p, _ = compute_local_uot(left['physical'], left['g'], left['y'],
                                    right['physical'], right['g'], right['y'], right['z_um']-left['z_um'],
                                    cfg['local'], cfg['alpha_spatial'], cfg['uot_reg'], cfg['uot_tau'])
        else:
            raise ValueError(f'Unknown pairing kind: {kind}')
        p.eliminate_zeros()
        if not np.isfinite(p.data).all() or (p.data < 0).any():
            raise ValueError('UOT returned an invalid plan.')
        plans.append(p)
    if sum(p.sum() for p in plans) <= 0:
        raise ValueError('No supported training edges; no synthetic pairs can be trained.')
    return plans


def audit_plans(slices, plans, state, kappa):
    rows = []
    for gap, p in enumerate(plans):
        for reverse in (False, True):
            s = slices[gap+int(reverse)]
            a = p.T.tocsr() if reverse else p
            mass = np.asarray(a.sum(1)).ravel()
            q = directed_distribution(a, kappa)
            qmass = np.asarray(q.sum(1)).ravel()
            for c, name in enumerate(state['classes']):
                mask = s['y'] == c
                weights = mass[mask]
                rows.append(dict(gap=gap, source_slice=s['index'], direction='backward' if reverse else 'forward',
                                 cell_class=name, input_n=int(mask.sum()), input_fraction=float(mask.mean()),
                                 supported_n=int((mask & (mass>0)).sum()),
                                 uot_mass=float(weights.sum()),
                                 uot_fraction=float(weights.sum()/mass.sum()) if mass.sum() else 0.,
                                 corrected_fraction=float(qmass[mask].sum()),
                                 source_ess=float(weights.sum()**2/(weights@weights)) if weights@weights else 0.))
    return pd.DataFrame(rows)


def support_diagnostics(slices, plans, state):
    """Per-endpoint mass/entropy/distance on the existing local graph."""
    records=[]
    for gap,p in enumerate(plans):
        for reverse in (False,True):
            src,dst=(slices[gap+1],slices[gap]) if reverse else (slices[gap],slices[gap+1])
            matrix=(p.T if reverse else p).tocsr()
            for i in range(matrix.shape[0]):
                start,end=matrix.indptr[i:i+2]
                w=matrix.data[start:end];j=matrix.indices[start:end];mass=float(w.sum())
                k=w/mass if mass else w
                distances=np.linalg.norm(dst['physical'][j]-src['physical'][i],axis=1)
                records.append(dict(gap=gap,source_slice=src['index'],target_slice=dst['index'],
                                    source_cell_id=src['ids'][i],cell_class=state['classes'][src['y'][i]],
                                    supported_edges=end-start,transport_mass=mass,
                                    conditional_entropy=float(-(k*np.log(k)).sum()) if mass else np.nan,
                                    nearest_candidate_um=float(distances.min()) if mass else np.nan,
                                    weighted_distance_um=float(k@distances) if mass else np.nan))
    return pd.DataFrame(records)


class Trajectories(Dataset):
    """Fixed finite trajectory dataset; direction is sampled BEFORE its own correction."""
    def __init__(self, slices, plans, state, total, seed, kappa=0., bidirectional=True,
                 balanced_risk=0., weight_cap=10.):
        if total < 1 or not 0 <= balanced_risk <= 1 or weight_cap < 1:
            raise ValueError('Invalid trajectory budget, risk mixture or cap.')
        rng = np.random.default_rng(seed)
        sizes = np.array([len(slices[i]['g'])*len(slices[i+1]['g']) if p.sum()>0 else 0
                          for i, p in enumerate(plans)], float)
        expected = total*sizes/sizes.sum()
        allocated = np.floor(expected).astype(int)
        allocated[np.argsort(-(expected-allocated), kind='stable')[:total-allocated.sum()]] += 1
        records, diagnostics = [], []
        for gap, n in enumerate(allocated):
            nrev = int(rng.binomial(int(n), .5)) if bidirectional else 0
            for reverse, count in ((False, int(n)-nrev), (True, nrev)):
                if not count:
                    continue
                s0, s1 = (slices[gap+1], slices[gap]) if reverse else (slices[gap], slices[gap+1])
                p = plans[gap].T.tocsr() if reverse else plans[gap]
                q = directed_distribution(p, kappa).tocoo()
                if not len(q.data):
                    raise ValueError('Allocated trajectories to an empty gap.')
                cls = s0['y'][q.row]
                cmass = np.bincount(cls, weights=q.data, minlength=len(state['classes']))
                present = cmass > 0
                # Proposal is q, not balanced sampling. Weight ONCE to target mixed risk.
                w = (1-balanced_risk)+balanced_risk/np.maximum(present.sum()*cmass[cls], 1e-300)
                choice = rng.choice(len(q.data), count, p=q.data/q.data.sum())
                i, j = q.row[choice], q.col[choice]
                selected_weight = np.minimum(w[choice], weight_cap).astype(np.float32)
                records.append(dict(x0=s0['x'][i], x1=s1['x'][j], g0=s0['g'][i], g1=s1['g'][j],
                                    c0=np.eye(len(state['classes']), dtype=np.float32)[s0['y'][i]],
                                    c1=np.eye(len(state['classes']), dtype=np.float32)[s1['y'][j]],
                                    h0=s0['h'][i], h1=s1['h'][j],
                                    z0=np.full((count,1), s0['z'], np.float32),
                                    z1=np.full((count,1), s1['z'], np.float32), weight=selected_weight,
                                    slice0=np.full(count,s0['index']), cell0=i, slice1=np.full(count,s1['index']), cell1=j))
                for c, name in enumerate(state['classes']):
                    keep = s0['y'][i] == c
                    diagnostics.append(dict(gap=gap, direction='backward' if reverse else 'forward', cell_class=name,
                                            expected_fraction=float(cmass[c]), sampled_n=int(keep.sum()),
                                            unique_source_n=int(len(np.unique(i[keep]))),
                                            unique_target_n=int(len(np.unique(j[keep]))),
                                            mean_risk_weight=float(selected_weight[keep].mean()) if keep.any() else np.nan,
                                            clipped_n=int((w[choice][keep]>weight_cap).sum())))
        self.tensors = {key: torch.from_numpy(np.concatenate([r[key] for r in records])) for key in records[0]}
        self.diagnostics = pd.DataFrame(diagnostics)
        self.ess = float(self.tensors['weight'].sum().square()/self.tensors['weight'].square().sum())

    def __len__(self):
        return len(self.tensors['g0'])

    def __getitem__(self, index):
        return {key: value[index] for key, value in self.tensors.items()}
