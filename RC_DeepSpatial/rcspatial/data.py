"""Train-only preparation. Hidden slice contents are loaded by evaluation only."""
import glob
from pathlib import Path
import re

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree

from .vendor.niche import NicheFeaturizer


def dense(adata):
    return (adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)).astype(np.float32)


def discover(config, project):
    paths = [Path(p).resolve() for p in glob.glob(str(project / config['input_glob']), recursive=True)]
    paths.sort(key=lambda p: [int(v) if v.isdigit() else v for v in re.split(r'(\d+)', p.name)])
    if len(paths) != len(config['slice_z_um']) or len({p.name for p in paths}) != len(paths):
        raise ValueError('Expected exactly one input for each configured depth; check the MERFISH glob.')
    if not np.all(np.diff(config['slice_z_um']) > 0):
        raise ValueError('Physical slice depths must increase strictly.')
    return paths


def validate(a, cfg, genes=None):
    if not a.n_obs or not a.obs_names.is_unique or not a.var_names.is_unique:
        raise ValueError('Require nonempty slices and unique cell/gene identifiers.')
    if genes is not None and a.var_names.tolist() != list(genes):
        raise ValueError('Gene names and order must equal the visible training vocabulary.')
    key, xy = cfg['label_key'], np.asarray(a.obsm[cfg['spatial_key']])
    if key not in a.obs or a.obs[key].isna().any():
        raise ValueError('Require nonmissing observed cell-type labels.')
    if xy.shape != (a.n_obs, 2) or not np.isfinite(xy).all():
        raise ValueError('Require finite N by 2 physical XY coordinates.')
    g = dense(a)
    if not np.isfinite(g).all() or (g < 0).any():
        raise ValueError('MERFISH input X must be finite nonnegative expression; no implicit transform.')
    return g


def geometry_features(xy, radius, sigma):
    """Relative first/second XY moments; self edges are excluded."""
    xy = np.asarray(xy, float)
    edges = cKDTree(xy).sparse_distance_matrix(cKDTree(xy), radius, output_type='coo_matrix')
    keep = edges.row != edges.col
    row, col, d = edges.row[keep], edges.col[keep], edges.data[keep]
    w = np.exp(-.5 * (d / sigma) ** 2)
    mass = np.bincount(row, weights=w, minlength=len(xy))
    delta = (xy[col] - xy[row]) / radius
    out = np.zeros((len(xy), 4), dtype=np.float32)
    for k, values in enumerate((delta[:, 0], delta[:, 1], delta[:, 0]**2, delta[:, 1]**2)):
        np.divide(np.bincount(row, weights=w * values, minlength=len(xy)), mass,
                  out=out[:, k], where=mass > 0)
    return out


def prepare(paths, config, heldout, excluded=()):
    """Exclude hidden inputs BEFORE reading; fit all statistics on visible slices."""
    if heldout <= 0 or heldout >= len(paths)-1:
        raise ValueError('Leave out an interior slice so two measured endpoints remain.')
    hidden = {int(heldout), *map(int, excluded)}
    indices = [i for i in range(len(paths)) if i not in hidden]
    if len(indices) < 2 or not any(i<heldout for i in indices) or not any(i>heldout for i in indices):
        raise ValueError('Visible training set must bracket the reconstruction depth.')
    cfg = config['data']
    adatas, arrays, genes = [], [], None
    for i in indices:
        a = ad.read_h5ad(paths[i])
        g = validate(a, cfg, genes)
        genes = a.var_names.tolist()
        a.obs[cfg['z_key']] = float(config['slice_z_um'][i])
        adatas.append(a)
        arrays.append(g)
    labels = np.concatenate([a.obs[cfg['label_key']].astype(str).to_numpy() for a in adatas])
    classes, counts = np.unique(labels, return_counts=True)
    xy = np.vstack([a.obsm[cfg['spatial_key']] for a in adatas]).astype(float)
    depths = np.array([config['slice_z_um'][i] for i in indices], float)
    state = dict(genes=genes, classes=classes.tolist(), counts=counts.tolist(),
                 xy_min=xy.min(0).tolist(), xy_range=np.maximum(np.ptp(xy, axis=0), 1e-8).tolist(),
                 z_min=float(depths.min()), z_range=float(np.ptp(depths)),
                 gene_scale=np.maximum(np.vstack(arrays).std(0), 1e-3).tolist(),
                 train_indices=indices, heldout=int(heldout), excluded=sorted(hidden-{heldout}))
    # Measured genes are features, not invented biological gene programs.
    niche_cfg = config['niche']
    programs = {f'measured::{gene}': {gene: 1.0} for gene in genes}
    featurizer = NicheFeaturizer(radius=niche_cfg['radius_um'], sigma=niche_cfg['sigma_um'],
                               programs=programs, expression_transform='log1p').fit(adatas, cfg['label_key'])
    state['niche'] = featurizer.to_dict()
    state['niche_features'] = featurizer.feature_names + ['relative_dx', 'relative_dy', 'relative_dx2', 'relative_dy2']
    slices = []
    for i, a, g in zip(indices, adatas, arrays):
        physical = np.asarray(a.obsm[cfg['spatial_key']], dtype=float)
        h = np.column_stack([featurizer.transform(a, cfg['spatial_key'], cfg['label_key']),
                             geometry_features(physical, niche_cfg['radius_um'], niche_cfg['sigma_um'])]).astype(np.float32)
        y = pd.Categorical(a.obs[cfg['label_key']].astype(str), categories=classes).codes.astype(np.int64)
        slices.append(dict(index=i, ids=a.obs_names.to_numpy().astype(str), physical=physical, g=g, y=y, h=h,
                           x=((physical-state['xy_min'])/state['xy_range']).astype(np.float32),
                           z=float((config['slice_z_um'][i]-state['z_min'])/state['z_range']),
                           z_um=float(config['slice_z_um'][i])))
    return slices, state


def context_view(slices, mode, seed):
    """Same-type, same-slice permutation preserves shape and model capacity."""
    if mode not in ('real', 'zero', 'shuffle', 'no_geometry'):
        raise ValueError(f'Unknown context mode: {mode}')
    result = []
    for s in slices:
        h = s['h'].copy()
        if mode == 'zero':
            h.fill(0)
        elif mode == 'no_geometry':
            h[:, -4:] = 0
        elif mode == 'shuffle':
            rng = np.random.default_rng(np.random.SeedSequence([seed, s['index'], 713]))
            for c in np.unique(s['y']):
                ix = np.flatnonzero(s['y'] == c)
                h[ix] = h[rng.permutation(ix)]
        result.append(dict(s, h=h))
    return result
