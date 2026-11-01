"""Visible-only spatial grids and position-conditioned endpoint neighborhoods."""
import glob
from pathlib import Path
import re

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree


def dense(a):
    return (a.X.toarray() if sparse.issparse(a.X) else np.asarray(a.X)).astype(np.float32)


def discover(config, project):
    paths = [Path(p).resolve() for p in glob.glob(str(project / config['input_glob']), recursive=True)]
    paths.sort(key=lambda p: [int(v) if v.isdigit() else v for v in re.split(r'(\d+)', p.name)])
    if len(paths) != len(config['slice_z_um']) or len({p.name for p in paths}) != len(paths):
        raise ValueError('Require one unambiguous input per configured slice depth.')
    if not np.all(np.diff(config['slice_z_um']) > 0):
        raise ValueError('Slice depths must increase strictly.')
    return paths


def validate(a, cfg, genes=None):
    if not a.n_obs or not a.obs_names.is_unique or not a.var_names.is_unique:
        raise ValueError('Require nonempty slices and unique cell/gene identifiers.')
    if genes is not None and a.var_names.tolist() != list(genes):
        raise ValueError('Gene order differs from the visible training vocabulary.')
    xy = np.asarray(a.obsm[cfg['spatial_key']])
    if xy.shape != (a.n_obs, 2) or not np.isfinite(xy).all():
        raise ValueError('Require finite N x 2 physical coordinates.')
    if cfg['label_key'] not in a.obs or a.obs[cfg['label_key']].isna().any():
        raise ValueError('Require observed cell-type annotations.')
    g = dense(a)
    if not np.isfinite(g).all() or (g < 0).any():
        raise ValueError('Expression X must be finite and nonnegative; no implicit normalization.')
    return g


def prepare(paths, config, heldout, excluded=()):
    hidden = {int(heldout), *map(int, excluded)}
    if heldout <= 0 or heldout >= len(paths)-1 or any(i < 0 or i >= len(paths) for i in hidden):
        raise ValueError('Hold out an interior slice; excluded indices must exist.')
    indices = [i for i in range(len(paths)) if i not in hidden]
    if len(indices) < 3 or not any(i < heldout for i in indices) or not any(i > heldout for i in indices):
        raise ValueError('Need >=3 visible slices and endpoints bracketing the hidden depth.')
    arrays, genes = [], None
    for i in indices:  # Hidden paths are never opened here.
        a = ad.read_h5ad(paths[i])
        g = validate(a, config['data'], genes)
        genes = a.var_names.tolist()
        arrays.append(dict(index=i, physical=np.asarray(a.obsm[config['data']['spatial_key']], float),
                           labels=a.obs[config['data']['label_key']].astype(str).to_numpy(),
                           ids=a.obs_names.to_numpy().astype(str), g=g, z_um=float(config['slice_z_um'][i])))
    classes, counts = np.unique(np.concatenate([s['labels'] for s in arrays]), return_counts=True)
    xy = np.vstack([s['physical'] for s in arrays]); g = np.vstack([s['g'] for s in arrays])
    width = float(config['spatial']['bin_um'])
    if width <= 0:
        raise ValueError('Spatial bin width must be positive.')
    origin = np.floor(xy.min(0)/width)*width-width
    shape = np.ceil((xy.max(0)-origin)/width).astype(int)+1
    if np.prod(shape) > config['spatial']['max_grid_cells']:
        raise ValueError('Spatial grid exceeds max_grid_cells; inspect coordinate units/bin size.')
    state = dict(genes=genes, classes=classes.tolist(), counts=counts.tolist(),
                 xy_min=xy.min(0).tolist(), xy_range=np.maximum(np.ptp(xy, axis=0), 1.).tolist(),
                 z_min=float(min(s['z_um'] for s in arrays)),
                 z_range=float(max(s['z_um'] for s in arrays)-min(s['z_um'] for s in arrays)),
                 gene_scale=np.maximum(g.std(0), .1).tolist(),
                 gene_mean=g.mean(0).tolist(), gene_positive=(g > 0).mean(0).tolist(),
                 gene_max_log=(np.log1p(g).max(0)+1.).tolist(),
                 grid_origin=origin.tolist(), grid_shape=shape.tolist(),
                 train_indices=indices, heldout=int(heldout), excluded=sorted(hidden-{heldout}))
    for s in arrays:
        s['y'] = pd.Categorical(s.pop('labels'), categories=classes).codes.astype(np.int64)
    return arrays, state


def endpoints(slices, z):
    """Closest other measured slice on each side; repeat the available side at edges."""
    lower = [s for s in slices if s['z_um'] < z]
    upper = [s for s in slices if s['z_um'] > z]
    if not lower and not upper:
        raise ValueError('No observed neighborhood remains after excluding target slice.')
    left = max(lower, key=lambda s: s['z_um']) if lower else min(upper, key=lambda s: s['z_um'])
    right = min(upper, key=lambda s: s['z_um']) if upper else left
    return left, right


def grid(state, config):
    ij = np.indices(state['grid_shape']).reshape(2, -1).T
    return np.asarray(state['grid_origin'])+(ij+.5)*config['spatial']['bin_um']


def histogram(s, state, config):
    ij = np.floor((s['physical']-state['grid_origin'])/config['spatial']['bin_um']).astype(int)
    flat = np.ravel_multi_index(ij.T, state['grid_shape'])
    return np.bincount(flat, minlength=int(np.prod(state['grid_shape']))).astype(np.float32)


def spatial_inputs(left, right, z, state, config):
    """Uses coordinates/counts ONLY: no labels, genes, or matching parents."""
    densities = []
    for s in (left, right):
        counts = histogram(s, state, config).reshape(state['grid_shape'])
        smooth = gaussian_filter(counts, config['spatial']['smooth_sigma_bins'], mode='constant')
        smooth *= len(s['physical'])/max(float(smooth.sum()), 1e-8)
        densities.append(smooth.ravel())
    gap = right['z_um']-left['z_um']
    t = (z-left['z_um'])/gap
    baseline = (1-t)*densities[0]+t*densities[1]+config['spatial']['intensity_floor']
    xy = (grid(state, config)-state['xy_min'])/state['xy_range']
    features = np.column_stack((xy, np.log1p(densities[0]), np.log1p(densities[1]),
                                np.full((len(xy), 1), t),
                                np.full((len(xy), 1), (z-state['z_min'])/state['z_range']),
                                np.full((len(xy), 1), gap/state['z_range'])))
    return features.astype(np.float32), baseline.astype(np.float32)


def neighborhood(xy, z, slices, state, config):
    """Return separate type-only features and measured-expression neighborhood features.

    Targets at depth z are excluded even if accidentally passed in slices.
    No source identity or source label is assigned to a generated cell.
    """
    left, right = endpoints(slices, z)
    xy = np.asarray(xy, float)
    cfg = config['neighborhood']; nc = len(state['classes']); ng = len(state['genes'])
    priors = np.asarray(state['counts'], float); priors /= priors.sum()
    type_parts, gene_parts, raw_parts, positive_parts, weights = [], [], [], [], []
    for s in (left, right):
        k = min(cfg['neighbors'], len(s['g']))
        dist, ix = cKDTree(s['physical']).query(xy, k=k)
        if k == 1:
            dist, ix = dist[:, None], ix[:, None]
        w = np.exp(-.5*(dist/cfg['sigma_xy_um'])**2)
        w *= dist <= cfg['radius_um']
        mass = w.sum(1, keepdims=True)
        normalized = w/np.maximum(mass, 1e-12)
        composition = (np.eye(nc, dtype=np.float32)[s['y'][ix]]*normalized[..., None]).sum(1)
        composition[mass[:, 0] == 0] = priors
        delta = ((s['physical'][ix]-xy[:, None, :])*normalized[..., None]).sum(1)/cfg['radius_um']
        type_parts.append(np.column_stack((composition, np.log1p(mass), delta,
                                           np.full(len(xy), (s['z_um']-z)/state['z_range']))))
        local_g = s['g'][ix]
        raw = (local_g*normalized[..., None]).sum(1)
        positive = ((local_g > 0)*normalized[..., None]).sum(1)
        gene_parts.append(np.column_stack((np.log1p(raw), positive)))
        raw_parts.append(raw); positive_parts.append(positive)
        depth_weight = 1/max(abs(s['z_um']-z), 1.)
        weights.append(depth_weight*(mass[:, 0] > 0))
    geometric = np.column_stack(((xy-state['xy_min'])/state['xy_range'],
                                  np.full(len(xy), (z-state['z_min'])/state['z_range'])))
    features = np.column_stack((geometric, *type_parts)).astype(np.float32)
    gene_features = np.column_stack(gene_parts).astype(np.float32)
    # Empty local neighborhoods receive a zero-expression baseline, never hidden data.
    w = np.column_stack(weights); w /= np.maximum(w.sum(1, keepdims=True), 1e-12)
    mean = sum(v*w[:, j:j+1] for j, v in enumerate(raw_parts))
    positive = sum(v*w[:, j:j+1] for j, v in enumerate(positive_parts))
    comp = sum(type_parts[j][:, :nc]*w[:, j:j+1] for j in range(2))
    empty = w.sum(1) == 0; comp[empty] = priors
    return dict(features=features, gene_features=gene_features, prior=comp.astype(np.float32),
                mean=mean.astype(np.float32), positive=positive.astype(np.float32),
                empty=empty, endpoint_indices=[left['index'], right['index']])
