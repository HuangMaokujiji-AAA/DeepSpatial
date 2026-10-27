"""V2 空间候选：显式局部运输路径，冻结 G zero 的表达、类型和来源调度。"""
import numpy as np
from scipy import sparse


def transport_endpoints(coupling, target_xy, mode, rng):
    """每个来源固定一条路径；无正质量支持返回 NaN，调用者保留来源。"""
    pi = sparse.csr_matrix(coupling)
    if mode not in ('ot_barycentric', 'ot_sampled'):
        raise ValueError('Use ot_barycentric or ot_sampled.')
    target_xy = np.asarray(target_xy, dtype=float)
    if (pi.shape[1] != len(target_xy) or target_xy.shape[1:] != (2,)
            or not np.isfinite(target_xy).all() or not np.isfinite(pi.data).all() or (pi.data < 0).any()):
        raise ValueError('Transport needs finite XY and nonnegative finite weights.')
    mass = np.asarray(pi.sum(1)).ravel()
    supported = mass > 0
    endpoints = np.full((pi.shape[0], 2), np.nan)
    targets = np.full(pi.shape[0], -1, dtype=np.int64)
    if mode == 'ot_barycentric':
        endpoints[supported] = (pi @ target_xy)[supported] / mass[supported, None]
    else:
        for row in np.flatnonzero(supported):
            start, end = pi.indptr[row:row + 2]
            weights = pi.data[start:end]
            targets[row] = rng.choice(pi.indices[start:end], p=weights / weights.sum())
        endpoints[supported] = target_xy[targets[supported]]
    return endpoints, supported, targets


def spatial_transport_volume(base, observed, couplings, mode, seed=42,
                             spatial_key='spatial', z_key='z_coord', label_key='cell_class'):
    """仅替换生成 XY；目标深度、表达/层、细胞类型、行顺序与总数保持原值。"""
    required = {'source_z', 'source_cell_index', 'source_cell_id', 'source_cell_class',
                'source_direction', z_key, 'is_observed'}
    if mode not in ('source_xy', 'ot_barycentric', 'ot_sampled'):
        raise ValueError('Use source_xy, ot_barycentric or ot_sampled.')
    if not required <= set(base.obs) or base.obs.is_observed.any():
        raise ValueError('V2 spatial expects generated-only G output with complete source metadata.')
    if len(observed) < 2 or len(couplings) != len(observed) - 1:
        raise ValueError('One coupling per adjacent visible-slice gap is required.')
    depths = np.array([float(a.obs[z_key].iloc[0]) for a in observed])
    if not np.isfinite(depths).all() or not (np.diff(depths) > 0).all():
        raise ValueError('Observed depths must be finite and strictly increasing.')
    if any(not a.var_names.equals(base.var_names) for a in observed):
        raise ValueError('Visible and generated gene order differs.')
    obs = base.obs
    xy = np.empty((base.n_obs, 2), dtype=float)
    assigned = np.zeros(base.n_obs, dtype=bool)
    matched = np.zeros(base.n_obs, dtype=bool)
    target_index = np.full(base.n_obs, -1, dtype=np.int64)
    source_xy = np.empty_like(xy)
    rng = np.random.default_rng(seed)
    for gap, (left, right, pi) in enumerate(zip(observed[:-1], observed[1:], couplings)):
        pi = sparse.csr_matrix(pi)
        if pi.shape != (left.n_obs, right.n_obs):
            raise ValueError('Coupling dimensions differ from visible slices.')
        edges = pi.tocoo()
        if (left.obs[label_key].astype(str).to_numpy()[edges.row] != right.obs[label_key].astype(str).to_numpy()[edges.col]).any():
            raise ValueError('V2 coupling must contain only same-type edges.')
        z0, z1 = depths[gap:gap + 2]
        for forward, source, target, plan in ((True, left, right, pi), (False, right, left, pi.T.tocsr())):
            direction = 'forward' if forward else 'backward'
            z_source = z0 if forward else z1
            ids = np.flatnonzero((obs.source_z.to_numpy() == z_source) & (obs.source_direction.to_numpy() == direction))
            if not len(ids):
                continue
            if assigned[ids].any():
                raise ValueError('A generated cell belongs to multiple intervals.')
            parents = obs.source_cell_index.to_numpy(dtype=np.int64)[ids]
            if (parents < 0).any() or (parents >= source.n_obs).any():
                raise ValueError('Source indices are out of range.')
            if (not np.array_equal(source.obs_names[parents].astype(str), obs.source_cell_id.iloc[ids].astype(str))
                    or not np.array_equal(source.obs[label_key].astype(str).to_numpy()[parents], obs.source_cell_class.iloc[ids].astype(str))):
                raise ValueError('Source cell identity/type differs from the saved schedule.')
            t = (obs[z_key].to_numpy()[ids] - z0) / (z1 - z0)
            if not np.isfinite(t).all() or (t < 0).any() or (t >= 1).any():
                raise ValueError('Target depth lies outside its visible interval.')
            if mode == 'source_xy':
                endpoint = np.asarray(source.obsm[spatial_key], float)
                valid = np.asarray(plan.sum(1)).ravel() > 0
                targets = np.full(source.n_obs, -1, dtype=np.int64)
            else:
                endpoint, valid, targets = transport_endpoints(plan, target.obsm[spatial_key], mode, rng)
            origins = np.asarray(source.obsm[spatial_key], float)[parents]
            weight = t if forward else 1 - t
            supported = valid[parents]
            positions = origins.copy()
            positions[supported] += weight[supported, None] * (endpoint[parents[supported]] - origins[supported])
            xy[ids], source_xy[ids] = positions, origins
            matched[ids], target_index[ids], assigned[ids] = supported, targets[parents], True
    if not assigned.all() or not np.isfinite(xy).all():
        raise ValueError('Every generated cell must have one finite spatial path.')
    result = base.copy()
    result.obsm['spatial_before_v2'] = np.asarray(base.obsm[spatial_key]).copy()
    result.obsm[spatial_key] = xy
    result.obsm['spatial_3d'] = np.column_stack([xy, result.obs[z_key].to_numpy()])
    result.obs['v1_generation_mode'] = result.obs['generation_mode'].astype(str)
    result.obs['generation_mode'] = np.where(matched, 'v2_' + mode, 'source_anchor')
    result.obs['match_status'] = np.where(matched, 'matched', 'unsupported')
    result.obs['v2_target_cell_index'] = target_index
    result.obs['v2_source_displacement_um'] = np.linalg.norm(xy - source_xy, axis=1)
    if 'local_matching' in result.uns:
        result.uns['v1_local_matching'] = result.uns.pop('local_matching')
    result.uns['v2_spatial'] = dict(mode=mode, seed=int(seed), expression_and_types_frozen=True,
                                  supported_cells=int(matched.sum()), source_anchor_cells=int((~matched).sum()),
                                  one_target_per_parent_per_gap=True,
                                  mean_displacement_um=float(np.linalg.norm(xy - source_xy, axis=1).mean()))
    return result
