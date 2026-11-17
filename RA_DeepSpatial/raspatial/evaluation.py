"""Unpaired 3D virtual-slice evaluation across five orthogonal dimensions:
1. Spatial localization & cardinality fidelity (OSPA-1, 1-to-1 P/R/F1, exact & sliced W2, Count MRE)
2. Marker spatial pattern & structural continuity (heldout_C vs loss 2D Grid PCC/Spearman, SSIM, |ΔMoran's I|, |ΔGeary's C|)
3. Single-cell transcriptomic identity purity (frozen train-only expression probe & Verified 1-to-1 Spatial F1)
4. Microenvironment & niche topology (unbiased class-aggregated Niche JSD, analytical permutation Z-score concordance, neighbor distance)
5. Global non-inferiority guardrails (train-selected Top-HVG 2D Grid PCC/SSIM/RMSE, common-type fidelity)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment, linprog
from scipy.spatial import cKDTree, distance
from scipy.sparse.csgraph import maximum_bipartite_matching, min_weight_full_bipartite_matching
from scipy.special import rel_entr
from scipy.stats import spearmanr
import torch
from torch import nn


def corr(a, b, rank=False):
    a, b = np.asarray(a, dtype=float).ravel(), np.asarray(b, dtype=float).ravel()
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if len(a) < 2 or a.std() <= 1e-12 or b.std() <= 1e-12:
        return np.nan
    return float(spearmanr(a, b).statistic if rank else np.corrcoef(a, b)[0, 1])


def js(a, b):
    a, b = np.asarray(a, float) + 1e-9, np.asarray(b, float) + 1e-9
    a, b = a / a.sum(-1, keepdims=True), b / b.sum(-1, keepdims=True)
    middle = 0.5 * (a + b)
    return 0.5 * (rel_entr(a, middle).sum(-1) + rel_entr(b, middle).sum(-1))


def kl_div(a, b):
    a, b = np.asarray(a, float) + 1e-6, np.asarray(b, float) + 1e-6
    a, b = a / a.sum(-1, keepdims=True), b / b.sum(-1, keepdims=True)
    return rel_entr(a, b).sum(-1)


def expression(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return dict(mse=float(np.mean((a - b) ** 2)), mae=float(np.mean(np.abs(a - b))), pcc=corr(a, b))


def assignment(a, b, cap):
    """Exact capped Hungarian/OSPA-1 distance; sparse dummy nodes punish unmatched cells."""
    if not len(a) or not len(b):
        return float(cap), np.empty(0, int), np.empty(0, int)
    swapped = len(a) > len(b)
    x, y = (b, a) if swapped else (a, b)
    n, m = len(x), len(y)
    edges = cKDTree(x).sparse_distance_matrix(cKDTree(y), cap, output_type='coo_matrix')
    # Each small-side cell has its own dummy; larger-side unmatched penalty is constant.
    row = np.concatenate((edges.row, np.arange(n)))
    col = np.concatenate((edges.col, m + np.arange(n)))
    costs = np.concatenate((edges.data, np.full(n, cap))) + 1e-9
    graph = sparse.csr_matrix((costs, (row, col)), shape=(n, m + n))
    ri, ci = min_weight_full_bipartite_matching(graph)
    total = float(graph[ri, ci].sum() - n * 1e-9 + (m - n) * cap) / m
    real = ci < m
    return total, (ci[real] if swapped else ri[real]), (ri[real] if swapped else ci[real])


def _emd2_exact_lp(sq_cost):
    """Exact balanced uniform-marginal OT objective via HiGHS LP (no external POT dependency)."""
    nt, npred = sq_cost.shape
    n_vars = nt * npred
    row_eq_r = np.repeat(np.arange(nt, dtype=np.int64), npred)
    row_eq_c = np.arange(n_vars, dtype=np.int64)
    col_eq_r = nt + np.tile(np.arange(npred - 1, dtype=np.int64), nt)
    col_eq_c = (np.arange(nt, dtype=np.int64)[:, None] * npred + np.arange(npred - 1, dtype=np.int64)[None, :]).ravel()
    rows = np.concatenate((row_eq_r, col_eq_r))
    cols = np.concatenate((row_eq_c, col_eq_c))
    a_eq = sparse.csc_matrix((np.ones(len(rows), dtype=np.float64), (rows, cols)), shape=(nt + npred - 1, n_vars))
    b_eq = np.concatenate((np.full(nt, 1.0 / nt), np.full(npred - 1, 1.0 / npred)))
    res = linprog(sq_cost.ravel(), A_eq=a_eq, b_eq=b_eq, bounds=(0.0, None), method='highs')
    if not res.success:
        raise RuntimeError(f'Exact W2 HiGHS solver failed: {res.message}')
    return float(max(res.fun, 0.0))


def w2_distance(a, b, max_entries=250000, use_pot=True):
    """Exact 2-Wasserstein distance (µm) between uniform empirical point clouds."""
    nt, npred = len(a), len(b)
    if not nt or not npred or nt * npred > max_entries:
        return np.nan
    sq_cost = np.asarray(distance.cdist(a, b, 'sqeuclidean'), dtype=np.float64)
    if nt == 1 or npred == 1:
        return float(np.sqrt(max(sq_cost.mean(), 0.0)))
    if nt == npred:
        ri, ci = linear_sum_assignment(sq_cost)
        return float(np.sqrt(max(sq_cost[ri, ci].mean(), 0.0)))
    if use_pot:
        try:
            import ot
            p = np.full(nt, 1.0 / nt, dtype=np.float64)
            q = np.full(npred, 1.0 / npred, dtype=np.float64)
            return float(np.sqrt(max(float(ot.emd2(p, q, sq_cost, numItermax=500000)), 0.0)))
        except ImportError:
            pass
    return float(np.sqrt(_emd2_exact_lp(sq_cost)))


def sliced_w2_distance(a, b, n_projections=64):
    """Deterministic Sliced 2-Wasserstein distance (µm) via exact 1D quantile integration."""
    nt, npred = len(a), len(b)
    if not nt or not npred:
        return np.nan
    thetas = np.linspace(0.0, np.pi, int(n_projections), endpoint=False, dtype=np.float64)
    directions = np.vstack((np.cos(thetas), np.sin(thetas)))
    pa = np.sort(np.asarray(a, dtype=np.float64) @ directions, axis=0)
    pb = np.sort(np.asarray(b, dtype=np.float64) @ directions, axis=0)
    if nt == npred:
        return float(np.sqrt(max(np.mean((pa - pb) ** 2), 0.0)))
    qa = np.arange(1, nt, dtype=np.float64) / nt
    qb = np.arange(1, npred, dtype=np.float64) / npred
    knots = np.unique(np.concatenate(([0.0], qa, qb, [1.0])))
    widths = np.diff(knots)
    mids = 0.5 * (knots[:-1] + knots[1:])
    ia = np.clip(np.floor(mids * nt).astype(np.int64), 0, nt - 1)
    ib = np.clip(np.floor(mids * npred).astype(np.int64), 0, npred - 1)
    w2_sq = (widths[:, None] * (pa[ia] - pb[ib]) ** 2).sum(axis=0)
    return float(np.sqrt(max(float(w2_sq.mean()), 0.0)))


def build_spatial_grid(tx, px, width, state=None):
    """Construct 2D lattice (H, W), active tissue mask Ω, and Queen 8-neighbor graph."""
    tx, px = np.asarray(tx, dtype=np.float64), np.asarray(px, dtype=np.float64)
    if state is not None and 'xy_min' in state and 'xy_range' in state:
        xy_min = np.asarray(state['xy_min'], dtype=np.float64)
        xy_max = xy_min + np.asarray(state['xy_range'], dtype=np.float64)
        lo = np.minimum(tx.min(0), xy_min) - 10.0 * width
        hi = np.maximum(tx.max(0), xy_max) + 10.0 * width
    else:
        span = np.maximum(np.ptp(tx, axis=0), width)
        lo = tx.min(0) - 2.0 * span - 10.0 * width
        hi = tx.max(0) + 2.0 * span + 10.0 * width
    px_safe = np.clip(px, lo, hi)
    raw_ij = np.floor(np.vstack((tx, px_safe)) / width).astype(np.int64)
    ij = raw_ij - raw_ij.min(0)
    h, w = int(ij[:, 0].max()) + 1, int(ij[:, 1].max()) + 1
    flat = ij[:, 0] * w + ij[:, 1]
    t_flat, p_flat = flat[:len(tx)], flat[len(tx):]
    occ = np.zeros(h * w, dtype=bool)
    occ[t_flat] = True
    occ[p_flat] = True
    active_idx = np.flatnonzero(occ)
    nb = len(active_idx)
    compact_map = np.full(h * w, -1, dtype=np.int64)
    compact_map[active_idx] = np.arange(nb, dtype=np.int64)
    tb, pb = compact_map[t_flat], compact_map[p_flat]
    grid_id = compact_map.reshape(h, w)
    rows_w, cols_w = [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            r0, r1 = max(0, -dx), h - max(0, dx)
            c0, c1 = max(0, -dy), w - max(0, dy)
            if r0 >= r1 or c0 >= c1:
                continue
            u = grid_id[r0:r1, c0:c1]
            v = grid_id[r0 + dx:r1 + dx, c0 + dy:c1 + dy]
            valid = (u >= 0) & (v >= 0)
            if valid.any():
                rows_w.append(u[valid].ravel())
                cols_w.append(v[valid].ravel())
    if rows_w:
        rr, cc = np.concatenate(rows_w), np.concatenate(cols_w)
        w_queen = sparse.csr_matrix((np.ones(len(rr), dtype=np.float64), (rr, cc)), shape=(nb, nb))
        w_queen = ((w_queen + w_queen.T) > 0).astype(np.float64).tocsr()
    else:
        w_queen = sparse.csr_matrix((nb, nb), dtype=np.float64)
    return dict(tb=tb, pb=pb, nb=nb, h=h, w=w, active_idx=active_idx, w_queen=w_queen)


def ssim_2d_batch(a_compact, b_compact, grid, sigma=1.0):
    """Gaussian-windowed 2D SSIM evaluated over active tissue mask Ω for M fields."""
    a_c = np.atleast_2d(np.asarray(a_compact, dtype=np.float64))
    b_c = np.atleast_2d(np.asarray(b_compact, dtype=np.float64))
    m, nb = a_c.shape
    if nb < 2:
        return np.full(m, np.nan)
    valid = a_c.std(axis=1) > 1e-12
    h, w, active_idx = grid['h'], grid['w'], grid['active_idx']
    a_img = np.zeros((m, h * w), dtype=np.float64)
    b_img = np.zeros((m, h * w), dtype=np.float64)
    a_img[:, active_idx] = a_c
    b_img[:, active_idx] = b_c
    a_img, b_img = a_img.reshape(m, h, w), b_img.reshape(m, h, w)
    span_a = a_c.max(axis=1) - a_c.min(axis=1)
    span_b = b_c.max(axis=1) - b_c.min(axis=1)
    data_range = np.maximum(np.maximum(span_a, span_b), 1e-6)
    c1 = (0.01 * data_range[:, None, None]) ** 2
    c2 = (0.03 * data_range[:, None, None]) ** 2
    sig = (0.0, float(sigma), float(sigma))
    mu_a = gaussian_filter(a_img, sigma=sig, mode='nearest')
    mu_b = gaussian_filter(b_img, sigma=sig, mode='nearest')
    var_a = np.maximum(gaussian_filter(a_img * a_img, sigma=sig, mode='nearest') - mu_a * mu_a, 0.0)
    var_b = np.maximum(gaussian_filter(b_img * b_img, sigma=sig, mode='nearest') - mu_b * mu_b, 0.0)
    cov_ab = gaussian_filter(a_img * b_img, sigma=sig, mode='nearest') - mu_a * mu_b
    num = (2.0 * mu_a * mu_b + c1) * (2.0 * cov_ab + c2)
    den = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2)
    ssim_map = num / np.maximum(den, 1e-18)
    scores = ssim_map.reshape(m, h * w)[:, active_idx].mean(axis=1)
    scores[~valid] = np.nan
    return scores


def spatial_autocorrelation_batch(y_compact, w_queen):
    """Vectorized Moran's I and Geary's C across M fields on active tissue mask Ω."""
    y = np.asarray(y_compact, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    nb, m = y.shape
    s0 = float(w_queen.sum())
    if nb < 3 or s0 <= 0.0:
        return np.full(m, np.nan), np.full(m, np.nan), np.zeros(m, dtype=bool)
    z = y - y.mean(axis=0, keepdims=True)
    denom = (z * z).sum(axis=0)
    valid = denom > 1e-12
    wz = w_queen @ z
    quad = (z * wz).sum(axis=0)
    deg = np.asarray(w_queen.sum(axis=1)).ravel()
    deg_z2 = deg @ (z * z)
    safe_denom = np.maximum(denom, 1e-18)
    # Constant fields have zero spatial autocorrelation (I=0, C=1).
    moran = np.where(valid, (nb / s0) * quad / safe_denom, 0.0)
    geary = np.where(valid, ((nb - 1.0) / s0) * (deg_z2 - quad) / safe_denom, 1.0)
    return moran, geary, valid


def compare_autocorrelation(a_compact, b_compact, w_queen):
    """Compare truth vs predicted Moran's I and Geary's C; penalize flat-zero collapse."""
    it, ct, vt = spatial_autocorrelation_batch(a_compact, w_queen)
    ip, cp, _ = spatial_autocorrelation_batch(b_compact, w_queen)
    it = np.where(vt, it, np.nan)
    ct = np.where(vt, ct, np.nan)
    ip_eval = np.where(vt, ip, np.nan)
    cp_eval = np.where(vt, cp, np.nan)
    return dict(moran_truth=it, moran_pred=ip_eval, moran_abs_error=np.abs(it - ip_eval),
                geary_truth=ct, geary_pred=cp_eval, geary_abs_error=np.abs(ct - cp_eval))


def composition(xy, labels, classes, radius):
    codes = np.array([classes.get(str(k), len(classes)) for k in labels])
    n = len(xy)
    tree = cKDTree(xy)
    edges = tree.sparse_distance_matrix(tree, radius, output_type='coo_matrix')
    keep = edges.row != edges.col
    graph = sparse.csr_matrix((np.ones(int(keep.sum())), (edges.row[keep], edges.col[keep])), shape=(n, n))
    onehot = sparse.csr_matrix((np.ones(n), (np.arange(n), codes)), shape=(n, len(classes) + 1))
    values = (graph @ onehot).toarray()
    mass = values.sum(1)
    return np.divide(values, mass[:, None], out=np.zeros_like(values), where=mass[:, None] > 0), mass


def nhood_enrichment_zscores(xy, labels, classes, radius):
    """Exact analytical permutation Z-score matrix (infinite-permutation Squidpy equivalent)."""
    cd = len(classes)
    codes = np.array([classes.get(str(k), -1) for k in labels], dtype=np.int64)
    valid_cells = codes >= 0
    xy, codes = np.asarray(xy, dtype=np.float64)[valid_cells], codes[valid_cells]
    n_total = len(xy)
    counts = np.bincount(codes, minlength=cd).astype(np.int64) if n_total else np.zeros(cd, dtype=np.int64)
    if n_total < 4 or cd == 0:
        return np.zeros((cd, cd), dtype=np.float64), counts
    tree = cKDTree(xy)
    edges = tree.sparse_distance_matrix(tree, radius, output_type='coo_matrix')
    keep = edges.row != edges.col
    adj = sparse.csr_matrix((np.ones(int(keep.sum()), dtype=np.float64), (edges.row[keep], edges.col[keep])), shape=(n_total, n_total))
    adj = ((adj + adj.T) > 0).astype(np.float64).tocsr()
    deg = np.asarray(adj.sum(axis=1)).ravel()
    s1 = float(deg.sum())
    if s1 <= 0.0:
        return np.zeros((cd, cd), dtype=np.float64), counts
    s2 = float((deg * (deg - 1.0)).sum())
    s_disjoint = max(s1 * s1 - 2.0 * s1 - 4.0 * s2, 0.0)
    onehot = sparse.csr_matrix((np.ones(n_total, dtype=np.float64), (np.arange(n_total), codes)), shape=(n_total, cd))
    obs = (onehot.T @ adj @ onehot).toarray()
    nf = counts.astype(np.float64)
    denom2 = n_total * (n_total - 1.0)
    denom3 = denom2 * (n_total - 2.0)
    denom4 = denom3 * (n_total - 3.0)
    p1 = (nf[:, None] * nf[None, :]) / denom2
    mu = s1 * p1
    p2_one = (nf[:, None] * nf[None, :] * np.maximum(nf[:, None] + nf[None, :] - 2.0, 0.0)) / denom3
    p2_zero = (nf[:, None] * np.maximum(nf[:, None] - 1.0, 0.0) * nf[None, :] * np.maximum(nf[None, :] - 1.0, 0.0)) / denom4
    ex2 = s1 * p1 + s2 * p2_one + s_disjoint * p2_zero
    p1_d = (nf * np.maximum(nf - 1.0, 0.0)) / denom2
    p2_one_d = (nf * np.maximum(nf - 1.0, 0.0) * np.maximum(nf - 2.0, 0.0)) / denom3
    p2_zero_d = (nf * np.maximum(nf - 1.0, 0.0) * np.maximum(nf - 2.0, 0.0) * np.maximum(nf - 3.0, 0.0)) / denom4
    idx = np.arange(cd)
    mu[idx, idx] = s1 * p1_d
    ex2[idx, idx] = 2.0 * s1 * p1_d + 4.0 * s2 * p2_one_d + s_disjoint * p2_zero_d
    var = np.maximum(ex2 - mu * mu, 0.0)
    z = np.where(var > 1e-12, (obs - mu) / np.sqrt(np.maximum(var, 1e-18)), 0.0)
    absent = counts == 0
    z[absent, :] = 0.0
    z[:, absent] = 0.0
    return z, counts


class ExpressionProbe(nn.Module):
    """Expression-only reference subclass classifier fitted strictly on training sections."""
    def __init__(self, n_genes, n_classes, log_mean, log_scale, hidden_size=256):
        super().__init__()
        self.register_buffer('log_mean', torch.as_tensor(log_mean, dtype=torch.float32))
        self.register_buffer('log_scale', torch.as_tensor(log_scale, dtype=torch.float32).clamp_min(1e-6))
        self.net = nn.Sequential(
            nn.Linear(n_genes, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, g_raw):
        lg = torch.log1p(torch.clamp_min(g_raw, 0.0))
        return self.net((lg - self.log_mean) / self.log_scale)


def fit_expression_probe(store, state, config, device, save_path=None):
    """Train frozen expression-only probe streaming over cached train sections; never touches val/test."""
    eval_cfg = config.get('evaluation', {})
    gd, cd = len(state['genes']), len(state['classes'])
    hidden = int(eval_cfg.get('probe_hidden_size', min(256, max(32, gd * 2))))
    epochs = int(eval_cfg.get('probe_epochs', 8))
    batch_size = int(eval_cfg.get('probe_batch_size', 512))
    lr = float(eval_cfg.get('probe_lr', 1e-3))
    seed = int(config.get('seed', 42)) + 911
    rng = np.random.default_rng(seed)
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        probe = ExpressionProbe(gd, cd, state['log_mean'], state['log_scale'], hidden_size=hidden).to(device)
        counts = np.asarray(state['counts'], dtype=np.float64)
        inv = np.where(counts > 0, (counts.sum() / np.maximum(counts, 1.0)) ** 0.5, 0.0)
        present = counts > 0
        if present.any():
            inv = np.minimum(inv / max(float(inv[present].mean()), 1e-8), 10.0)
        weights = torch.as_tensor(inv, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
        train_sections = [row['section'] for row in state['train']]
        probe.train()
        last_loss = 0.0
        for _ in range(epochs):
            order = rng.permutation(len(train_sections))
            losses = []
            for sec_idx in order:
                sec = store[train_sections[int(sec_idx)]]
                n_sec = len(sec['y'])
                perm = rng.permutation(n_sec)
                for start in range(0, n_sec, batch_size):
                    idx = perm[start:start + batch_size]
                    gx = torch.as_tensor(np.asarray(sec['g'][idx], dtype=np.float32), device=device)
                    gy = torch.as_tensor(np.asarray(sec['y'][idx], dtype=np.int64), device=device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(probe(gx), gy)
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach()))
            last_loss = float(np.mean(losses)) if losses else 0.0
        probe.eval()
        for param in probe.parameters():
            param.requires_grad_(False)
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = save_path.with_name(save_path.name + '.tmp')
            torch.save(dict(state_dict=probe.cpu().state_dict(), n_genes=gd, n_classes=cd,
                            hidden_size=hidden, train_loss=last_loss, seed=seed), tmp)
            tmp.replace(save_path)
            probe = probe.to(device).eval()
        return probe, dict(hidden_size=hidden, epochs=epochs, batch_size=batch_size, lr=lr, final_train_loss=last_loss)
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def load_expression_probe(path, state, device='cpu'):
    saved = torch.load(Path(path), map_location='cpu', weights_only=False)
    probe = ExpressionProbe(saved['n_genes'], saved['n_classes'], state['log_mean'], state['log_scale'],
                            hidden_size=saved['hidden_size'])
    probe.load_state_dict(saved['state_dict'])
    probe = probe.to(device).eval()
    for param in probe.parameters():
        param.requires_grad_(False)
    return probe


@torch.no_grad()
def predict_expression_classes(g_raw, state, probe=None, batch_size=4096):
    """Decode top-1 and top-3 subclass indices purely from expression; train-centroid fallback if probe is None."""
    n = len(g_raw)
    cd = len(state['classes'])
    k_top = min(3, cd)
    if n == 0:
        return np.empty(0, dtype=np.int64), np.empty((0, k_top), dtype=np.int64)
    if probe is not None:
        device = next(probe.parameters()).device
        probe.eval()
        top1 = np.empty(n, dtype=np.int64)
        topk = np.empty((n, k_top), dtype=np.int64)
        for start in range(0, n, batch_size):
            batch = torch.as_tensor(np.asarray(g_raw[start:start + batch_size], dtype=np.float32), device=device)
            logits = probe(batch)
            vals = torch.topk(logits, k=k_top, dim=-1).indices.cpu().numpy()
            top1[start:start + len(batch)] = vals[:, 0]
            topk[start:start + len(batch)] = vals
        return top1, topk
    if 'type_log_mean' not in state or 'log_mean' not in state or 'log_scale' not in state:
        return np.full(n, -1, dtype=np.int64), np.full((n, k_top), -1, dtype=np.int64)
    log_mean = np.asarray(state['log_mean'], dtype=np.float32)
    log_scale = np.maximum(np.asarray(state['log_scale'], dtype=np.float32), 1e-6)
    centroids = (np.asarray(state['type_log_mean'], dtype=np.float32) - log_mean) / log_scale
    norms = np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-8)
    centroids = centroids / norms
    absent = np.asarray(state['counts']) <= 0
    top1 = np.empty(n, dtype=np.int64)
    topk = np.empty((n, k_top), dtype=np.int64)
    for start in range(0, n, batch_size):
        chunk = np.log1p(np.maximum(np.asarray(g_raw[start:start + batch_size], dtype=np.float32), 0.0))
        z = (chunk - log_mean) / log_scale
        z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)
        scores = z @ centroids.T
        if absent.any():
            scores[:, absent] = -np.inf
        order = np.argsort(-scores, axis=1, kind='stable')[:, :k_top]
        top1[start:start + len(chunk)] = order[:, 0]
        topk[start:start + len(chunk)] = order
    return top1, topk


def _bipartite_match_count(a, b, radius):
    if not len(a) or not len(b):
        return 0
    edges = cKDTree(a).sparse_distance_matrix(cKDTree(b), radius, output_type='coo_matrix')
    if not len(edges.data):
        return 0
    graph = sparse.csr_matrix((np.ones(len(edges.data)), (edges.row, edges.col)), shape=(len(a), len(b)))
    return int((maximum_bipartite_matching(graph, perm_type='column') >= 0).sum())


def evaluate(truth, prediction, state, config, markers, independent, probe=None):
    tx, tg, ty = truth['physical'], truth['g'], truth['labels']
    px = np.asarray(prediction.obsm['spatial'], dtype=np.float64)
    pg = prediction.X.toarray() if sparse.issparse(prediction.X) else np.asarray(prediction.X)
    py = prediction.obs.cell_class.astype(str).to_numpy()
    if list(truth['genes']) != state['genes'] or prediction.var_names.tolist() != state['genes']:
        raise ValueError('Frozen training gene order differs at evaluation.')
    if not len(px) or not np.isfinite(pg).all() or not np.isfinite(px).all():
        raise ValueError('Empty or nonfinite prediction.')

    eval_cfg = config['evaluation']
    ssim_sigma = float(eval_cfg.get('ssim_sigma', 1.0))
    max_w2_entries = int(eval_cfg.get('max_w2_entries', 250000))
    n_projections = int(eval_cfg.get('sliced_w2_projections', 64))
    z_thresh = float(eval_cfg.get('nhood_z_threshold', 2.0))

    tl, pl = np.log1p(np.maximum(tg, 0)), np.log1p(np.maximum(pg, 0))
    has_pre = 'expression_before_sparsification' in prediction.layers
    if has_pre:
        raw_pre = prediction.layers['expression_before_sparsification']
        pg_pre = raw_pre.toarray() if sparse.issparse(raw_pre) else np.asarray(raw_pre)
    else:
        pg_pre = None

    gen_mode = prediction.obs['generation_mode'] if 'generation_mode' in prediction.obs else pd.Series('external', index=prediction.obs.index)
    result = dict(
        truth_n=len(tx),
        generated_n=len(px),
        total_count_relative_error=abs(len(px) - len(tx)) / len(tx),
        negative_expression_fraction=float((pg < 0).mean()),
        source_anchor_fraction=float((gen_mode == 'source_anchor').mean()),
        expression_probe_mode='trained_mlp' if probe is not None else 'train_centroid_fallback',
    )
    for scale, t, p in (('provided_X', tg, pg), ('log1p', tl, pl)):
        result.update({f'global_{scale}_{key}': value for key, value in expression(t.mean(0), p.mean(0)).items()})

    classes = {name: i for i, name in enumerate(state['classes'])}
    truth_codes = np.array([classes.get(str(k), -1) for k in ty], dtype=np.int64)
    pred_codes = np.array([classes.get(str(k), -1) for k in py], dtype=np.int64)

    # Dimension 3: Expression-only decoding (never observes coordinates or labels)
    t_probe_top1, _ = predict_expression_classes(tg, state, probe=probe)
    p_probe_top1, p_probe_topk = predict_expression_classes(pg, state, probe=probe)
    p_pre_top1, _ = predict_expression_classes(pg_pre, state, probe=probe) if pg_pre is not None else (None, None)

    known_truth = truth_codes >= 0
    known_pred = pred_codes >= 0
    result['global_truth_probe_accuracy'] = float((t_probe_top1[known_truth] == truth_codes[known_truth]).mean()) if known_truth.any() else np.nan
    result['global_generated_probe_concordance'] = float((p_probe_top1[known_pred] == pred_codes[known_pred]).mean()) if known_pred.any() else np.nan
    result['global_generated_probe_concordance_pre_sparsify'] = (
        float((p_pre_top1[known_pred] == pred_codes[known_pred]).mean()) if (p_pre_top1 is not None and known_pred.any()) else np.nan
    )

    # Dimension 4: Local niche composition & analytical permutation neighborhood Z-scores
    niche_radius = float(eval_cfg['niche_radius_um'])
    tniche, tmass = composition(tx, ty, classes, niche_radius)
    pniche, pmass = composition(px, py, classes, niche_radius)
    z_true, z_counts_t = nhood_enrichment_zscores(tx, ty, classes, niche_radius)
    z_pred, z_counts_p = nhood_enrichment_zscores(px, py, classes, niche_radius)

    # Dimension 2 & 5: 2D Spatial Grid, Tissue Mask Ω, and Queen Contiguity Graph
    width = float(eval_cfg['spatial_bin_um'])
    grid = build_spatial_grid(tx, px, width, state=state)
    tb, pb, nb, w_queen = grid['tb'], grid['pb'], grid['nb'], grid['w_queen']

    cell_count_t = np.bincount(tb, minlength=nb).astype(np.float64)
    cell_count_p = np.bincount(pb, minlength=nb).astype(np.float64)
    result['global_cell_density_grid_pcc'] = corr(cell_count_t, cell_count_p)
    result['global_cell_density_grid_ssim'] = float(ssim_2d_batch(cell_count_t, cell_count_p, grid, sigma=ssim_sigma)[0])

    # Spatially explicit global non-inferiority on train-selected Top-HVGs
    hvg_k = min(int(eval_cfg.get('top_hvg_n', 250)), len(state['genes']))
    log_scale_arr = np.asarray(state.get('log_scale', np.ones(len(state['genes']))), dtype=np.float64)
    hvg_cols = np.argsort(-log_scale_arr, kind='stable')[:hvg_k]
    t_bin_mat = sparse.csr_matrix((np.ones(len(tb), dtype=np.float64), (tb, np.arange(len(tb)))), shape=(nb, len(tb)))
    p_bin_mat = sparse.csr_matrix((np.ones(len(pb), dtype=np.float64), (pb, np.arange(len(pb)))), shape=(nb, len(pb)))
    hvg_grid_t = (t_bin_mat @ tl[:, hvg_cols]) / np.maximum(cell_count_t[:, None], 1.0)
    hvg_grid_p = (p_bin_mat @ pl[:, hvg_cols]) / np.maximum(cell_count_p[:, None], 1.0)
    hvg_pccs = np.array([corr(hvg_grid_t[:, m], hvg_grid_p[:, m]) for m in range(hvg_k)], dtype=np.float64)
    hvg_spms = np.array([corr(hvg_grid_t[:, m], hvg_grid_p[:, m], True) for m in range(hvg_k)], dtype=np.float64)
    hvg_ssims = ssim_2d_batch(hvg_grid_t.T, hvg_grid_p.T, grid, sigma=ssim_sigma)
    hvg_auto = compare_autocorrelation(hvg_grid_t, hvg_grid_p, w_queen)
    result.update(
        global_top_hvg_n=int(hvg_k),
        global_top_hvg_grid_pcc=float(np.nanmean(hvg_pccs)) if np.isfinite(hvg_pccs).any() else np.nan,
        global_top_hvg_grid_spearman=float(np.nanmean(hvg_spms)) if np.isfinite(hvg_spms).any() else np.nan,
        global_top_hvg_grid_ssim=float(np.nanmean(hvg_ssims)) if np.isfinite(hvg_ssims).any() else np.nan,
        global_top_hvg_grid_rmse=float(np.sqrt(np.mean((hvg_grid_t - hvg_grid_p) ** 2))),
        global_top_hvg_morans_i_abs_error=float(np.nanmean(hvg_auto['moran_abs_error'])) if np.isfinite(hvg_auto['moran_abs_error']).any() else np.nan,
    )

    # Precompute slice-wide (label-free) 2D spatial fields for all unique evaluated marker genes
    unique_marker_genes = sorted({j for cols in (*markers.values(), *independent.values()) for j in cols})
    if unique_marker_genes:
        slice_mk_t = t_bin_mat @ tl[:, unique_marker_genes]
        slice_mk_p = p_bin_mat @ pl[:, unique_marker_genes]
        slice_mk_ssim = ssim_2d_batch(slice_mk_t.T, slice_mk_p.T, grid, sigma=ssim_sigma)
        slice_mk_auto = compare_autocorrelation(slice_mk_t, slice_mk_p, w_queen)
        mk_lookup = {gene_col: idx for idx, gene_col in enumerate(unique_marker_genes)}
    else:
        slice_mk_t = slice_mk_p = slice_mk_ssim = slice_mk_auto = None
        mk_lookup = {}

    train_counts = dict(zip(state['classes'], state['counts']))
    total_train = sum(train_counts.values())
    present_partners = z_counts_t > 0

    # Slice-level rare interaction matrix concordance (upper triangle where >=1 endpoint is rare and both exist in truth)
    cd = len(state['classes'])
    if cd > 0 and present_partners.any():
        rare_mask_cls = np.array([0 < train_counts.get(k, 0) / total_train <= eval_cfg['rare_fraction'] for k in state['classes']], dtype=bool)
        iu, ju = np.triu_indices(cd, k=0)
        keep_pairs = present_partners[iu] & present_partners[ju] & (rare_mask_cls[iu] | rare_mask_cls[ju])
        if keep_pairs.any():
            zt_vec, zp_vec = z_true[iu[keep_pairs], ju[keep_pairs]], z_pred[iu[keep_pairs], ju[keep_pairs]]
            result['rare_nhood_zscore_matrix_pcc'] = corr(zt_vec, zp_vec)
            result['rare_nhood_zscore_matrix_spearman'] = corr(zt_vec, zp_vec, True)
            sig_true = np.abs(zt_vec) >= z_thresh
            sig_pred = np.abs(zp_vec) >= z_thresh
            tp = int((sig_true & sig_pred & (np.sign(zt_vec) == np.sign(zp_vec))).sum())
            denom_sig = int(sig_true.sum() + sig_pred.sum())
            result['rare_nhood_significant_sign_f1'] = float(2 * tp / denom_sig) if denom_sig > 0 else np.nan
        else:
            result['rare_nhood_zscore_matrix_pcc'] = np.nan
            result['rare_nhood_zscore_matrix_spearman'] = np.nan
            result['rare_nhood_significant_sign_f1'] = np.nan
    else:
        result['rare_nhood_zscore_matrix_pcc'] = np.nan
        result['rare_nhood_zscore_matrix_spearman'] = np.nan
        result['rare_nhood_significant_sign_f1'] = np.nan

    rows, marker_rows = [], []
    for name in sorted(set(state['classes']) | set(ty) | set(py)):
        ti, pi = np.flatnonzero(ty == name), np.flatnonzero(py == name)
        nt, npred = len(ti), len(pi)
        if not nt and not npred:
            continue
        a, b = tx[ti], px[pi]
        code = classes.get(name)
        fraction = train_counts.get(name, 0) / total_train
        rel_err = abs(npred - nt) / nt if nt else np.nan
        row = dict(
            subclass=name,
            train_n=train_counts.get(name, 0),
            train_fraction=fraction,
            is_rare=0 < fraction <= eval_cfg['rare_fraction'],
            is_ultra_rare=0 < fraction <= 0.001,
            unseen=not train_counts.get(name, 0),
            truth_n=nt,
            generated_n=npred,
            count_relative_error=rel_err,
            count_relative_error_capped=min(rel_err, 2.0) if nt else np.nan,
            false_presence=nt == 0 and npred > 0,
            missing=nt > 0 and npred == 0,
            missing_rate=float(npred == 0) if nt > 0 else np.nan,
        )

        # Probe transcriptomic identity checks for this subclass
        if code is not None:
            row['truth_probe_concordance'] = float((t_probe_top1[ti] == code).mean()) if nt else np.nan
            verified_local = p_probe_top1[pi] == code if npred else np.zeros(0, dtype=bool)
            row['probe_concordance'] = float(verified_local.mean()) if npred else (0.0 if nt else np.nan)
            row['probe_top3_concordance'] = float((p_probe_topk[pi] == code).any(axis=1).mean()) if npred else (0.0 if nt else np.nan)
            row['probe_concordance_pre_sparsify'] = (
                float((p_pre_top1[pi] == code).mean()) if (p_pre_top1 is not None and npred)
                else (0.0 if (p_pre_top1 is not None and nt) else np.nan)
            )
            b_verified = b[verified_local]
        else:
            row['truth_probe_concordance'] = np.nan
            row['probe_concordance'] = np.nan
            row['probe_top3_concordance'] = np.nan
            row['probe_concordance_pre_sparsify'] = np.nan
            b_verified = np.empty((0, 2), dtype=np.float64)

        t2p = cKDTree(b).query(a)[0] if nt and npred else np.full(nt, np.inf)
        p2t = cKDTree(a).query(b)[0] if nt and npred else np.full(npred, np.inf)
        for radius in eval_cfg['radii_um']:
            key = f'{radius:g}um'
            row[f'coverage_recall_{key}'] = float((t2p <= radius).mean()) if nt else np.nan
            row[f'coverage_precision_{key}'] = float((p2t <= radius).mean()) if npred else (0.0 if nt else np.nan)
            matched = _bipartite_match_count(a, b, radius)
            row[f'one_to_one_precision_{key}'] = matched / npred if npred else (0.0 if nt else np.nan)
            row[f'one_to_one_recall_{key}'] = matched / nt if nt else np.nan
            row[f'one_to_one_f1_{key}'] = 2 * matched / (nt + npred) if nt + npred else np.nan
            if code is not None:
                v_matched = _bipartite_match_count(a, b_verified, radius)
                row[f'verified_one_to_one_precision_{key}'] = v_matched / npred if npred else (0.0 if nt else np.nan)
                row[f'verified_one_to_one_recall_{key}'] = v_matched / nt if nt else np.nan
                row[f'verified_one_to_one_f1_{key}'] = 2 * v_matched / (nt + npred) if nt + npred else np.nan
            else:
                row[f'verified_one_to_one_precision_{key}'] = np.nan
                row[f'verified_one_to_one_recall_{key}'] = np.nan
                row[f'verified_one_to_one_f1_{key}'] = np.nan

        cap = float(eval_cfg['matching_cap_um'])
        capped, im, jm = assignment(a, b, cap)
        row['hungarian_capped_mean_um'] = capped
        row['ospa1_um'] = capped
        row['matching_cap_um'] = cap
        row['matched_pairs_for_niche'] = len(im)
        row['w2_distance_um'] = w2_distance(a, b, max_entries=max_w2_entries)
        row['sliced_w2_distance_um'] = sliced_w2_distance(a, b, n_projections=n_projections)

        if nt and npred and nt * npred <= eval_cfg['max_assignment_entries']:
            costs = distance.cdist(a, b)
            ri, ci = linear_sum_assignment(costs)
            ds = costs[ri, ci]
            row.update(hungarian_matched_mean_um=float(ds.mean()), hungarian_matched_median_um=float(np.median(ds)))
        else:
            row.update(hungarian_matched_mean_um=np.nan, hungarian_matched_median_um=np.nan)

        # Unbiased class-level aggregated neighborhood composition JSD/KL (evaluates ALL cells of this type)
        ti_valid = ti[tmass[ti] > 0]
        pi_valid = pi[pmass[pi] > 0]
        if len(ti_valid) and len(pi_valid):
            mean_tniche = tniche[ti_valid].mean(0)
            mean_pniche = pniche[pi_valid].mean(0)
            u_js = float(js(mean_tniche, mean_pniche))
            row['unbiased_niche_js'] = u_js
            row['unbiased_niche_js_penalized'] = u_js
            row['unbiased_niche_kl'] = float(kl_div(mean_tniche, mean_pniche))
        elif len(ti_valid) and not len(pi_valid):
            row['unbiased_niche_js'] = np.nan
            row['unbiased_niche_js_penalized'] = float(np.log(2.0))
            row['unbiased_niche_kl'] = np.nan
        else:
            row['unbiased_niche_js'] = np.nan
            row['unbiased_niche_js_penalized'] = np.nan
            row['unbiased_niche_kl'] = np.nan

        # Matched-pair local composition (retained for diagnostic comparison)
        valid_niche = (tmass[ti[im]] > 0) & (pmass[pi[jm]] > 0)
        row['niche_evaluable_pairs'] = int(valid_niche.sum())
        row['niche_missing_neighbor_pairs'] = int((~valid_niche).sum())
        if valid_niche.any():
            ni, nj = ti[im[valid_niche]], pi[jm[valid_niche]]
            row['local_composition_js'] = float(js(tniche[ni], pniche[nj]).mean())
            row['local_composition_kl'] = float(kl_div(tniche[ni], pniche[nj]).mean())
        else:
            row['local_composition_js'], row['local_composition_kl'] = np.nan, np.nan
        row['niche_neighbor_mass_error'] = abs(float(tmass[ti].mean()) - float(pmass[pi].mean())) if nt and npred else np.nan

        # Per-type permutation neighborhood enrichment profile concordance
        if code is not None and z_counts_t[code] > 0 and z_counts_p[code] > 0 and int(present_partners.sum()) >= 2:
            row['nhood_zscore_pcc'] = corr(z_true[code, present_partners], z_pred[code, present_partners])
            row['nhood_zscore_spearman'] = corr(z_true[code, present_partners], z_pred[code, present_partners], True)
        else:
            row['nhood_zscore_pcc'] = np.nan
            row['nhood_zscore_spearman'] = np.nan

        if code is not None and state['preferred_neighbor_code'][code] >= 0:
            pref = state['classes'][state['preferred_neighbor_code'][code]]
            row['preferred_neighbor_subclass'] = pref

            def preference(xy, labels, index):
                if not len(index):
                    return np.nan
                pool = np.flatnonzero(labels == pref)
                if not len(pool) or (pref == name and len(pool) < 2):
                    return np.nan
                ds_nn = cKDTree(xy[pool]).query(xy[index], k=2 if pref == name else 1)[0]
                return float(ds_nn[:, 1].mean() if pref == name else ds_nn.mean())

            dt, dp = preference(tx, ty, ti), preference(px, py, pi)
            row.update(
                truth_preferred_neighbor_distance_um=dt,
                generated_preferred_neighbor_distance_um=dp,
                niche_preference_distance_error_um=abs(dt - dp),
            )

        d0 = np.bincount(tb[ti], minlength=nb).astype(np.float64)
        d1 = np.bincount(pb[pi], minlength=nb).astype(np.float64)
        row['density_pcc'] = corr(d0, d1)
        row['density_tv'] = float(np.abs(d0 / max(nt, 1) - d1 / max(npred, 1)).sum() / 2)
        row['density_ssim'] = float(ssim_2d_batch(d0, d1, grid, sigma=ssim_sigma)[0])
        dens_auto = compare_autocorrelation(d0, d1, w_queen)
        row['density_morans_i_abs_error'] = float(dens_auto['moran_abs_error'][0])

        if nt:
            true_mean = tl[ti].mean(0)
            pred_mean = pl[pi].mean(0) if npred else np.zeros(pl.shape[1])
            row.update({f'type_log_expression_{k}': v for k, v in expression(true_mean, pred_mean).items()})
            row['type_log_expression_cosine'] = float(
                true_mean @ pred_mean / max(np.linalg.norm(true_mean) * np.linalg.norm(pred_mean), 1e-12)
            )

        # Marker evaluation (strictly separated into 'loss' vs 'heldout_C')
        if code is not None:
            mk_specs = [('loss', markers.get(code, [])), ('heldout_C', independent.get(code, []))]
            all_cls_cols = [j for _, cols in mk_specs for j in cols]
            if all_cls_cols:
                fields_t = np.column_stack([np.bincount(tb[ti], weights=tl[ti, j], minlength=nb) for j in all_cls_cols])
                fields_p = np.column_stack([np.bincount(pb[pi], weights=pl[pi, j], minlength=nb) for j in all_cls_cols])
                cls_ssims = ssim_2d_batch(fields_t.T, fields_p.T, grid, sigma=ssim_sigma)
                cls_auto = compare_autocorrelation(fields_t, fields_p, w_queen)
                offset = 0
                other_mask_t = ty != name
                other_mask_p = py != name
                for group, columns in mk_specs:
                    for j in columns:
                        field_t = fields_t[:, offset]
                        field_p = fields_p[:, offset]
                        mean_t = float(tl[ti, j].mean()) if nt else np.nan
                        mean_p = float(pl[pi, j].mean()) if npred else 0.0
                        other_t_mean = float(tl[other_mask_t, j].mean()) if other_mask_t.any() else np.nan
                        other_p_mean = float(pl[other_mask_p, j].mean()) if other_mask_p.any() else 0.0
                        enrichment_t = mean_t / (other_t_mean + 1e-6) if np.isfinite(other_t_mean) else np.nan
                        enrichment_p = mean_p / (other_p_mean + 1e-6) if other_mask_p.any() else np.nan
                        lfc_t = mean_t - other_t_mean if np.isfinite(other_t_mean) else np.nan
                        lfc_p = mean_p - other_p_mean if other_mask_p.any() else np.nan
                        s_idx = mk_lookup[j]
                        marker_rows.append(dict(
                            subclass=name,
                            gene=state['genes'][j],
                            gene_symbol=state['symbols'][j],
                            marker_group=group,
                            spatial_pcc=corr(field_t, field_p),
                            spatial_spearman=corr(field_t, field_p, True),
                            spatial_ssim=float(cls_ssims[offset]),
                            morans_i_truth=float(cls_auto['moran_truth'][offset]),
                            morans_i_generated=float(cls_auto['moran_pred'][offset]),
                            morans_i_abs_error=float(cls_auto['moran_abs_error'][offset]),
                            gearys_c_abs_error=float(cls_auto['geary_abs_error'][offset]),
                            slice_spatial_pcc=corr(slice_mk_t[:, s_idx], slice_mk_p[:, s_idx]),
                            slice_spatial_spearman=corr(slice_mk_t[:, s_idx], slice_mk_p[:, s_idx], True),
                            slice_spatial_ssim=float(slice_mk_ssim[s_idx]),
                            slice_morans_i_abs_error=float(slice_mk_auto['moran_abs_error'][s_idx]),
                            truth_mean_log=mean_t,
                            generated_mean_log=mean_p,
                            truth_enrichment=enrichment_t,
                            generated_enrichment=enrichment_p,
                            enrichment_absolute_error=abs(enrichment_t - enrichment_p),
                            log_fold_enrichment_abs_error=abs(lfc_t - lfc_p),
                        ))
                        offset += 1
        rows.append(row)

    table, markers_table = pd.DataFrame(rows), pd.DataFrame(marker_rows)
    marker_metric_cols = [
        'spatial_pcc', 'spatial_spearman', 'spatial_ssim',
        'morans_i_abs_error', 'gearys_c_abs_error',
        'slice_spatial_pcc', 'slice_spatial_spearman', 'slice_spatial_ssim', 'slice_morans_i_abs_error',
        'enrichment_absolute_error', 'log_fold_enrichment_abs_error',
    ]
    strat_groups = (
        ('all', table.truth_n > 0, pd.Series(True, index=table.index)),
        ('rare', table.is_rare & (table.truth_n > 0), table.is_rare),
        ('ultra_rare', table.is_ultra_rare & (table.truth_n > 0), table.is_ultra_rare),
        ('common', ~table.is_rare & ~table.unseen & (table.truth_n > 0), ~table.is_rare & ~table.unseen),
    )
    for group, mask, fp_mask in strat_groups:
        part = table[mask]
        result[f'{group}_types'] = len(part)
        result[f'{group}_missing_types'] = int(part.missing.sum()) if len(part) else 0
        result[f'{group}_false_present_types'] = int(table.loc[fp_mask, 'false_presence'].sum())
        for column in table.select_dtypes(include=np.number):
            if column not in ('train_n', 'train_fraction', 'truth_n', 'generated_n', 'matching_cap_um'):
                result[f'{group}_macro_{column}'] = float(part[column].mean()) if len(part) else np.nan
        for marker_group in ('loss', 'heldout_C'):
            subset = (
                markers_table[(markers_table.subclass.isin(part.subclass)) & (markers_table.marker_group == marker_group)]
                if len(markers_table) else pd.DataFrame()
            )
            result[f'{group}_{marker_group}_marker_pairs'] = len(subset)
            if len(subset):
                per_type = subset.groupby('subclass')[marker_metric_cols].mean()
                for column in per_type:
                    result[f'{group}_{marker_group}_marker_macro_{column}'] = float(per_type[column].mean())
                result[f'{group}_{marker_group}_marker_morans_i_rank_corr'] = corr(
                    subset['morans_i_truth'], subset['morans_i_generated'], rank=True
                )
            else:
                for column in marker_metric_cols:
                    result[f'{group}_{marker_group}_marker_macro_{column}'] = np.nan
                result[f'{group}_{marker_group}_marker_morans_i_rank_corr'] = np.nan

    result['false_present_types'] = int(table.false_presence.sum())
    result['missing_types'] = int(table.missing.sum())
    result['unseen_truth_cells'] = int(table.loc[table.unseen, 'truth_n'].sum())
    return result, table, markers_table
