"""实验 G 的纯几何候选、局部 UOT 与受支持配对抽样。"""
import warnings

import numpy as np
import ot
from scipy import sparse
from scipy.spatial import cKDTree


def matching_parameters(config, gap_um):
    """按声明的参考层间距缩放物理半径；不从隐藏切片选择参数。"""
    required = {'max_distance_um', 'spatial_scale_um', 'reference_gap_um'}
    if set(config) != required:
        raise ValueError(f'local_matching_config must contain exactly {sorted(required)}.')
    values = {k: float(config[k]) for k in required}
    if not np.isfinite([*values.values(), gap_um]).all() or min(*values.values(), gap_um) <= 0:
        raise ValueError('Local matching scales and physical gap must be finite and positive.')
    factor = gap_um / values['reference_gap_um']
    return values['max_distance_um'] * factor, values['spatial_scale_um'] * factor


def local_support(x0, labels0, x1, labels1, radius_um):
    """只保留同类且位于物理半径内的边；零距离边仍是有效候选。"""
    x0, x1 = np.asarray(x0, float), np.asarray(x1, float)
    labels0, labels1 = np.asarray(labels0), np.asarray(labels1)
    if (x0.shape != (len(labels0), 2) or x1.shape != (len(labels1), 2)
            or not np.isfinite(x0).all() or not np.isfinite(x1).all()
            or not np.isfinite(radius_um) or radius_um <= 0):
        raise ValueError('Local support needs finite N×2 physical coordinates and a positive radius.')
    counts = [np.zeros(len(x), dtype=np.int64) for x in (x0, x1)]
    nearest = [np.full(len(x), np.inf) for x in (x0, x1)]
    rows, cols, distances = [], [], []
    for label in np.intersect1d(labels0, labels1):
        r, c = np.flatnonzero(labels0 == label), np.flatnonzero(labels1 == label)
        left, right = cKDTree(x0[r]), cKDTree(x1[c])
        nearest[0][r], nearest[1][c] = right.query(x0[r])[0], left.query(x1[c])[0]
        edges = left.sparse_distance_matrix(right, radius_um, output_type='coo_matrix')
        counts[0][r] = np.bincount(edges.row, minlength=len(r))
        counts[1][c] = np.bincount(edges.col, minlength=len(c))
        rows.append(r[edges.row]); cols.append(c[edges.col]); distances.append(edges.data)
    def join(parts, dtype):
        return np.concatenate(parts).astype(dtype, copy=False) if parts else np.empty(0, dtype=dtype)
    support = [dict(candidate_count=n, nearest_same_type_um=d,
                    nearest_candidate_um=np.where(n > 0, d, np.nan), radius_um=float(radius_um))
               for n, d in zip(counts, nearest)]
    return join(rows, np.int64), join(cols, np.int64), join(distances, float), support


def compute_local_uot(x0, g0, labels0, x1, g1, labels1, gap_um, config,
                      alpha_spatial=.5, uot_reg=.05, uot_tau=.05):
    """在同类局部支持上求解 UOT；隔离无边端点，不重归一原始边际。"""
    if not np.isfinite([alpha_spatial, uot_reg, uot_tau]).all() or not 0 <= alpha_spatial <= 1 or uot_reg <= 0 or uot_tau <= 0:
        raise ValueError('Use alpha in [0,1] and positive UOT regularization.')
    radius, scale = matching_parameters(config, gap_um)
    rows, cols, distances, support = local_support(x0, labels0, x1, labels1, radius)
    g0, g1 = np.asarray(g0, float), np.asarray(g1, float)
    if (g0.ndim != 2 or g1.ndim != 2 or g0.shape[1] != g1.shape[1]
            or len(g0) != len(x0) or len(g1) != len(x1)
            or not np.isfinite(g0).all() or not np.isfinite(g1).all()):
        raise ValueError('Local UOT requires finite expression arrays in the same gene order.')
    unit = []
    for genes in (g0, g1):
        norm = np.linalg.norm(genes, axis=1, keepdims=True)
        unit.append(np.divide(genes, norm, out=np.zeros_like(genes), where=norm > 0))
    # 保留原表达余弦成本的全局最大值归一化，分块避免构造全局密集距离矩阵。
    gene_max = 0.
    for start in range(0, len(g0), 256):
        gene_max = max(gene_max, float(np.clip(1 - unit[0][start:start + 256] @ unit[1].T, 0, 2).max()))
    gene_cost = np.empty(len(rows))
    for start in range(0, len(rows), 8192):
        end = start + 8192
        gene_cost[start:end] = np.clip(1 - np.einsum('ij,ij->i', unit[0][rows[start:end]], unit[1][cols[start:end]]), 0, 2)
    edge_cost = alpha_spatial * (distances / scale)**2 + (1 - alpha_spatial) * gene_cost / (gene_max + 1e-9)
    data, out_rows, out_cols = [], [], []
    labels0, labels1 = np.asarray(labels0), np.asarray(labels1)
    for label in np.unique(labels0[rows]):
        mask = labels0[rows] == label
        r, c = np.unique(rows[mask]), np.unique(cols[mask])
        ri, ci = np.searchsorted(r, rows[mask]), np.searchsorted(c, cols[mask])
        cost = np.full((len(r), len(c)), np.inf)
        cost[ri, ci] = edge_cost[mask]
        # 各类块仍使用完整切片的 a_i=1/N0、b_j=1/N1，不能把剩余支持重新归一为1。
        with warnings.catch_warnings(record=True) as caught:
            plan, log = ot.unbalanced.sinkhorn_unbalanced(
                np.full(len(r), 1 / len(x0)), np.full(len(c), 1 / len(x1)), cost,
                reg=uot_reg, reg_m=uot_tau, reg_type='kl', numItermax=100, stopThr=1e-3, log=True,
            )
        error = log['err'][-1] if log.get('err') else np.inf
        if (not np.isfinite(plan).all() or (plan < 0).any() or not np.isfinite(error) or error >= 1e-3
                or any('Numerical' in str(w.message) for w in caught)
                or (plan.sum(0) <= 0).any() or (plan.sum(1) <= 0).any()):
            raise RuntimeError(f'Local UOT failed for class {label}; inspect spatial scales and solver convergence.')
        assert not np.any(plan[~np.isfinite(cost)])
        weights = plan[ri, ci]
        positive = weights > 0
        data.append(weights[positive]); out_rows.append(rows[mask][positive]); out_cols.append(cols[mask][positive])
    coupling = sparse.csr_matrix((np.concatenate(data), (np.concatenate(out_rows), np.concatenate(out_cols))),
                                 shape=(len(x0), len(x1))) if data else sparse.csr_matrix((len(x0), len(x1)))
    for side, matrix in enumerate((coupling, coupling.T.tocsr())):
        mass = np.asarray(matrix.sum(1)).ravel()
        log_data = matrix.copy()
        log_data.data = log_data.data * np.log(log_data.data)
        weighted_log = np.asarray(log_data.sum(1)).ravel()
        entropy = np.full(len(mass), np.nan)
        positive = mass > 0
        entropy[positive] = np.log(mass[positive]) - weighted_log[positive] / mass[positive]
        support[side].update(transport_mass=mass, conditional_entropy=entropy)
    return coupling, support


def sample_local_pairs(coupling, labels0, labels1, total, balanced_fraction):
    """只从有效正质量边抽样；均衡部分不得重新引入被拒绝的类块。"""
    edges = coupling.tocoo()
    if not len(edges.data) or edges.data.sum() <= 0:
        raise ValueError('No supported local transport edges are available.')
    if not 0 <= balanced_fraction <= 1:
        raise ValueError('balanced_fraction must lie in [0,1].')
    labels0, labels1 = np.asarray(labels0), np.asarray(labels1)
    edge_classes = labels0[edges.row]
    assert np.array_equal(edge_classes, labels1[edges.col])
    eligible = np.unique(edge_classes)
    balanced = int(round(total * balanced_fraction))
    draws = [np.random.choice(len(edges.data), total - balanced, p=edges.data / edges.data.sum())]
    counts = np.bincount(np.random.randint(len(eligible), size=balanced), minlength=len(eligible))
    for label, count in zip(eligible, counts):
        ids = np.flatnonzero(edge_classes == label)
        draws.append(np.random.choice(ids, int(count), p=edges.data[ids] / edges.data[ids].sum()))
    selected = np.concatenate(draws)
    np.random.shuffle(selected)
    return edges.row[selected], edges.col[selected], dict(balanced_pairs=balanced, eligible_classes=eligible.tolist())
