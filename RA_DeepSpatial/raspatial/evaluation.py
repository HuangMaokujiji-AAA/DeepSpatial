"""Unpaired section metrics with explicit missing-type and unmatched-point penalties."""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree, distance
from scipy.sparse.csgraph import maximum_bipartite_matching, min_weight_full_bipartite_matching
from scipy.special import rel_entr
from scipy.stats import spearmanr


def corr(a, b, rank=False):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if len(a) < 2 or a.std() <= 1e-12 or b.std() <= 1e-12:
        return np.nan
    return float(spearmanr(a, b).statistic if rank else np.corrcoef(a, b)[0, 1])


def js(a, b):
    a, b = np.asarray(a, float)+1e-9, np.asarray(b, float)+1e-9
    a, b = a/a.sum(-1, keepdims=True), b/b.sum(-1, keepdims=True)
    middle = (a+b)/2
    return .5*(rel_entr(a, middle).sum(-1)+rel_entr(b, middle).sum(-1))


def expression(a, b):
    return dict(mse=float(np.mean((a-b)**2)), mae=float(np.mean(np.abs(a-b))), pcc=corr(a, b))


def assignment(a, b, cap):
    """Exact capped Hungarian/OSPA distance, sparse dummy nodes punish unmatched cells."""
    if not len(a) or not len(b):
        return float(cap), np.empty(0, int), np.empty(0, int)
    swapped = len(a) > len(b)
    x, y = (b, a) if swapped else (a, b)
    n, m = len(x), len(y)
    edges = cKDTree(x).sparse_distance_matrix(cKDTree(y), cap, output_type='coo_matrix')
    # Each small-side cell has its own dummy; larger-side unmatched penalty is constant.
    row = np.concatenate((edges.row, np.arange(n)))
    col = np.concatenate((edges.col, m+np.arange(n)))
    costs = np.concatenate((edges.data, np.full(n, cap)))+1e-9
    graph = sparse.csr_matrix((costs, (row, col)), shape=(n, m+n))
    ri, ci = min_weight_full_bipartite_matching(graph)
    total = float(graph[ri, ci].sum()-n*1e-9+(m-n)*cap)/m
    real = ci < m
    return total, (ci[real] if swapped else ri[real]), (ri[real] if swapped else ci[real])


def composition(xy, labels, classes, radius):
    codes = np.array([classes.get(str(k), len(classes)) for k in labels])
    n = len(xy)
    tree = cKDTree(xy)
    edges = tree.sparse_distance_matrix(tree, radius, output_type='coo_matrix')
    keep = edges.row != edges.col
    graph = sparse.csr_matrix((np.ones(int(keep.sum())), (edges.row[keep], edges.col[keep])), shape=(n, n))
    onehot = sparse.csr_matrix((np.ones(n), (np.arange(n), codes)), shape=(n, len(classes)+1))
    values = (graph@onehot).toarray()
    mass = values.sum(1)
    return np.divide(values, mass[:, None], out=np.zeros_like(values), where=mass[:, None] > 0), mass


def evaluate(truth, prediction, state, config, markers, independent):
    tx, tg, ty = truth['physical'], truth['g'], truth['labels']
    px = np.asarray(prediction.obsm['spatial'])
    pg = prediction.X.toarray() if sparse.issparse(prediction.X) else np.asarray(prediction.X)
    py = prediction.obs.cell_class.astype(str).to_numpy()
    if list(truth['genes']) != state['genes'] or prediction.var_names.tolist() != state['genes']:
        raise ValueError('Frozen training gene order differs at evaluation.')
    if not len(px) or not np.isfinite(pg).all() or not np.isfinite(px).all():
        raise ValueError('Empty or nonfinite prediction.')
    tl, pl = np.log1p(np.maximum(tg, 0)), np.log1p(np.maximum(pg, 0))
    result = dict(truth_n=len(tx), generated_n=len(px), total_count_relative_error=abs(len(px)-len(tx))/len(tx),
                  negative_expression_fraction=float((pg < 0).mean()), source_anchor_fraction=float((prediction.obs.generation_mode == 'source_anchor').mean()))
    for scale, t, p in (('provided_X', tg, pg), ('log1p', tl, pl)):
        result.update({f'global_{scale}_{key}': value for key, value in expression(t.mean(0), p.mean(0)).items()})
    classes = {name: i for i, name in enumerate(state['classes'])}
    tniche, tmass = composition(tx, ty, classes, config['evaluation']['niche_radius_um'])
    pniche, pmass = composition(px, py, classes, config['evaluation']['niche_radius_um'])
    width = config['evaluation']['spatial_bin_um']
    _, bins = np.unique(np.floor(np.vstack((tx, px))/width).astype(np.int64), axis=0, return_inverse=True)
    tb, pb = bins[:len(tx)], bins[len(tx):]
    nb = int(bins.max()+1)
    train_counts = dict(zip(state['classes'], state['counts'])); total_train = sum(train_counts.values())
    rows, marker_rows = [], []
    for name in sorted(set(state['classes']) | set(ty) | set(py)):
        ti, pi = np.flatnonzero(ty == name), np.flatnonzero(py == name)
        nt, npred = len(ti), len(pi)
        if not nt and not npred:
            continue
        a, b = tx[ti], px[pi]
        fraction = train_counts.get(name, 0)/total_train
        row = dict(subclass=name, train_n=train_counts.get(name, 0), train_fraction=fraction,
                   is_rare=0 < fraction <= config['evaluation']['rare_fraction'], is_ultra_rare=0 < fraction <= .001,
                   unseen=not train_counts.get(name, 0), truth_n=nt, generated_n=npred,
                   count_relative_error=abs(npred-nt)/nt if nt else np.nan, false_presence=nt == 0 and npred > 0,
                   missing=nt > 0 and npred == 0)
        t2p = cKDTree(b).query(a)[0] if nt and npred else np.full(nt, np.inf)
        p2t = cKDTree(a).query(b)[0] if nt and npred else np.full(npred, np.inf)
        for radius in config['evaluation']['radii_um']:
            key = f'{radius:g}um'
            row[f'coverage_recall_{key}'] = float((t2p <= radius).mean()) if nt else np.nan
            row[f'coverage_precision_{key}'] = float((p2t <= radius).mean()) if npred else 0. if nt else np.nan
            if nt and npred:
                edges = cKDTree(a).sparse_distance_matrix(cKDTree(b), radius, output_type='coo_matrix')
                graph = sparse.csr_matrix((np.ones(len(edges.data)), (edges.row, edges.col)), shape=(nt, npred))
                matched = int((maximum_bipartite_matching(graph, perm_type='column') >= 0).sum())
            else:
                matched = 0
            row[f'one_to_one_precision_{key}'] = matched/npred if npred else 0. if nt else np.nan
            row[f'one_to_one_recall_{key}'] = matched/nt if nt else np.nan
            row[f'one_to_one_f1_{key}'] = 2*matched/(nt+npred) if nt+npred else np.nan
        cap = config['evaluation']['matching_cap_um']
        capped, im, jm = assignment(a, b, cap)
        row['hungarian_capped_mean_um'] = capped
        row['matching_cap_um'] = cap
        row['matched_pairs_for_niche'] = len(im)
        if nt and npred and nt*npred <= config['evaluation']['max_assignment_entries']:
            costs = distance.cdist(a, b)
            ri, ci = linear_sum_assignment(costs)
            ds = costs[ri, ci]
            row.update(hungarian_matched_mean_um=float(ds.mean()), hungarian_matched_median_um=float(np.median(ds)))
        else:
            row.update(hungarian_matched_mean_um=np.nan, hungarian_matched_median_um=np.nan)
        valid_niche = (tmass[ti[im]] > 0) & (pmass[pi[jm]] > 0)
        row['niche_evaluable_pairs'] = int(valid_niche.sum())
        row['niche_missing_neighbor_pairs'] = int((~valid_niche).sum())
        if valid_niche.any():
            ni, nj = ti[im[valid_niche]], pi[jm[valid_niche]]
            row['local_composition_js'] = float(js(tniche[ni], pniche[nj]).mean())
            smooth_a, smooth_b = tniche[ni]+1e-6, pniche[nj]+1e-6
            smooth_a /= smooth_a.sum(1, keepdims=True); smooth_b /= smooth_b.sum(1, keepdims=True)
            row['local_composition_kl'] = float(rel_entr(smooth_a, smooth_b).sum(1).mean())
        else:
            row['local_composition_js'], row['local_composition_kl'] = np.nan, np.nan
        row['niche_neighbor_mass_error'] = abs(float(tmass[ti].mean())-float(pmass[pi].mean())) if nt and npred else np.nan
        code = classes.get(name)
        if code is not None and state['preferred_neighbor_code'][code] >= 0:
            pref = state['classes'][state['preferred_neighbor_code'][code]]
            row['preferred_neighbor_subclass'] = pref
            def preference(xy, labels, index):
                if not len(index):
                    return np.nan
                # Exclude self if the learned preferred neighbor is the same type.
                pool = np.flatnonzero(labels == pref)
                if not len(pool) or (pref == name and len(pool) < 2):
                    return np.nan
                ds = cKDTree(xy[pool]).query(xy[index], k=2 if pref == name else 1)[0]
                return float(ds[:, 1].mean() if pref == name else ds.mean())
            dt, dp = preference(tx, ty, ti), preference(px, py, pi)
            row.update(truth_preferred_neighbor_distance_um=dt, generated_preferred_neighbor_distance_um=dp,
                       niche_preference_distance_error_um=abs(dt-dp))
        d0, d1 = np.bincount(tb[ti], minlength=nb), np.bincount(pb[pi], minlength=nb)
        row['density_pcc'] = corr(d0, d1)
        row['density_tv'] = float(np.abs(d0/max(nt, 1)-d1/max(npred, 1)).sum()/2)
        if nt:
            true_mean = tl[ti].mean(0)
            pred_mean = pl[pi].mean(0) if npred else np.zeros(pl.shape[1])
            row.update({f'type_log_expression_{k}': v for k, v in expression(true_mean, pred_mean).items()})
            row['type_log_expression_cosine'] = float(true_mean@pred_mean/max(np.linalg.norm(true_mean)*np.linalg.norm(pred_mean), 1e-12))
        c = classes.get(name)
        for group, columns in (('loss', markers.get(c, [])), ('heldout_C', independent.get(c, []))):
            for j in columns:
                field_t = np.bincount(tb[ti], weights=tl[ti, j], minlength=nb)
                field_p = np.bincount(pb[pi], weights=pl[pi, j], minlength=nb)
                mean_t = float(tl[ti, j].mean()) if nt else np.nan
                mean_p = float(pl[pi, j].mean()) if npred else 0.
                other_t, other_p = tl[ty != name, j], pl[py != name, j]
                enrichment_t = mean_t/(float(other_t.mean())+1e-6) if len(other_t) else np.nan
                enrichment_p = mean_p/(float(other_p.mean())+1e-6) if len(other_p) else np.nan
                marker_rows.append(dict(subclass=name, gene=state['genes'][j], gene_symbol=state['symbols'][j], marker_group=group,
                                        spatial_pcc=corr(field_t, field_p), spatial_spearman=corr(field_t, field_p, True),
                                        truth_mean_log=mean_t, generated_mean_log=mean_p,
                                        truth_enrichment=enrichment_t, generated_enrichment=enrichment_p,
                                        enrichment_absolute_error=abs(enrichment_t-enrichment_p)))
        rows.append(row)
    table, markers_table = pd.DataFrame(rows), pd.DataFrame(marker_rows)
    for group, mask in (('all', table.truth_n > 0), ('rare', table.is_rare & (table.truth_n > 0)),
                        ('ultra_rare', table.is_ultra_rare & (table.truth_n > 0)), ('common', ~table.is_rare & ~table.unseen & (table.truth_n > 0))):
        part = table[mask]
        result[group+'_types'] = len(part)
        for column in table.select_dtypes(include=np.number):
            if column not in ('train_n', 'train_fraction', 'truth_n', 'generated_n'):
                result[f'{group}_macro_{column}'] = float(part[column].mean()) if len(part) else np.nan
        for marker_group in ('loss', 'heldout_C'):
            subset = markers_table[(markers_table.subclass.isin(part.subclass)) & (markers_table.marker_group == marker_group)] if len(markers_table) else pd.DataFrame()
            result[f'{group}_{marker_group}_marker_pairs'] = len(subset)
            if len(subset):
                per_type = subset.groupby('subclass')[['spatial_pcc', 'spatial_spearman', 'enrichment_absolute_error']].mean()
                for column in per_type:
                    result[f'{group}_{marker_group}_marker_macro_{column}'] = float(per_type[column].mean())
    result['false_present_types'] = int(table.false_presence.sum())
    result['missing_types'] = int(table.missing.sum())
    result['unseen_truth_cells'] = int(table.loc[table.unseen, 'truth_n'].sum())
    return result, table, markers_table
