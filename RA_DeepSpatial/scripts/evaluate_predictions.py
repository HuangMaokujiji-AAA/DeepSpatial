#!/usr/bin/env python3
"""Evaluate external .h5ad predictions and physical lower-bound baselines (source_copy, uot_linear)
under the identical 5-dimensional RA_DeepSpatial evaluation protocol and frozen train-only probe."""
import argparse
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def build_physical_baseline(kind, left, right, sched, state, config):
    import anndata as ad
    import numpy as np
    import pandas as pd
    from scipy import sparse

    if kind not in ('source_copy', 'uot_linear'):
        raise ValueError(f'Unknown physical baseline: {kind}')
    total = len(sched['parent'])
    gd, cd = len(state['genes']), len(state['classes'])
    physical = np.empty((total, 2), dtype=np.float64)
    g = np.empty((total, gd), dtype=np.float32)
    source_y = np.empty(total, dtype=np.int64)
    source_id = np.empty(total, dtype=object)
    source_z = np.empty(total, dtype=np.float64)
    parents, forward, targets = sched['parent'], sched['forward'], sched['target']
    supported = (targets >= 0) & (kind == 'uot_linear')

    for side, (src, dst) in ((True, (left, right)), (False, (right, left))):
        mask = forward == side
        if not mask.any():
            continue
        p = parents[mask]
        t = targets[mask]
        source_y[mask] = src['y'][p]
        source_id[mask] = src['ids'][p]
        source_z[mask] = src['z_um']
        alpha = sched['fraction'] if side else 1.0 - sched['fraction']
        has_pair = (t >= 0) & (kind == 'uot_linear')
        idx_all = np.flatnonzero(mask)
        if (~has_pair).any():
            idx_copy = idx_all[~has_pair]
            p_copy = p[~has_pair]
            physical[idx_copy] = src['physical'][p_copy]
            g[idx_copy] = src['g'][p_copy]
        if has_pair.any():
            idx_lin = idx_all[has_pair]
            p_lin, t_lin = p[has_pair], t[has_pair]
            physical[idx_lin] = (1.0 - alpha) * src['physical'][p_lin] + alpha * dst['physical'][t_lin]
            g[idx_lin] = ((1.0 - alpha) * src['g'][p_lin] + alpha * dst['g'][t_lin]).astype(np.float32)

    before = g.copy()
    if kind == 'uot_linear' and config['reconstruction'].get('sparsify', True):
        nnz = np.empty(total, dtype=int)
        nnz[forward] = (left['g'][parents[forward]] > 0).sum(1)
        nnz[~forward] = (right['g'][parents[~forward]] > 0).sum(1)
        for start in range(0, total, 256):
            chunk = g[start:start + 256]
            order = np.argsort(-chunk, axis=1, kind='stable')
            keep = np.zeros(chunk.shape, dtype=bool)
            np.put_along_axis(keep, order, np.arange(gd)[None, :] < nnz[start:start + len(chunk), None], axis=1)
            chunk[~keep] = 0.0
        anchor_mask = ~supported
        before[anchor_mask] = g[anchor_mask]

    names = np.asarray(state['classes'])
    continuous = np.eye(cd, dtype=np.float32)[source_y]
    mode = np.where(supported, 'uot_linear', 'source_copy' if kind == 'source_copy' else 'source_anchor')
    obs = pd.DataFrame(
        dict(
            cell_class=names[source_y],
            source_cell_class=names[source_y],
            source_cell_id=source_id,
            source_z=source_z,
            source_cell_index=parents,
            source_direction=np.where(forward, 'forward', 'backward'),
            target_cell_index=targets,
            z_um=sched['z_um'],
            generation_mode=mode,
        ),
        index=[f'{kind}_{i}' for i in range(total)],
    )
    return ad.AnnData(
        X=sparse.csr_matrix(g),
        obs=obs,
        var=pd.DataFrame({'gene_symbol': state['symbols']}, index=state['genes']),
        obsm={'spatial': physical, 'cell_class_continuous': continuous},
        layers={'expression_before_sparsification': before},
        uns={'generation': dict(target_depth_um=sched['z_um'], baseline_kind=kind, coordinate_units='um')},
    )


def align_external_prediction(pred, state, label_column='cell_class', spatial_key='spatial', coordinate_scale=1.0):
    import anndata as ad
    import numpy as np
    import pandas as pd
    from scipy import sparse

    if spatial_key not in pred.obsm:
        raise ValueError(f'Missing obsm[{spatial_key!r}] in external prediction.')
    if label_column not in pred.obs:
        raise ValueError(f'Missing obs[{label_column!r}] in external prediction.')
    var_genes = pred.var_names.astype(str).tolist()
    if set(var_genes) != set(state['genes']):
        if 'gene_symbol' in pred.var and set(pred.var['gene_symbol'].astype(str)) == set(state['symbols']):
            sym_lookup = {s: i for i, s in enumerate(pred.var['gene_symbol'].astype(str))}
            order = [sym_lookup[s] for s in state['symbols']]
        else:
            raise ValueError('External prediction gene panel does not match state["genes"] or state["symbols"].')
    else:
        gene_lookup = {g: i for i, g in enumerate(var_genes)}
        order = [gene_lookup[g] for g in state['genes']]
    x_mat = pred.X[:, order]
    if not sparse.issparse(x_mat):
        x_mat = sparse.csr_matrix(np.asarray(x_mat, dtype=np.float32))
    obs = pred.obs.copy()
    obs['cell_class'] = obs[label_column].astype(str).to_numpy()
    if 'generation_mode' not in obs:
        obs['generation_mode'] = 'external'
    xy = np.asarray(pred.obsm[spatial_key], dtype=np.float64)[:, :2] * float(coordinate_scale)
    layers = {}
    if 'expression_before_sparsification' in pred.layers:
        raw_pre = pred.layers['expression_before_sparsification']
        layers['expression_before_sparsification'] = raw_pre[:, order]
    return ad.AnnData(
        X=x_mat,
        obs=obs,
        var=pd.DataFrame({'gene_symbol': state['symbols']}, index=state['genes']),
        obsm={'spatial': xy},
        layers=layers,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/abca2.json')
    parser.add_argument('--cache', default='cache/abca2')
    parser.add_argument('--experiment-output', default='outputs/abca2_seed42',
                        help='Experiment directory holding expression_probe.pt (and freeze guard for test role).')
    parser.add_argument('--eval-output', default='outputs/abca2_baselines_eval',
                        help='Output directory for baseline / external evaluation tables.')
    parser.add_argument('--role', choices=['validation', 'test'], default='validation')
    parser.add_argument('--baselines', nargs='*', choices=['source_copy', 'uot_linear'],
                        default=['source_copy', 'uot_linear'])
    parser.add_argument('--external', nargs='*', default=[],
                        help='External entries formatted as NAME=DIR_WITH_SECTION_H5AD (expects <DIR>/<sec_XXX>.h5ad).')
    parser.add_argument('--external-label-col', default='cell_class')
    parser.add_argument('--external-coord-scale', type=float, default=1.0)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    import anndata as ad
    import numpy as np
    import pandas as pd
    import torch
    from raspatial.data import prepare, SliceStore, read_section
    from raspatial.evaluation import evaluate
    from raspatial.experiment import MAIN_TABLE_COLUMNS, ensure_expression_probe, verify_freeze
    from raspatial.io import digest, read_json, write_json
    from raspatial.markers import load_markers
    from raspatial.reconstruction import schedule, add_targets, write_h5ad
    from raspatial.transport import Transports

    cfg = read_json(PROJECT / args.config)
    cache = (PROJECT / args.cache).resolve()
    exp_out = (PROJECT / args.experiment_output).resolve()
    eval_out = (PROJECT / args.eval_output).resolve()
    eval_out.mkdir(parents=True, exist_ok=True)

    state, splits, paths = prepare(PROJECT, cfg, cache)
    if args.role == 'test' and (exp_out / 'frozen_test_protocol.json').is_file():
        verify_freeze(exp_out, splits)
    elif args.role == 'test':
        raise ValueError(f'Refusing to open test sections before freeze protocol exists in {exp_out}.')

    marker_path = (PROJECT / cfg['markers']['path']).resolve()
    markers, independent, _ = load_markers(marker_path, state, cfg)
    store = SliceStore(cache, state, cfg['data']['cache_slices'])
    transports = Transports(cache, store, cfg)
    device = torch.device(args.device)
    probe_root = exp_out if exp_out.exists() else eval_out
    probe, probe_hash = ensure_expression_probe(probe_root, store, state, cfg, device)

    ext_specs = {}
    for item in args.external:
        if '=' not in item:
            raise ValueError(f'--external entry must be NAME=DIR, got {item!r}')
        k, v = item.split('=', 1)
        ext_specs[k.strip()] = Path(v).resolve()

    targets = [r for r in splits if r['role'] == args.role]
    all_rows = []
    for target in targets:
        sec_name = target['section']
        left, right = store[target['context_left']], store[target['context_right']]
        natural = schedule(left, right, target['z_um'], state, cfg)
        local = add_targets(natural, left, right, transports, 'local', cfg['seed'])

        generated_preds = {}
        for kind in args.baselines:
            folder = eval_out / kind / args.role / sec_name
            pred_path = folder / 'prediction.h5ad'
            if not pred_path.is_file():
                sched = local if kind == 'uot_linear' else dict(natural, target=np.full(len(natural['parent']), -1, np.int64))
                pred = build_physical_baseline(kind, left, right, sched, state, cfg)
                write_h5ad(pred_path, pred)
            generated_preds[kind] = pred_path

        for ext_name, ext_dir in ext_specs.items():
            cand = ext_dir / f'{sec_name}.h5ad'
            if not cand.is_file():
                raise FileNotFoundError(f'Missing external prediction for {sec_name}: {cand}')
            generated_preds[ext_name] = cand

        truth = read_section(paths[sec_name], cfg)
        truth_hash = digest(paths[sec_name])
        for method_name, pred_path in generated_preds.items():
            folder = eval_out / method_name / args.role / sec_name
            folder.mkdir(parents=True, exist_ok=True)
            raw_pred = ad.read_h5ad(pred_path)
            if method_name in ext_specs:
                raw_pred = align_external_prediction(
                    raw_pred, state, label_column=args.external_label_col, coordinate_scale=args.external_coord_scale
                )
            metrics, per_type, per_marker = evaluate(truth, raw_pred, state, cfg, markers, independent, probe=probe)
            per_type.to_csv(folder / 'per_type.csv', index=False)
            per_marker.to_csv(folder / 'per_marker.csv', index=False)
            record = dict(arm=method_name, role=args.role, target=sec_name, seed=cfg['seed'],
                          truth_sha256=truth_hash, prediction_sha256=digest(pred_path),
                          probe_sha256=probe_hash, **metrics)
            write_json(folder / 'metrics.json', record)
            all_rows.append(record)
            print(f'Evaluated {method_name}/{args.role}/{sec_name}: OSPA1={metrics.get("rare_macro_ospa1_um", np.nan):.2f}um', flush=True)

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(eval_out / f'{args.role}_all_metrics.csv', index=False)
        headline = [c for c in MAIN_TABLE_COLUMNS if c in df.columns]
        if headline:
            main_df = df.groupby('arm', sort=False)[headline].agg(['mean', 'std'])
            main_df.columns = [f'{col}_{stat}' for col, stat in main_df.columns]
            main_df.reset_index().to_csv(eval_out / f'{args.role}_main_table.csv', index=False)


if __name__ == '__main__':
    main()
