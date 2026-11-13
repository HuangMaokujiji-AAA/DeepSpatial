#!/usr/bin/env python3
"""Check actual 1122-gene/train-taxonomy shapes; no fitted model or biological scores."""
import argparse
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    import h5py
    import numpy as np
    import torch
    from raspatial.data import make_split, h5_values, neighborhood
    from raspatial.experiment import ARMS
    from raspatial.io import read_json, write_json
    from raspatial.model import BaseFlow, ResidualFlow
    from raspatial.training import seed_all
    torch.set_num_threads(1)
    cfg = read_json(PROJECT/'configs/abca2.json')
    rows, paths = make_split(PROJECT, cfg)
    train = [r for r in rows if r['role'] == 'train']
    counts = {}
    for row in train:
        with h5py.File(paths[row['section']]) as f:
            labels, n = np.unique(h5_values(f['obs'], cfg['data']['label_key']), return_counts=True)
            for name, value in zip(labels, n):
                counts[str(name)] = counts.get(str(name), 0)+int(value)
    with h5py.File(paths[train[0]['section']]) as f:
        genes, symbols = h5_values(f['var'], 'ensembl'), h5_values(f['var'], 'gene_symbol')
        g = f['X'][:max(args.batch_size, 64)].astype(np.float32)
        xy = f['obsm/spatial'][:len(g)]*cfg['data']['coordinate_to_um']
        labels = h5_values(f['obs'], cfg['data']['label_key'])[:len(g)]
    classes = sorted(counts); lookup = {c: i for i, c in enumerate(classes)}
    y = np.array([lookup[str(c)] for c in labels])
    lg = np.log1p(g)
    cols = np.argsort(-lg.std(0), kind='stable')[:cfg['data']['niche_genes']]
    state = dict(genes=genes.tolist(), symbols=symbols.tolist(), classes=classes, counts=[counts[c] for c in classes],
                 gene_scale=np.maximum(g.std(0), .001).tolist(), log_mean=lg.mean(0).tolist(),
                 log_scale=np.maximum(lg.std(0), .001).tolist(), niche_gene_indices=cols.tolist(),
                 niche_dim=len(classes)+2*len(cols)+7)
    h = neighborhood(xy, y, g, state, cfg)
    if not np.isfinite(h).all():
        raise FloatingPointError('Real-panel neighborhood features are nonfinite.')
    seed_all(42); device = torch.device(args.device)
    base = BaseFlow(state, cfg).to(device)
    residual = ResidualFlow(state, cfg, ARMS['full'], base.state_dict()).to(device)
    n = min(args.batch_size, len(g))
    c = torch.nn.functional.one_hot(torch.as_tensor(y[:n], device=device), len(classes)).float()
    tensor = lambda a: torch.as_tensor(a.copy(), dtype=torch.float32, device=device)
    x = ((xy-xy.min(0))/np.maximum(np.ptp(xy, axis=0), 1)).astype(np.float32)
    inputs = dict(xt=tensor(x[:n]), gt=tensor(g[:n]), ct=c, source_c=c, h0=tensor(h[:n]), h1=tensor(h[:n]),
                  t=torch.full((n,), .5, device=device), zt=torch.full((n, 1), .5, device=device),
                  delta_z=torch.full((n, 1), .1, device=device))
    expected, actual = base(**inputs), residual(**inputs)
    for a, b in zip(expected, actual):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    sum(a.square().mean() for a in actual).backward()
    if not torch.isfinite(residual.head[-1].weight.grad).all():
        raise FloatingPointError('Real-panel shape check produced nonfinite gradients.')
    report = dict(purpose='shape_and_gradient_engineering_only_not_performance', genes=len(genes), training_subclasses=len(classes),
                  train_cells=sum(counts.values()), niche_dimension=h.shape[1], source_section=train[0]['section'],
                  real_cells_in_feature_check=len(g), forward_batch_size=n, device=str(device),
                  base_parameters=sum(p.numel() for p in base.parameters()), zero_residual_exact=True,
                  finite_neighborhood=True, finite_gradient=True, formal_training_run=False)
    if args.output:
        write_json(args.output, report)
    print(report)


if __name__ == '__main__':
    main()
