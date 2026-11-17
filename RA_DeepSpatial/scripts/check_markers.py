#!/usr/bin/env python3
"""Audit marker coverage using train-only log-expression statistics; no model training."""
from collections import Counter
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def main():
    import argparse
    import h5py
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from raspatial.data import h5_values, make_split
    from raspatial.io import digest, read_json, write_json
    from raspatial.markers import load_markers
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/abca2.json')
    args = parser.parse_args()
    config = read_json(PROJECT/args.config)
    rows, paths = make_split(PROJECT, config)
    train = [r for r in rows if r['role'] == 'train']
    counts = Counter()
    genes = symbols = None
    for row in train:
        with h5py.File(paths[row['section']]) as f:
            if genes is None:
                genes, symbols = h5_values(f['var'], 'ensembl'), h5_values(f['var'], 'gene_symbol')
            elif not np.array_equal(h5_values(f['var'], 'ensembl'), genes):
                raise ValueError('Training gene order mismatch.')
            labels, numbers = np.unique(h5_values(f['obs'], config['data']['label_key']), return_counts=True)
            counts.update({str(k): int(n) for k, n in zip(labels, numbers)})
    classes = sorted(counts); lookup = {name: i for i, name in enumerate(classes)}
    sums = np.zeros((len(classes), len(genes)), np.float64)
    global_sum = np.zeros(len(genes), np.float64)
    for row in train:
        with h5py.File(paths[row['section']]) as f:
            labels = h5_values(f['obs'], config['data']['label_key'])
            codes = np.array([lookup[str(k)] for k in labels])
            if not isinstance(f['X'], h5py.Dataset):
                raise ValueError('This standalone streaming audit expects this dataset\'s dense X.')
            for start in range(0, len(labels), 4096):
                g = np.asarray(f['X'][start:start+4096], np.float32)
                if not np.isfinite(g).all() or (g < 0).any():
                    raise ValueError('Nonfinite or negative training expression.')
                # Match prepare(): global log mean uses float64; type means use log1p float32.
                global_sum += np.log1p(g.astype(np.float64)).sum(0)
                membership = sparse.csr_matrix((np.ones(len(g)), (codes[start:start+len(g)], np.arange(len(g)))),
                                               shape=(len(classes), len(g)))
                sums += membership@np.log1p(g)
        print(f'Audited {row["section"]}', flush=True)
    n = np.array([counts[name] for name in classes])
    state = dict(classes=classes, counts=n.tolist(), genes=genes.tolist(), symbols=symbols.tolist(),
                 log_mean=(global_sum/n.sum()).tolist(), type_log_mean=(sums/n[:, None]).tolist())
    marker_path = PROJECT/config['markers']['path']
    selected, heldout, coverage = load_markers(marker_path, state, config)
    coverage['train_n'] = n
    coverage['train_fraction'] = n/n.sum()
    coverage['is_rare'] = coverage.train_fraction <= config['evaluation']['rare_fraction']
    coverage.to_csv(PROJECT/'resources/marker_coverage.csv', index=False)
    rare = coverage[coverage.is_rare]
    report = dict(purpose='train_only_marker_readiness_not_reconstruction_performance',
                  marker_sha256=digest(marker_path), train_sections=[r['section'] for r in train],
                  train_cells=int(n.sum()), genes=len(genes), training_subclasses=len(classes),
                  external_panel_covered_types=int((coverage.external_panel_n > 0).sum()),
                  loss_covered_types=int((coverage.loss_marker_n > 0).sum()),
                  heldout_C_covered_types=int((coverage.heldout_C_marker_n > 0).sum()),
                  rare_types=len(rare), rare_loss_covered_types=int((rare.loss_marker_n > 0).sum()),
                  rare_without_loss_markers=rare.loc[rare.loss_marker_n == 0, 'subclass'].tolist(),
                  validation_test_expression_read=False, formal_training_run=False)
    write_json(PROJECT/'resources/marker_audit.json', report)
    print({k: v for k, v in report.items() if k != 'train_sections'})


if __name__ == '__main__':
    main()
