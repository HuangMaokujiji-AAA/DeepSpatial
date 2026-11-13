#!/usr/bin/env python3
"""Write only split geometry; do not open validation/test expression or labels."""
import argparse
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from raspatial.data import make_split
from raspatial.io import read_json, write_json
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/abca2.json')
    args = parser.parse_args()
    config = read_json(PROJECT/args.config)
    rows, _ = make_split(PROJECT, config)
    write_json(PROJECT/'docs/split_manifest.json', dict(seed=config['seed'], split_policy='fixed_sections_geometry_only', rows=rows))
    pd.DataFrame(rows).to_csv(PROJECT/'docs/split_manifest.csv', index=False)
    print(pd.DataFrame(rows).groupby('role').agg(sections=('section', 'size'), cells=('n_cells', 'sum')))
    print(pd.DataFrame(rows).query('role != "train"').to_string(index=False))
