#!/usr/bin/env python3
"""Standalone entry point; defaults to physical server GPU 0."""
import argparse
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/abca2.json')
    parser.add_argument('--cache', default='cache/abca2')
    parser.add_argument('--output', default='outputs/abca2_seed42')
    parser.add_argument('--stage', choices=['prepare', 'train', 'validate', 'freeze', 'test', 'summary', 'export', 'all'], default='prepare')
    parser.add_argument('--arms', nargs='+', choices=['baseline', 'control', 'R0', 'full', 'no_A', 'no_B', 'no_C', 'no_D'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    from raspatial.experiment import run
    from raspatial.io import read_json
    run(PROJECT, read_json(PROJECT/args.config), PROJECT/args.cache, PROJECT/args.output,
        args.stage, args.arms, args.device, args.resume)


if __name__ == '__main__':
    main()
