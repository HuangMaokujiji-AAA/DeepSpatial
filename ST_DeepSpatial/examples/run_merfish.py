"""Run sequential reconstruction on the existing MERFISH data."""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from stspatial.experiment import run, VARIANTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT/'configs/merfish.json')
    parser.add_argument('--output', type=Path, default=PROJECT/'outputs/merfish_staged_v1')
    parser.add_argument('--holdouts', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--variants', choices=VARIANTS, nargs='+', default=['full'])
    parser.add_argument('--stage', choices=['all', 'train', 'generate', 'evaluate'], default='all')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--interventions', action='store_true', help='Fixed-point shuffles; no extra model training.')
    parser.add_argument('--excluded', type=int, nargs='*', default=[])
    parser.add_argument('--role', choices=['development', 'validation', 'test'], default='development')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    records = run(config, args.output, args.holdouts, args.seeds, args.variants, args.stage,
                  args.device, args.resume, args.excluded, args.role, args.interventions)
    print(f'Completed {args.stage}: {args.output}; {len(records)} evaluation record(s).')


if __name__ == '__main__':
    main()
