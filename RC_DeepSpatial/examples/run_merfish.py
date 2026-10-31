"""Run independent MERFISH folds; no dependency on the HM/MM project directories."""
import argparse
import json
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))

from rcspatial.experiment import ARMS,run,compare


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=PROJECT/'configs/merfish.json')
    parser.add_argument('--output',type=Path,default=PROJECT/'outputs/merfish_v1')
    parser.add_argument('--holdouts',nargs='+',type=int,default=[1,2,3])
    parser.add_argument('--seeds',nargs='+',type=int,default=[42])
    parser.add_argument('--arms',nargs='+',choices=list(ARMS),default=['baseline','control','R0','A','B','AB','ABC'])
    parser.add_argument('--stage',choices=['audit','base','train','generate','evaluate','all','compare'],default='all')
    parser.add_argument('--device',default='auto',help='auto, cpu, cuda:0; set physical GPU with CUDA_VISIBLE_DEVICES.')
    parser.add_argument('--resume',action='store_true',help='Identical config/source/visible inputs only; epochs are total epochs.')
    parser.add_argument('--excluded',nargs='*',type=int,default=[],help='Outer-test slice indices excluded from validation training.')
    parser.add_argument('--role',choices=['development','validation','test'],default='development')
    args=parser.parse_args()
    if args.stage=='compare':
        compare(args.output);return
    config=json.loads(args.config.read_text(encoding='utf-8-sig'))
    records=run(config,args.output,args.holdouts,args.seeds,args.arms,args.stage,args.device,args.resume,args.excluded,args.role)
    if args.stage in ('all','evaluate'):
        compare(args.output,allow_incomplete=True)
    print(f'Completed {args.stage}: {args.output}; {len(records)} record(s).')


if __name__=='__main__':
    main()
