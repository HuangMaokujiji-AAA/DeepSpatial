"""从副本运行 V1；只读取可见切片，将模型、日志和生成结果写入副本目录。"""

import argparse
import glob
import json
from pathlib import Path
import re
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
if sys.platform == 'win32' and (PROJECT / '.runtime_deps').is_dir():
    # 本机验证依赖不覆盖 Linux 服务器已经安装的包。
    sys.path.insert(1, str(PROJECT / '.runtime_deps'))

import anndata as ad
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only

import deepspatial as ds


@rank_zero_only
def prepare_output(output, config):
    """仅主进程创建新目录并写入配置；重复实验仍禁止覆盖已有结果。"""
    output.mkdir(parents=True, exist_ok=False)
    (output / 'run_config.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    """解析配置、先排除留出文件，再训练、保存并重建可见切片间体积。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT / 'configs/merfish_v1.json')
    parser.add_argument('--output', type=Path, required=True, help='副本目录内的全新输出目录。')
    parser.add_argument('--holdout', type=int, nargs='*', default=[], help='不参与训练的文件编号；不读取其内容。')
    parser.add_argument('--devices', type=int, help='覆盖配置中的设备数量，例如 --devices 1。')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT):
        parser.error('Output must stay inside HM_DeepSpital to preserve the original experiment files.')
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    if args.devices is not None:
        if args.devices < 1:
            parser.error('--devices must be a positive integer.')
        config.setdefault('training', {})['devices'] = args.devices
    paths = [Path(p) for p in glob.glob(str(PROJECT / config['input_glob']), recursive=True)]
    paths.sort(key=lambda p: [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', p.name)])
    depths = config['slice_z_um']
    if len(paths) != len(depths) or len(paths) < 2:
        parser.error('The input file count must match slice_z_um and contain at least two slices.')
    heldout = set(args.holdout)
    if not heldout.issubset(range(len(paths))) or len(paths) - len(heldout) < 2:
        parser.error('Holdout indices must be valid and leave at least two observed slices.')
    config['run'] = {
        'implementation': str(Path(ds.__file__).resolve().relative_to(PROJECT)),
        'visible_files': [str(p.relative_to(PROJECT.parent)) for i, p in enumerate(paths) if i not in heldout],
        'heldout_indices': sorted(heldout), 'z_source': 'explicit_user_configuration',
    }
    prepare_output(output, config)
    pl.seed_everything(config.get('seed', 42), workers=True)
    observed = []
    for index, (path, depth) in enumerate(zip(paths, depths)):
        if index in heldout:
            continue
        adata = ad.read_h5ad(path)
        adata.obs[config['data'].get('z_key', 'z_coord')] = float(depth)
        observed.append(adata)
    model = ds.DeepSpatial()
    model.setup_data(observed, **config['data'])
    model.build_model(**config.get('model', {}))
    model.fit(save_dir=str(output / 'checkpoints'), save_ckpt=True, **config.get('training', {}))
    # DDP 的各进程都会返回这里；只让主进程保存最终模型并执行重建。
    if not model.module.trainer.is_global_zero:
        return
    model.save_model(output / 'model')
    result = model.reconstruct_full_volume(observed, **config.get('reconstruction', {}))
    result.write_h5ad(output / 'generated.h5ad')
    print(f'Saved {result.n_obs} generated cells to {output / "generated.h5ad"}')


if __name__ == '__main__':
    main()
