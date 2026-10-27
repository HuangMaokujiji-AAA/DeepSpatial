"""Collect verified V1/S/G results as typed workbook inputs; no experiment changes."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[4]
PROJECT = ROOT / 'HM_DeepSpital'
WORK = Path(__file__).resolve().parents[1] / 'build/excel'
sys.path[:0] = [str(PROJECT), str(PROJECT / '.runtime_deps')]
import anndata as ad
import numpy as np
import pandas as pd
import torch
from examples.run_s import file_sha256


def records(frame):
    return json.loads(frame.to_json(orient='records', force_ascii=False, double_precision=15))


def main():
    analysis = PROJECT / 'outputs/g_holdouts_1_2_3/analysis'
    summary = pd.read_csv(analysis / 'metrics.csv')
    classes = pd.read_csv(analysis / 'cell_type_metrics.csv')
    assert len(summary) == 21 and len(classes) == 315
    assert (summary.groupby('run').heldout_id.nunique() == 3).all()
    integrity = pd.read_csv(analysis / 'integrity.csv')
    assert len(integrity) == 9 and (integrity.action == 'skip').all()
    status = []
    for fold in (1, 2, 3):
        folder = PROJECT / f'outputs/v1_holdouts_1_2_3/holdout_{fold}'
        checkpoint = torch.load(folder / 'checkpoints/last.ckpt', map_location='cpu', weights_only=False)
        final = torch.load(folder / 'model/model.ckpt', map_location='cpu', weights_only=False)['state_dict']
        assert checkpoint['epoch'] == 99
        assert set(final) == set(checkpoint['state_dict'])
        assert all(torch.equal(final[k], checkpoint['state_dict'][k]) for k in final)
        assert all(np.isfinite(t.numpy()).all() for t in final.values())
        volume = ad.read_h5ad(folder / 'generated.h5ad')
        assert np.isfinite(volume.X.data).all() and np.isfinite(volume.obsm['spatial_3d']).all()
        assert fold * 50 not in set(volume.obs.source_z)
        status.append(dict(stage='V1', holdout=fold, mode='real', epochs=100,
                           global_step=checkpoint['global_step'], generated_cells=volume.n_obs,
                           status='完整', detail='最后epoch=99；最终权重与检查点一致；生成文件可读',
                           source=str(folder.relative_to(ROOT)).replace('\\', '/')))
        s_folder = PROJECT / f'outputs/s_holdouts_1_2_3/holdout_{fold}'
        s_config = json.loads((s_folder / 'run_config.json').read_text(encoding='utf-8'))
        assert file_sha256(folder / 'model/model.ckpt') == s_config['experiment_s']['checkpoint_sha256']
        s_volume = ad.read_h5ad(s_folder / 'generated.h5ad', backed='r')
        status.append(dict(stage='S', holdout=fold, mode='real', epochs=0,
                           global_step=None, generated_cells=s_volume.n_obs, status='完整',
                           detail='复用V1权重，未新增训练；原权重SHA256一致',
                           source=str(s_folder.relative_to(ROOT)).replace('\\', '/')))
        s_volume.file.close()
        del volume, checkpoint, final
    for row in integrity.itertuples():
        status.append(dict(stage='G', holdout=row.holdout, mode=row.mode, epochs=int(row.completed_epochs),
                           global_step=int(row.global_step), generated_cells=int(row.generated_cells), status='完整',
                           detail='权重/生成/配对与S调度已核验',
                           source=f'HM_DeepSpital/outputs/g_holdouts_1_2_3/holdout_{row.holdout}/{row.mode}'))
    data = dict(summary=records(summary), classes=records(classes), status=status,
                comparisons=records(pd.read_csv(analysis / 'paired_comparisons.csv')),
                displacement=records(pd.read_csv(analysis / 'source_displacement.csv')),
                protocol=json.loads((analysis / 'protocol.json').read_text()),
                sources=[dict(name=p.name, path=str(p.relative_to(ROOT)).replace('\\', '/'), sha256=file_sha256(p))
                         for p in [analysis / 'metrics.csv', analysis / 'cell_type_metrics.csv', analysis / 'paired_comparisons.csv', analysis / 'source_displacement.csv', analysis / 'protocol.json']])
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / 'workbook_inputs.json').write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print('Verified: V1 3/3, S 3/3, G 9/9 complete; 21 summaries and 315 class records.')


if __name__ == '__main__':
    main()
