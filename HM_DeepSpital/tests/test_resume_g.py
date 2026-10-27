"""实际训练中断/恢复检查；不作为实验性能证据。"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(PROJECT), str(PROJECT / '.runtime_deps')]

import numpy as np
import torch
from deepspatial import DeepSpatial
from deepspatial.module import DeepSpatialModule
from examples.run_g import run_experiment, file_sha256
from examples.resume_g import inspect_results, resume_experiment
from test_pipeline import slices

torch.set_num_threads(1)


class ResumeGTests(unittest.TestCase):
    def test_skip_resume_optimizer_ema_and_generation_only(self):
        with tempfile.TemporaryDirectory(dir=PROJECT) as temp:
            root = Path(temp)
            data = root / 'data'
            data.mkdir()
            visible = []
            for i in range(5):
                a = slices()[0]
                a.obs_names = [f'slice{i}_{j}' for j in range(a.n_obs)]
                a.obs['z_coord'] = float(i * 50)
                a.obsm['spatial'] += i
                a.write_h5ad(data / f'merfish_{i}.h5ad')
                if i != 2:
                    visible.append(a)
            config = dict(seed=42, input_glob=str(data / 'merfish_[0-4].h5ad'), slice_z_um=[0, 50, 100, 150, 200],
                          data=dict(spatial_key='spatial', z_key='z_coord', label_key='cell_class',
                                    batch_size=32, num_workers=0, n_samples_base=64,
                                    uot_reg=.05, uot_tau=.05, alpha_spatial=.5,
                                    balanced_fraction=.2, source_sampling='stratified',
                                    niche_config=dict(radius=10, programs={'p': ['Mbp']}, mode='real')),
                          model=dict(hidden_size=32, depth=1, num_heads=4, patch_size=2,
                                     sampling_method='euler', bidirectional_training=True),
                          training=dict(max_epochs=2, accelerator='cpu', devices=1),
                          reconstruction=dict(thickness=10., steps=4, chunk_size=4, save_intermediates=True,
                                              source_sampling='depth_type_stratified', depth_bin_width_um=10., depth_grid_origin_um=0.),
                          run=dict(heldout_indices=[2], visible_files=[f'data/merfish_{i}.h5ad' for i in (0, 1, 3, 4)]))
            model = DeepSpatial()
            model.setup_data(visible, **config['data'])
            model.build_model(**config['model'])
            source = root / 'S/holdout_2'
            source.mkdir(parents=True)
            reference = model.reconstruct_full_volume(visible, device='cpu', **config['reconstruction'])
            reference.write_h5ad(source / 'generated.h5ad')
            (source / 'run_config.json').write_text(json.dumps(config), encoding='utf-8')
            config['data']['local_matching_config'] = dict(max_distance_um=100., spatial_scale_um=50., reference_gap_um=100.)
            config_path = root / 'g.json'
            config_path.write_text(json.dumps(config), encoding='utf-8')
            real_fit = DeepSpatial.fit

            def interrupt_zero(instance, *args, **kwargs):
                if instance.niche_featurizer.mode == 'zero':
                    kwargs['max_epochs'] = 1
                    real_fit(instance, *args, **kwargs)
                    raise RuntimeError('simulated interruption')
                return real_fit(instance, *args, **kwargs)

            with patch.object(DeepSpatial, 'fit', new=interrupt_zero), self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                run_experiment(root / 'S', root / 'G', config_path, holdouts=(2,), device='cpu')
            fold = root / 'G/holdout_2'
            before = {p: file_sha256(p) for p in (fold / 'real').rglob('*') if p.is_file()}
            plan = inspect_results(root / 'G', holdouts=(2,)).set_index('mode')
            self.assertEqual(plan.loc['real', 'action'], 'skip')
            self.assertEqual(plan.loc['zero', 'action'], 'resume_training')
            self.assertEqual(plan.loc['zero', 'completed_epochs'], 1)
            self.assertEqual(plan.loc['shuffle', 'action'], 'train_new')
            interrupted = torch.load(fold / 'zero/checkpoints/last.ckpt', map_location='cpu', weights_only=False)
            calls, restored = [], []

            def record_fit(instance, *args, **kwargs):
                calls.append((instance.niche_featurizer.mode, kwargs['max_epochs'], kwargs.get('resume_ckpt_path')))
                return real_fit(instance, *args, **kwargs)

            def check_train_start(module):
                if module.trainer.global_step:
                    self.assertEqual(module.trainer.global_step, interrupted['global_step'])
                    state = module.trainer.optimizers[0].state_dict()
                    old = interrupted['optimizer_states'][0]
                    self.assertEqual(set(state['state']), set(old['state']))
                    for k in old['state']:
                        for name, value in old['state'][k].items():
                            self.assertTrue(torch.equal(value, state['state'][k][name]))
                    for k, value in module.state_dict().items():
                        if k.startswith('ema_model.'):
                            self.assertTrue(torch.equal(value.cpu(), interrupted['state_dict'][k]))
                    restored.append(True)

            with patch.object(DeepSpatial, 'fit', new=record_fit), \
                    patch.object(DeepSpatialModule, 'on_train_start', new=check_train_start), \
                    patch('deepspatial.data_utils.dataset.compute_local_uot', side_effect=AssertionError('must use saved pair cache')):
                final = resume_experiment(root / 'S', root / 'G', holdouts=(2,), device='cpu')
            self.assertTrue((final.action == 'skip').all())
            self.assertEqual([c[0] for c in calls], ['zero', 'shuffle'])
            self.assertTrue(all(c[1] == 2 for c in calls))  # 总计2轮，而不是再加2轮。
            self.assertTrue(calls[0][2].endswith('last.ckpt'))
            self.assertIsNone(calls[1][2])
            self.assertEqual(restored, [True])
            self.assertEqual(before, {p: file_sha256(p) for p in before})
            checkpoint = torch.load(fold / 'zero/checkpoints/last.ckpt', map_location='cpu', weights_only=False)
            self.assertEqual(checkpoint['epoch'], 1)
            self.assertEqual(checkpoint['global_step'], interrupted['global_step'] * 2)
            with patch.object(DeepSpatial, 'build_model', side_effect=AssertionError('all completed groups must skip')):
                again = resume_experiment(root / 'S', root / 'G', holdouts=(2,), device='cpu')
            self.assertTrue((again.action == 'skip').all())
            # 只丢失生成侧文件时不能再次训练。
            (fold / 'zero/sampling_budget.csv').unlink()
            before_checkpoint = file_sha256(fold / 'zero/checkpoints/last.ckpt')
            with patch.object(DeepSpatial, 'fit', side_effect=AssertionError('generation repair must not train')):
                repaired = resume_experiment(root / 'S', root / 'G', holdouts=(2,), modes=('zero',), device='cpu')
            self.assertEqual(repaired.action.iloc[0], 'skip')
            self.assertEqual(file_sha256(fold / 'zero/checkpoints/last.ckpt'), before_checkpoint)


if __name__ == '__main__':
    unittest.main(verbosity=2)
