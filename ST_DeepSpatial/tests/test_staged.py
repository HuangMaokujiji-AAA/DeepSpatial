"""Engineering checks; synthetic data do not establish biological improvements."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import anndata as ad
import numpy as np
import pandas as pd
import torch

from stspatial.data import prepare, neighborhood, spatial_inputs, endpoints
from stspatial.experiment import run, training_data, stage_losses, generate, load_models
from stspatial.models import SpatialDensity, PositionType, ConditionalExpression, fit, seed_all

torch.set_num_threads(1)


def fixture(root):
    cfg = json.loads((PROJECT/'configs/merfish.json').read_text(encoding='utf-8'))
    cfg['input_glob'] = str(root/'merfish_[0-4].h5ad')
    cfg['training'].update(width=16, batch_size=32, epochs=dict(spatial=1, type=1, expression=1))
    cfg['git'].update(hidden_size=32, depth=1, num_heads=4, patch_size=2)
    cfg['spatial']['bin_um'] = 20.
    rng = np.random.default_rng(123)
    for i in range(5):
        xy = rng.normal(0, 8, (32, 2))+np.array([50+i*2, 60])
        xy[-4:] += 50
        y = ['common']*28+['rare']*4
        g = rng.poisson(2, (32, 4)).astype(np.float32); g[-4:, 0] += 6
        a = ad.AnnData(X=g, obs=pd.DataFrame(dict(cell_class=y), index=[f'{i}_{j}' for j in range(32)]),
                       var=pd.DataFrame(index=['g0', 'g1', 'g2', 'g3']), obsm={'spatial':xy})
        a.write_h5ad(root/f'merfish_{i}.h5ad')
    return cfg, [root/f'merfish_{i}.h5ad' for i in range(5)]


class StagedTests(unittest.TestCase):
    def test_hidden_slice_and_context_exclusion(self):
        with tempfile.TemporaryDirectory(dir=PROJECT) as tmp:
            root = Path(tmp); cfg, paths = fixture(root)
            opened = []; original = ad.read_h5ad
            def reader(path, *args, **kwargs):
                opened.append(Path(path).name)
                return original(path, *args, **kwargs)
            with patch('anndata.read_h5ad', side_effect=reader):
                slices, state = prepare(paths, cfg, 1, excluded=[3])
                training_data(slices, state, cfg)
            self.assertEqual(set(opened), {'merfish_0.h5ad', 'merfish_2.h5ad', 'merfish_4.h5ad'})
            self.assertEqual(state['train_indices'], [0, 2, 4])
            target = slices[1]
            reference = neighborhood(target['physical'], target['z_um'], slices, state, cfg)
            modified = deepcopy(slices)
            modified[1]['g'] *= 1000
            modified[1]['y'] = 1-modified[1]['y']
            actual = neighborhood(target['physical'], target['z_um'], modified, state, cfg)
            for key in ('features', 'gene_features', 'prior', 'mean', 'positive'):
                np.testing.assert_array_equal(reference[key], actual[key])

    def test_stage_dependencies_and_expression_mean(self):
        with tempfile.TemporaryDirectory(dir=PROJECT) as tmp:
            cfg, paths = fixture(Path(tmp)); slices, state = prepare(paths, cfg, 2)
            left, right = endpoints(slices, 100.)
            f, baseline = spatial_inputs(left, right, 100., state, cfg)
            altered = deepcopy(left); altered['g'] *= 1000; altered['y'] = 1-altered['y']
            af, ab = spatial_inputs(altered, right, 100., state, cfg)
            np.testing.assert_array_equal(f, af); np.testing.assert_array_equal(baseline, ab)
            density = SpatialDensity(cfg)
            rate, _ = density(torch.tensor(f), torch.tensor(baseline))
            np.testing.assert_allclose(rate.detach().numpy(), baseline)
            xy = left['physical'][:8]
            ctx = neighborhood(xy, 100., slices, state, cfg)
            changed = deepcopy(slices)
            for s in changed:
                s['g'] *= 1000
            ctx2 = neighborhood(xy, 100., changed, state, cfg)
            np.testing.assert_array_equal(ctx['features'], ctx2['features'])
            np.testing.assert_array_equal(ctx['prior'], ctx2['prior'])
            expression = ConditionalExpression(state, cfg)
            types = np.eye(2, dtype=np.float32)[np.arange(8) % 2]
            args = [torch.tensor(a) for a in (ctx['features'], types, ctx['gene_features'], ctx['mean'], ctx['positive'])]
            expected, logits, magnitude = expression(*args)
            np.testing.assert_allclose(expected.detach().numpy(), ctx['mean'], rtol=2e-5, atol=1e-5)
            self.assertTrue(torch.isfinite(expected).all() and (expected >= 0).all())

    def test_pipeline_saved_interventions_and_resume(self):
        with tempfile.TemporaryDirectory(dir=PROJECT) as tmp:
            root = Path(tmp); cfg, _ = fixture(root); output = root/'out'
            variants = ['full', 'spatial_linear', 'type_prior', 'expression_no_type', 'expression_no_niche']
            rows = run(cfg, output, [2], [9], variants, device='cpu', interventions=True)
            self.assertEqual(len(rows), 8)
            folder = output/'holdout_2/seed_9'
            full = ad.read_h5ad(folder/'full/predicted_holdout.h5ad')
            self.assertGreater(full.n_obs, 0)
            self.assertTrue(np.isfinite(full.X).all() and (full.X >= 0).all())
            np.testing.assert_allclose(full.obsm['cell_type_probabilities'].sum(1), 1., atol=1e-6)
            for variant in ('expression_type_shuffle', 'expression_niche_shuffle'):
                current = ad.read_h5ad(folder/variant/'predicted_holdout.h5ad')
                np.testing.assert_array_equal(current.obsm['spatial_3d'], full.obsm['spatial_3d'])
                np.testing.assert_array_equal(current.obs.cell_class.to_numpy(), full.obs.cell_class.to_numpy())
                np.testing.assert_array_equal(current.obsm['cell_type_probabilities'], full.obsm['cell_type_probabilities'])
            prior = ad.read_h5ad(folder/'type_prior/predicted_holdout.h5ad')
            np.testing.assert_array_equal(prior.obsm['spatial_3d'], full.obsm['spatial_3d'])
            self.assertTrue((folder/'oracle_position_expression_interventions.csv').exists())
            before = full.X.copy()
            run(cfg, output, [2], [9], variants, device='cpu', resume=True, interventions=True)
            after = ad.read_h5ad(folder/'full/predicted_holdout.h5ad')
            np.testing.assert_array_equal(before, after.X)
            changed = deepcopy(cfg); changed['training']['lr'] *= 2
            with self.assertRaises(ValueError):
                run(changed, output, [2], [9], variants, device='cpu', resume=True, interventions=True)

    def test_train_generate_does_not_open_truth(self):
        with tempfile.TemporaryDirectory(dir=PROJECT) as tmp:
            root = Path(tmp); cfg, _ = fixture(root); original = ad.read_h5ad
            def reader(path, *args, **kwargs):
                if Path(path).name == 'merfish_2.h5ad':
                    raise AssertionError('Training/generation opened hidden truth.')
                return original(path, *args, **kwargs)
            with patch('anndata.read_h5ad', side_effect=reader):
                run(cfg, root/'out', [2], [7], stage='train', device='cpu')
                run(cfg, root/'out', [2], [7], stage='generate', device='cpu', resume=True)

    def test_interrupted_epoch_resume_matches(self):
        with tempfile.TemporaryDirectory(dir=PROJECT) as tmp:
            root = Path(tmp); cfg, paths = fixture(root)
            cfg['training']['epochs']['spatial'] = 3
            slices, state = prepare(paths, cfg, 2)
            arrays, _, _, _, _, _ = training_data(slices, state, cfg)
            loss_fn = stage_losses(state, cfg, 'full')[0]
            seed_all(5); initial = deepcopy(SpatialDensity(cfg).state_dict())
            whole = SpatialDensity(cfg); whole.load_state_dict(initial)
            fit(whole, arrays, loss_fn, cfg, root/'whole', 'spatial', 6, torch.device('cpu'))
            batches_per_epoch = int(np.ceil(len(arrays[0])/cfg['training']['batch_size']))
            calls = 0
            def interrupted(model, *batch):
                nonlocal calls
                calls += 1
                if calls > batches_per_epoch:
                    raise RuntimeError('simulated interruption')
                return loss_fn(model, *batch)
            partial = SpatialDensity(cfg); partial.load_state_dict(initial)
            with self.assertRaisesRegex(RuntimeError, 'simulated'):
                fit(partial, arrays, interrupted, cfg, root/'partial', 'spatial', 6, torch.device('cpu'))
            resumed = SpatialDensity(cfg)
            fit(resumed, arrays, loss_fn, cfg, root/'partial', 'spatial', 6, torch.device('cpu'), resume=True)
            for key, value in whole.state_dict().items():
                torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
