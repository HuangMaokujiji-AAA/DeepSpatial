"""Check the actual notebook's three-fold control flow without training a model."""
import ast
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


PROJECT = Path(__file__).resolve().parents[1]


class NotebookTests(unittest.TestCase):
    def test_independent_folds_and_existing_output_guard(self):
        notebook = json.loads((PROJECT / 'examples/run_v1.ipynb').read_text(encoding='utf-8'))
        cells = {c['id']: ''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code'}
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                self.assertIsNone(cell['execution_count'])
                self.assertEqual(cell['outputs'], [])
                compile(''.join(cell['source']), cell['id'], 'exec')
        base = json.loads((PROJECT / 'configs/merfish_v1.json').read_text(encoding='utf-8'))
        reads, seeds, models, writes = [], [], [], []

        class Model:
            def __init__(self):
                self.module = SimpleNamespace(trainer=SimpleNamespace(is_global_zero=True))
                models.append(self)

            def setup_data(self, observed, **kwargs):
                self.ids = [a.index for a in observed]
                self.depths = [a.obs[base['data']['z_key']] for a in observed]
                self.data_config = kwargs

            def build_model(self, **kwargs):
                self.model_config = kwargs

            def fit(self, **kwargs):
                self.fit_config = kwargs

            def save_model(self, path):
                self.model_path = path

            def reconstruct_full_volume(self, observed, **kwargs):
                self.reconstruction_config = kwargs
                return SimpleNamespace(n_obs=12, write_h5ad=lambda path: writes.append(path))

        def read(path):
            reads.append(int(path.stem.split('_')[-1]))
            return SimpleNamespace(index=reads[-1], obs={})

        # Reuse the actual CLI output helper, with rank-zero decoration neutralized.
        module = ast.parse((PROJECT / 'examples/run_v1.py').read_text(encoding='utf-8'))
        helper = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == 'prepare_output')
        helper_scope = dict(json=json, rank_zero_only=lambda fn: fn)
        exec(compile(ast.Module(body=[helper], type_ignores=[]), 'run_v1.py:prepare_output', 'exec'), helper_scope)

        with tempfile.TemporaryDirectory(dir=PROJECT) as temp, redirect_stdout(StringIO()):
            project = Path(temp)
            data = project / 'data'
            data.mkdir()
            for i in range(5):
                (data / f'merfish_{i}.h5ad').touch()
            local_config = dict(base, input_glob='data/merfish_[0-4].h5ad')
            config_path = project / 'config.json'
            config_path.write_text(json.dumps(local_config), encoding='utf-8')
            scope = {}
            exec(cells['configuration'], scope)
            self.assertEqual(scope['HOLDOUT_SLICE_IDS'], [1, 2, 3])
            scope.update(PROJECT=project, CONFIG_PATH=config_path, OUTPUT_ROOT=project / 'outputs',
                         ad=SimpleNamespace(read_h5ad=read),
                         pl=SimpleNamespace(seed_everything=lambda seed, workers: seeds.append((seed, workers))),
                         ds=SimpleNamespace(__file__=str(project / 'deepspatial/__init__.py'), DeepSpatial=Model),
                         torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
                         prepare_output=helper_scope['prepare_output'])
            exec(cells['validate-inputs'], scope)
            self.assertEqual(reads, [])
            exec(cells['run-holdouts'], scope)
            exec(cells['summary'], scope)
            expected_ids = [[0, 2, 3, 4], [0, 1, 3, 4], [0, 1, 2, 4]]
            self.assertEqual([m.ids for m in models], expected_ids)
            self.assertEqual(reads, [i for fold in expected_ids for i in fold])
            self.assertEqual(seeds, [(base.get('seed', 42), True)] * 3)
            self.assertEqual([r['heldout_id'] for r in scope['completed_runs']], [1, 2, 3])
            for heldout_id, model, ids, target in zip((1, 2, 3), models, expected_ids, writes):
                output = project / f'outputs/holdout_{heldout_id}'
                saved = json.loads((output / 'run_config.json').read_text(encoding='utf-8'))
                self.assertEqual(saved.pop('run')['heldout_indices'], [heldout_id])
                self.assertEqual(saved, local_config)
                self.assertEqual(model.depths, [base['slice_z_um'][i] for i in ids])
                self.assertEqual(model.data_config, base['data'])
                self.assertEqual(model.model_config, base['model'])
                self.assertEqual(model.fit_config, dict(save_dir=str(output / 'checkpoints'), save_ckpt=True, **base['training']))
                self.assertEqual(model.reconstruction_config, base['reconstruction'])
                self.assertEqual(model.model_path, output / 'model')
                self.assertEqual(target, output / 'generated.h5ad')
            with self.assertRaises(FileExistsError):
                exec(cells['validate-inputs'], scope)
            self.assertEqual(len(models), 3)
            self.assertEqual(scope['config'], local_config)


if __name__ == '__main__':
    unittest.main()
