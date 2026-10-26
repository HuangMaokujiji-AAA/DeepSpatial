"""最小回归检查：条件来源、抽样预算、训练/推理、保存恢复和表达细化。"""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT))
if (PROJECT / '.runtime_deps').is_dir():
    sys.path.insert(1, str(PROJECT / '.runtime_deps'))

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import torch

from deepspatial import DeepSpatial
from deepspatial.bio_prior import NicheFeaturizer, aggregate_neighbors, refine_program_expression
from deepspatial.data_utils.sampling import sample_coupling_indices, stratified_source_indices
from deepspatial.module import DeepSpatialModule

torch.set_num_threads(1)


def slices():
    """构造含常见类和少数类的两张小切片，保持物理坐标及基因顺序明确。"""
    result = []
    for k in range(2):
        result.append(ad.AnnData(
            X=np.arange(1, 25, dtype=np.float32).reshape(6, 4) + k,
            obs=pd.DataFrame({'cell_class': ['common'] * 4 + ['rare'] * 2, 'z_coord': k * 50.0},
                             index=[f'cell_{k}_{i}' for i in range(6)]),
            var=pd.DataFrame(index=['Mbp', 'Opalin', 'Ermn', 'Gjc3']),
            obsm={'spatial': np.array([[0, 0], [3, 0], [6, 0], [30, 0], [0, 4], [30, 4]],
                                     dtype=np.float32) + k},
        ))
    return result


def small_model(niche=True):
    """建立工程验证用的小模型；不作为论文性能配置。"""
    model = DeepSpatial()
    model.setup_data(slices(), batch_size=32, num_workers=0, n_samples_base=64,
                     balanced_fraction=0.25 if niche else 0,
                     source_sampling='stratified' if niche else 'random',
                     niche_config={'radius': 10, 'programs': {'myelin': ['Mbp', 'Opalin']}} if niche else None)
    model.build_model(hidden_size=32, depth=1, num_heads=4, patch_size=2,
                      sampling_method='euler', bidirectional_training=niche)
    return model


class CaptureVelocity(torch.nn.Module):
    """记录实际网络收到的条件，检查端点交换不会错用另一端邻域。"""

    def __init__(self):
        """保留一个参数，使此检查模型可以参与标准反向传播。"""
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))
        self.last = None

    def forward(self, **kwargs):
        """保存当前输入并返回形状相同的零速度。"""
        self.last = kwargs
        return tuple(torch.zeros_like(kwargs[key]) + self.bias for key in ('xt', 'gt', 'ct'))


class PipelineTests(unittest.TestCase):
    """覆盖新增计算路径以及关闭开关后的上游数值一致性。"""

    def test_neighbors_exclude_self_and_mark_missing(self):
        """自己的高表达不能进入自己的邻域，孤立细胞使用显式缺失标志。"""
        features = aggregate_neighbors([[0, 0], [1, 0], [100, 0]], ['a', 'b', 'a'],
                                       ['a', 'b'], [[1000], [2], [3]], 2, 1)
        np.testing.assert_allclose(features[0, :3], [0, 1, 2])
        np.testing.assert_allclose(features[1, :3], [1, 0, 1000])
        np.testing.assert_array_equal(features[2], [0, 0, 0, 0, 0, 1])

    def test_sampling_budget_and_valid_same_type_blocks(self):
        """低方差采样保留来源比例，训练平衡项只进入有支持的同类子块。"""
        np.random.seed(11)
        labels = np.array([0] * 99 + [1])
        sampled = stratified_source_indices(labels, 1000)
        np.testing.assert_array_equal(np.bincount(sampled, minlength=100), np.full(100, 10))
        coupling = np.array([[0.4, 0.1, 0.0], [0.1, 0.39, 0.0], [0.0, 0.0, 0.01]])
        source, target, stats = sample_coupling_indices(coupling, [0, 0, 1], [0, 0, 1], 1000, 1)
        types = np.array([0, 0, 1])
        np.testing.assert_array_equal(types[source], types[target])
        self.assertTrue(400 < (types[source] == 1).sum() < 600)
        self.assertEqual(stats['balanced_pairs'], 1000)

    def test_training_statistics_are_frozen(self):
        """新切片及序列化恢复均使用训练期的基因尺度。"""
        feature = NicheFeaturizer(10, programs={'p': ['Mbp']}).fit(slices(), 'cell_class')
        state = json.loads(json.dumps(feature.to_dict()))
        new = slices()[0]
        new.X *= 100
        recovered = NicheFeaturizer.from_dict(state)
        np.testing.assert_allclose(feature.transform(new, 'spatial', 'cell_class'),
                                   recovered.transform(new, 'spatial', 'cell_class'))
        self.assertEqual(feature.to_dict(), state)
        with self.assertRaisesRegex(ValueError, 'missing'):
            NicheFeaturizer(10, programs={'p': ['AbsentGene']}).fit(slices(), 'cell_class')

    def test_bidirectional_training_uses_actual_source_condition(self):
        """反向样本的深度符号与 h1 一起交换；正向样本继续使用 h0。"""
        model = small_model()
        batch = {key: value[:32] for key, value in model.dataset.tensors.items()}
        batch['h0'] = torch.full_like(batch['h0'], 1)
        batch['h1'] = torch.full_like(batch['h1'], 9)
        capture = CaptureVelocity()
        module = DeepSpatialModule({'path_type': 'Linear', 'prediction': 'velocity',
                                    'bidirectional_training': True}, capture)
        torch.manual_seed(2)
        loss = module._shared_step(batch)['loss']
        reverse = capture.last['delta_z'].flatten() < 0
        self.assertTrue(reverse.any() and (~reverse).any())
        np.testing.assert_array_equal(capture.last['niche_cond'][reverse].detach().numpy(), batch['h1'][reverse])
        np.testing.assert_array_equal(capture.last['niche_cond'][~reverse].detach().numpy(), batch['h0'][~reverse])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_fit_ode_save_reload_and_h5ad_trace(self):
        """真实执行短训练与 ODE，核查固定条件、结果追溯及独立加载。"""
        torch.manual_seed(7)
        np.random.seed(7)
        model = small_model()
        with tempfile.TemporaryDirectory(dir=PROJECT) as temp:
            model.fit(max_epochs=1, accelerator='cpu', devices=1, save_ckpt=True, save_dir=temp)
            self.assertTrue((Path(temp) / 'last.ckpt').is_file())
            self.assertTrue(any((Path(temp) / 'logs').rglob('metrics.csv')))
            sample = {key: value[:3] for key, value in model.dataset.tensors.items()}
            sample['niche_cond'] = sample['h0']
            received = []
            hook = model.module.ema_model.niche_embedder.register_forward_pre_hook(
                lambda module, inputs: received.append(inputs[0].detach().clone()))
            model.module.sample(sample, steps=4)
            hook.remove()
            self.assertGreater(len(received), 1)
            for condition in received:
                torch.testing.assert_close(condition, sample['h0'], rtol=0, atol=0)
            model.save_model(Path(temp) / 'saved')
            restored = DeepSpatial()
            restored.load_checkpoint(str(Path(temp) / 'saved/model.ckpt'))
            self.assertEqual(restored.train_config['sampling_method'], 'euler')
            self.assertEqual(restored.niche_featurizer.to_dict(), model.niche_featurizer.to_dict())
            torch.manual_seed(12)
            np.random.seed(12)
            output = restored.reconstruct_full_volume(slices(), thickness=10, steps=4,
                                                      chunk_size=3, device='cpu', save_intermediates=True)
            self.assertEqual(output.n_obs, 30)
            self.assertEqual(output.obsm['spatial_3d'].shape, (30, 3))
            self.assertEqual(output.layers['expression_before_sparsification'].shape, (30, 4))
            self.assertTrue(np.isfinite(output.X.toarray()).all())
            self.assertTrue(output.obs['parent_id'].str.contains(':').all())
            self.assertFalse(output.obs['is_observed'].any())
            self.assertEqual(set(output.obs['source_direction']), {'forward', 'backward'})
            output.write_h5ad(Path(temp) / 'generated.h5ad')
            readback = ad.read_h5ad(Path(temp) / 'generated.h5ad')
            self.assertEqual(readback.obs['parent_id'].tolist(), output.obs['parent_id'].tolist())
            wrong_order = [adata[:, ::-1].copy() for adata in slices()]
            with self.assertRaisesRegex(ValueError, 'gene names'):
                restored.reconstruct_full_volume(wrong_order, thickness=10, steps=4)

    def test_disabled_switches_match_upstream(self):
        """开关关闭时，原 UOT 抽样、模型参数和 ODE 数值路径与上游一致。"""
        location = PROJECT.parent / 'deepspatial/__init__.py'
        spec = importlib.util.spec_from_file_location('_upstream_deepspatial', location,
                                                     submodule_search_locations=[str(location.parent)])
        original = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = original
        spec.loader.exec_module(original)
        pairs = []
        for implementation in (original.DeepSpatial, DeepSpatial):
            torch.manual_seed(13)
            np.random.seed(13)
            model = implementation()
            visible = slices()
            for adata in visible:
                adata.obsm['spatial'] = adata.obsm['spatial'].astype(np.float64) + 0.123456789
            model.setup_data(visible, num_workers=0, batch_size=32, n_samples_base=64)
            model.build_model(hidden_size=32, depth=1, num_heads=4, patch_size=2, sampling_method='euler')
            pairs.append(model)
        for name, state in pairs[0].module.state_dict().items():
            torch.testing.assert_close(state, pairs[1].module.state_dict()[name], rtol=0, atol=0)
        for name, tensor in pairs[0].dataset.tensors.items():
            torch.testing.assert_close(tensor, pairs[1].dataset.tensors[name], rtol=0, atol=0)
        results = []
        for model in pairs:
            visible = slices()
            for adata in visible:
                adata.obsm['spatial'] = adata.obsm['spatial'].astype(np.float64) + 0.123456789
            if model is pairs[0]:
                model._normalize_spatial(visible)
            torch.manual_seed(21)
            np.random.seed(21)
            results.append(model.reconstruct_full_volume(visible, thickness=10, steps=4, device='cpu'))
        np.testing.assert_allclose(results[0].X.toarray(), results[1].X.toarray(), rtol=0, atol=0)
        np.testing.assert_allclose(results[0].obsm['spatial'], results[1].obsm['spatial'], rtol=0, atol=0)
        self.assertEqual(results[0].obs['cell_class'].tolist(), results[1].obs['cell_class'].tolist())

    def test_refinement_changes_only_generated_program_genes(self):
        """提供明确区间后，仅修正生成细胞的程序基因，观测与坐标不变。"""
        training = slices()
        feature = NicheFeaturizer(10, programs={'p': ['Mbp']}).fit(training, 'cell_class')
        candidate = training[0].copy()
        candidate.X = sparse.csr_matrix(candidate.X)
        candidate.obs['is_observed'] = [True] + [False] * 5
        before = feature.program_scores(candidate)
        lower, upper = before + 0.5, before + 1.0
        corrected = refine_program_expression(candidate, feature, lower, upper, steps=20,
                                               learning_rate=0.1, anchor_weight=0.01)
        after = feature.program_scores(corrected)
        self.assertTrue(np.all(after[1:] > before[1:]))
        np.testing.assert_array_equal(candidate.X.toarray()[0], corrected.X.toarray()[0])
        np.testing.assert_array_equal(candidate.X.toarray()[:, 1:], corrected.X.toarray()[:, 1:])
        np.testing.assert_array_equal(candidate.obsm['spatial'], corrected.obsm['spatial'])
        self.assertEqual(candidate.obs['cell_class'].tolist(), corrected.obs['cell_class'].tolist())
        unchanged = refine_program_expression(candidate, feature, before - 1, before + 1, steps=2)
        np.testing.assert_array_equal(unchanged.X.toarray(), candidate.X.toarray())


if __name__ == '__main__':
    unittest.main(verbosity=2)
