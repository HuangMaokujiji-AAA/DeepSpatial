"""V2 的表达细化算子：接收外部已冻结的逐细胞程序区间，不自动学习先验。"""

import numpy as np
from scipy import sparse
import torch


def refine_program_expression(adata, featurizer, lower, upper, *, steps=100,
                              learning_rate=0.01, band_weight=1.0, anchor_weight=1.0,
                              chunk_size=2048, device='cpu'):
    """固定坐标/类型，在给定区间约束下仅调整虚拟细胞的程序基因表达。"""
    if not featurizer.programs:
        raise ValueError('Expression refinement requires explicitly defined programs.')
    lower, upper = np.asarray(lower, dtype=np.float32), np.asarray(upper, dtype=np.float32)
    expected = (adata.n_obs, len(featurizer.programs))
    if lower.shape != expected or upper.shape != expected or not np.isfinite([lower, upper]).all() or (lower > upper).any():
        raise ValueError('Finite lower/upper bounds must match cells x programs, with lower <= upper.')
    if steps < 1 or chunk_size < 1 or learning_rate <= 0 or band_weight < 0 or anchor_weight <= 0:
        raise ValueError('Use positive steps/chunk_size/rate/anchor weight and nonnegative band weight.')
    genes = list(featurizer.program_genes)
    if not set(genes).issubset(adata.var_names):
        raise ValueError('Generated expression is missing program genes.')
    indices = adata.var_names.get_indexer(genes)
    values = adata[:, genes].X
    values = values.toarray() if sparse.issparse(values) else np.asarray(values)
    if not np.isfinite(values).all() or (featurizer.expression_transform == 'log1p' and (values < 0).any()):
        raise ValueError('Refinement input must be finite and nonnegative for log1p scores.')
    if 'is_observed' not in adata.obs:
        raise ValueError('Mark is_observed explicitly; refinement must not alter real observations.')
    generated = ~adata.obs['is_observed'].to_numpy(dtype=bool)
    weights = np.zeros((len(genes), len(featurizer.programs)), dtype=np.float32)
    for column, program in enumerate(featurizer.programs.values()):
        total_weight = sum(map(abs, program.values()))
        for gene, weight in program.items():
            weights[genes.index(gene), column] = weight / total_weight
    weight_tensor = torch.as_tensor(weights, device=device)
    mean = torch.as_tensor(featurizer.gene_mean, dtype=torch.float32, device=device)
    scale = torch.as_tensor(featurizer.gene_scale, dtype=torch.float32, device=device)
    adjusted = np.asarray(values, dtype=np.float32).copy()
    rows = np.flatnonzero(generated)
    for start in range(0, len(rows), chunk_size):
        selected = rows[start:start + chunk_size]
        base = torch.as_tensor(adjusted[selected], device=device)
        expression = base.clone().requires_grad_(True)
        lo, hi = torch.as_tensor(lower[selected], device=device), torch.as_tensor(upper[selected], device=device)
        optimizer = torch.optim.Adam([expression], lr=learning_rate)
        for _ in range(steps):
            transformed = torch.log1p(expression) if featurizer.expression_transform == 'log1p' else expression
            score = ((transformed - mean) / scale) @ weight_tensor
            band_loss = ((lo - score).relu().square() + (score - hi).relu().square()).mean()
            # 用原始生成表达作锚点；区间内没有额外向均值收缩的梯度。
            loss = band_weight * band_loss + anchor_weight * ((expression - base) / base.abs().clamp_min(1)).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if featurizer.expression_transform == 'log1p':
                with torch.no_grad():
                    expression.clamp_(min=0)
        adjusted[selected] = expression.detach().cpu().numpy()
    result = adata.copy()
    if sparse.issparse(result.X):
        matrix = result.X.astype(np.result_type(result.X.dtype, np.float32)).tolil()
        matrix[np.ix_(rows, indices)] = adjusted[rows]
        result.X = matrix.tocsr()
    else:
        result.X = np.asarray(result.X, dtype=np.result_type(result.X.dtype, np.float32)).copy()
        result.X[np.ix_(rows, indices)] = adjusted[rows]
    result.uns['hm_refinement'] = {'steps': int(steps), 'fixed_coordinates_and_types': True,
                                  'interval_source': 'externally_fitted_frozen_prior',
                                  'refined_cells': int(generated.sum())}
    return result
