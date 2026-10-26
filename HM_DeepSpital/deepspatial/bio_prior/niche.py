"""只用可见细胞构造邻域；表达标准化参数仅由训练切片拟合。"""

import hashlib

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree


def aggregate_neighbors(coords, labels, categories, scores, radius, sigma):
    """按物理半径聚合类型、程序和支持量，始终排除目标细胞自身。"""
    coords = np.asarray(coords, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=str)
    if radius <= 0 or sigma <= 0 or not np.isfinite([radius, sigma]).all():
        raise ValueError("radius and sigma must be finite and positive.")
    if coords.ndim != 2 or coords.shape[1] not in (2, 3) or not np.isfinite(coords).all():
        raise ValueError("Coordinates must be a finite N x 2 or N x 3 array.")
    if len(labels) != len(coords) or scores.ndim != 2 or len(scores) != len(coords):
        raise ValueError("Coordinates, labels and program scores must have matching rows.")
    if not np.isfinite(scores).all() or not np.isin(labels, categories).all():
        raise ValueError("Program scores must be finite and cell types must be known.")
    tree = cKDTree(coords)
    edges = tree.sparse_distance_matrix(tree, radius, output_type="coo_matrix")
    keep = edges.row != edges.col
    weights = np.exp(-0.5 * (edges.data[keep] / sigma) ** 2)
    graph = sparse.csr_matrix(
        (weights, (edges.row[keep], edges.col[keep])), shape=(len(coords), len(coords))
    )
    mass = np.asarray(graph.sum(axis=1)).ravel()
    squared_mass = np.asarray(graph.power(2).sum(axis=1)).ravel()
    inverse_mass = np.divide(1.0, mass, out=np.zeros_like(mass), where=mass > 0)
    onehot = (labels[:, None] == np.asarray(categories)[None, :]).astype(float)
    composition = (graph @ onehot) * inverse_mass[:, None]
    programs = (graph @ scores) * inverse_mass[:, None]
    effective = np.divide(mass ** 2, squared_mass, out=np.zeros_like(mass), where=squared_mass > 0)
    support = np.column_stack([np.log1p(mass), np.log1p(effective), mass == 0])
    return np.column_stack([composition, programs, support]).astype(np.float32)


class NicheFeaturizer:
    """预计算切片内生态位，并保存可重用的训练基因统计量。"""

    def __init__(self, radius, sigma=None, programs=None, expression_transform="log1p",
                 mode="real", shuffle_seed=0):
        """设置物理尺度和明确的程序基因权重；默认不猜测基因程序。"""
        self.radius = float(radius)
        self.sigma = float(sigma if sigma is not None else radius / 2)
        if self.radius <= 0 or self.sigma <= 0 or not np.isfinite([self.radius, self.sigma]).all():
            raise ValueError("Niche radius and sigma must be finite and positive.")
        if expression_transform not in ("log1p", "identity") or mode not in ("real", "shuffle", "zero"):
            raise ValueError("Unknown expression transform or niche control mode.")
        self.programs = {}
        for name, genes in (programs or {}).items():
            weights = genes if isinstance(genes, dict) else {gene: 1.0 for gene in genes}
            weights = {str(gene): float(weight) for gene, weight in weights.items()}
            if not weights or not np.isfinite(list(weights.values())).all() or sum(map(abs, weights.values())) == 0:
                raise ValueError(f"Program {name!r} needs finite, nonzero gene weights.")
            self.programs[str(name)] = weights
        self.expression_transform = expression_transform
        self.mode = mode
        self.shuffle_seed = int(shuffle_seed)
        self.categories = None

    def _expression(self, adata):
        """提取程序涉及的基因；缺失基因报错，避免静默替换程序定义。"""
        missing = sorted(set(self.program_genes) - set(adata.var_names))
        if missing:
            raise ValueError(f"Program genes missing from the measured panel: {missing}")
        matrix = adata[:, self.program_genes].X
        matrix = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
        matrix = matrix.astype(np.float64)
        if not np.isfinite(matrix).all():
            raise ValueError("Program expression must be finite.")
        if self.expression_transform == "log1p":
            if (matrix < 0).any():
                raise ValueError("log1p programs require nonnegative input expression.")
            matrix = np.log1p(matrix)
        return matrix

    def fit(self, adatas, label_key):
        """仅从传入的训练切片拟合类别顺序及程序基因均值、尺度。"""
        self.categories = sorted({str(c) for adata in adatas for c in adata.obs[label_key]})
        self.program_genes = sorted({gene for genes in self.programs.values() for gene in genes})
        expression = np.concatenate([self._expression(adata) for adata in adatas], axis=0)
        self.gene_mean = expression.mean(axis=0).tolist()
        self.gene_scale = np.maximum(expression.std(axis=0), 1e-6).tolist()
        self.feature_names = (
            [f"type::{c}" for c in self.categories]
            + [f"program::{name}" for name in self.programs]
            + ["log_neighbor_mass", "log_effective_neighbors", "missing_neighbors"]
        )
        return self

    def program_scores(self, adata):
        """用训练阶段固定的变换和带符号权重计算细胞程序分数。"""
        if self.categories is None:
            raise ValueError("Fit the niche featurizer before scoring new cells.")
        expression = (self._expression(adata) - self.gene_mean) / self.gene_scale
        scores = np.empty((adata.n_obs, len(self.programs)), dtype=np.float32)
        for column, weights in enumerate(self.programs.values()):
            indices = [self.program_genes.index(gene) for gene in weights]
            values = np.asarray(list(weights.values()), dtype=float)
            scores[:, column] = expression[:, indices] @ values / np.abs(values).sum()
        return scores

    def transform(self, adata, spatial_key, label_key):
        """在单张可见切片上构图；推理时沿用训练统计量，不重新拟合。"""
        labels = adata.obs[label_key].astype(str).to_numpy()
        features = aggregate_neighbors(
            np.asarray(adata.obsm[spatial_key])[:, :2], labels, self.categories,
            self.program_scores(adata), self.radius, self.sigma,
        )
        if self.mode == "zero":
            return np.zeros_like(features)
        if self.mode == "shuffle":
            # 在同类型、同切片内打乱，且不消耗训练/采样的全局随机数。
            identity = "\0".join(map(str, adata.obs_names)).encode("utf-8")
            seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], "little")
            rng = np.random.default_rng(seed ^ self.shuffle_seed)
            for label in self.categories:
                rows = np.flatnonzero(labels == label)
                features[rows] = features[rng.permutation(rows)]
        return features

    def to_dict(self):
        """导出纯 JSON 元数据，供模型保存和独立推理复用。"""
        return {name: getattr(self, name) for name in (
            "radius", "sigma", "programs", "expression_transform", "mode", "shuffle_seed",
            "categories", "program_genes", "gene_mean", "gene_scale", "feature_names",
        )}

    @classmethod
    def from_dict(cls, state):
        """恢复已拟合特征器，不读取测试表达来更新训练参数。"""
        featurizer = cls(**{key: state[key] for key in (
            "radius", "sigma", "programs", "expression_transform", "mode", "shuffle_seed",
        )})
        for key in ("categories", "program_genes", "gene_mean", "gene_scale", "feature_names"):
            setattr(featurizer, key, state[key])
        return featurizer
