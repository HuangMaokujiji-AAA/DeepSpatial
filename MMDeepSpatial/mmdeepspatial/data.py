"""Train-only expression preprocessing; held-out files are opened after fitting."""
from dataclasses import dataclass
from pathlib import Path
import glob
import re

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def input_paths(pattern):
    paths = sorted((Path(p).resolve() for p in glob.glob(str(Path(pattern).expanduser()), recursive=True)),
                   key=lambda p: [int(s) if s.isdigit() else s
                                  for s in re.split(r"(\d+)", str(p))])
    if len(paths) < 3 or len(set(paths)) != len(paths):
        raise ValueError("Need at least three distinct H5AD slices; check input_glob.")
    return paths


def read_slice(path, label_key):
    a = ad.read_h5ad(path)
    if not a.n_obs or not a.n_vars or not a.var_names.is_unique or not a.obs_names.is_unique:
        raise ValueError(f"{path}: empty data or duplicate cell/gene IDs.")
    if label_key not in a.obs or a.obs[label_key].isna().any():
        raise ValueError(f"{path}: missing/NA cell-type annotation {label_key!r}.")
    return a


def expression(a, genes, layer, transform):
    missing = pd.Index(genes).difference(a.var_names)
    if len(missing):
        raise ValueError(f"Missing {len(missing)} training-defined genes: {list(missing[:5])}")
    view = a[:, genes]
    values = view.X if layer is None else view.layers[layer]
    x = values.toarray() if sparse.issparse(values) else np.asarray(values)
    x = np.array(x, dtype=np.float32, copy=True)
    if not np.isfinite(x).all():
        raise ValueError("Expression contains NaN/Inf.")
    if transform == "log1p":
        if (x < 0).any():
            raise ValueError("log1p expects nonnegative expression. Use identity for scaled X.")
        np.log1p(x, out=x)
    elif transform != "identity":
        raise ValueError("transform must be log1p or identity.")
    return x


@dataclass
class Preprocessor:
    genes: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    layer: str | None
    transform: str

    @classmethod
    def fit(cls, training, layer=None, transform="log1p", n_genes=0):
        # Gene intersection and variance ranking use training slices exclusively.
        genes = training[0].var_names
        for a in training[1:]:
            genes = genes[genes.isin(a.var_names)]
        if not len(genes):
            raise ValueError("Training slices have no common genes.")
        x = np.concatenate([expression(a, genes, layer, transform) for a in training])
        variance = x.var(axis=0, dtype=np.float64)
        keep = np.flatnonzero(variance > 1e-12)
        if n_genes:
            keep = keep[np.argsort(-variance[keep], kind="stable")[:n_genes]]
        if not len(keep):
            raise ValueError("No variable training genes.")
        x = x[:, keep]
        return cls(np.asarray(genes[keep], dtype=str), x.mean(0, dtype=np.float64).astype("float32"),
                   np.maximum(x.std(0, dtype=np.float64), 1e-6).astype("float32"), layer, transform)

    def apply(self, a):
        return np.ascontiguousarray((expression(a, self.genes, self.layer, self.transform)
                                     - self.mean) / self.scale, dtype=np.float32)

    def save(self, path):
        np.savez_compressed(path, genes=self.genes, mean=self.mean, scale=self.scale,
                            layer="" if self.layer is None else self.layer, transform=self.transform)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as p:
            return cls(p["genes"], p["mean"], p["scale"], str(p["layer"]) or None, str(p["transform"]))


def cell_metadata(adatas, indices, label_key):
    return pd.concat([
        pd.DataFrame({"cell_id": a.obs_names.astype(str), "slice_index": index,
                      "cell_type": a.obs[label_key].astype(str).to_numpy()})
        for index, a in zip(indices, adatas)
    ], ignore_index=True)
