"""Two-layer row-normalized GCN, with one-way inductive query inference.

Only expression and visible slice IDs enter this module. No query-to-query
edges, spatial coordinates or labels are used. Native gathers avoid a PyG
dependency and CUDA sparse/scatter nondeterminism.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch

from .representation import ExpressionAE


def training_neighbors(x, slice_ids, k, query_chunk=512):
    """Directed within-slice expression-cosine kNN, with one explicit self loop."""
    ids = np.empty((len(x), k+1), dtype="int64")
    for label in np.unique(slice_ids):
        rows = np.flatnonzero(slice_ids == label)
        if len(rows) <= k:
            raise ValueError(f"Slice {label}: need more than graph_k={k} visible cells.")
        search = NearestNeighbors(n_neighbors=k+1, metric="cosine", algorithm="brute", n_jobs=1).fit(x[rows])
        for start in range(0, len(rows), query_chunk):
            local = np.arange(start, min(start+query_chunk, len(rows)))
            candidates = search.kneighbors(x[rows[local]], return_distance=False)
            for query, neighbors in zip(local, candidates):
                others = neighbors[neighbors != query][:k]
                ids[rows[query]] = np.r_[rows[query], rows[others]]
    return ids


def aggregate_features(x, neighbors, chunk=512):
    return np.concatenate([x[neighbors[i:i+chunk]].mean(1)
                           for i in range(0, len(x), chunk)]).astype("float32")


class InductiveGCNAE(ExpressionAE):
    """z = S ReLU(S X W0 + b0) W1 + b1; decoder matches the MLP AE.

    S is a directed, row-normalized kNN+self matrix. For unseen cells, the
    first layer averages query expression with visible reference neighbors;
    the second averages the query hidden state with frozen reference hidden
    states. Query cells never update the reference graph or each other.
    """
    def encode(self, x):
        raise ValueError("Use encode_training or encode_queries for a graph encoder.")

    def encode_neighborhood(self, neighbor_inputs):
        hidden = self.encoder[1](self.encoder[0](neighbor_inputs))
        return self.encoder[2](hidden.mean(1))

    @torch.no_grad()
    def encode_training(self, x, neighbors, device, batch_size=512):
        self.eval()
        inputs = torch.as_tensor(aggregate_features(x, neighbors), device=device)
        return np.concatenate([self.encode_neighborhood(inputs[torch.as_tensor(
            neighbors[i:i+batch_size], device=device)]).cpu().numpy()
            for i in range(0, len(x), batch_size)])

    def bind_reference(self, x, neighbors):
        self.reference_x = np.array(x, dtype="float32", copy=True)
        self.reference_inputs = aggregate_features(self.reference_x, neighbors)
        self.reference_k = neighbors.shape[1]-1
        self.reference_search = NearestNeighbors(
            n_neighbors=self.reference_k, metric="cosine", algorithm="brute", n_jobs=1).fit(self.reference_x)

    @torch.no_grad()
    def encode_queries(self, x, device, batch_size=512):
        self.eval()
        ref_h = np.concatenate([self.encoder[1](self.encoder[0](torch.as_tensor(
            self.reference_inputs[i:i+batch_size], device=device))).cpu().numpy()
            for i in range(0, len(self.reference_inputs), batch_size)])
        result = []
        for i in range(0, len(x), batch_size):
            query = x[i:i+batch_size]
            indices = self.reference_search.kneighbors(query, return_distance=False)
            first_input = (query + self.reference_x[indices].sum(1))/(self.reference_k+1)
            query_h = self.encoder[1](self.encoder[0](torch.as_tensor(first_input, device=device)))
            ref_sum = torch.as_tensor(ref_h[indices].sum(1), device=device)
            result.append(self.encoder[2]((query_h+ref_sum)/(self.reference_k+1)).cpu().numpy())
        return np.concatenate(result)
