"""Matched-capacity Euclidean, Poincare and product-space autoencoders."""
import random
import time

import geoopt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


class ExpressionAE(nn.Module):
    def __init__(self, n_genes, dim=32, hidden=128, arm="E0", curvature=1.0,
                 max_tangent_radius=1.5):
        super().__init__()
        if arm not in {"E0", "E1", "E2"} or dim < 2 or (arm == "E2" and dim % 2):
            raise ValueError("Use E0/E1/E2, dim>=2, and even dim for E2.")
        if not np.isfinite(curvature) or curvature <= 0:
            raise ValueError("Poincare c must be positive; sectional curvature is -c.")
        if max_tangent_radius is not None and not (0 < max_tangent_radius <= 2.):
            raise ValueError("max_tangent_radius must be in (0, 2], or None for legacy checkpoints.")
        self.arm, self.dim, self.curvature = arm, dim, curvature
        self.max_tangent_radius = max_tangent_radius
        self.e_dim = dim if arm == "E0" else dim // 2 if arm == "E2" else 0
        self.ball = geoopt.PoincareBall(c=curvature, learnable=False)
        # Identical linear layers and parameter initialization in every arm.
        # E2 splits one shared output into E/H heads, rather than doubling capacity.
        self.encoder = nn.Sequential(nn.Linear(n_genes, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        self.decoder = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, n_genes))

    def encode(self, x):
        return self.map_latent(self.encoder(x))

    def map_latent(self, raw):
        if self.arm == "E0":
            return raw
        tangent = raw[:, self.e_dim:]
        if self.max_tangent_radius is not None:
            # Smooth, strictly monotone radial contraction before expmap. With
            # limit=1.5, sqrt(c)*||h|| < tanh(1.5)=0.905, below either hard clip.
            tangent = tangent / (1 + self.curvature * tangent.square().sum(-1, keepdim=True)
                                 / self.max_tangent_radius**2).sqrt()
        h = self.ball.expmap0(tangent)
        # Extra margin beyond geoopt's numeric projection, in float32 as well.
        radius = .98 / self.curvature ** .5
        h = h * (radius / h.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1)
        return torch.cat((raw[:, :self.e_dim], h), dim=-1)

    def forward(self, x):
        z = self.encode(x)
        # Ambient ball coordinates enter the SAME decoder: expmap/logmap do not
        # cancel, and no arm-specific rescaling confounds the curvature comparison.
        return self.decoder(z), z


def fit_ae(x, cfg, arm, seed, device, folder):
    set_seed(seed)
    model = ExpressionAE(x.shape[1], cfg["dim"], cfg["hidden"], arm, cfg["curvature"],
                         cfg["max_tangent_radius"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=cfg["batch_size"],
                        shuffle=True, generator=generator, num_workers=0)
    history = []
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(cfg["epochs"]):
        model.train()
        loss_sum, maximum_grad, max_radius, boundary_cells = 0., 0., 0., 0
        raw_norms, ball_norms = [], []
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            raw = model.encoder(batch)
            z = model.map_latent(raw)
            recon = model.decoder(z)
            loss = (recon - batch).square().mean()
            if not torch.isfinite(loss) or not torch.isfinite(z).all():
                raise FloatingPointError(f"Nonfinite loss/embedding at epoch {epoch+1}.")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch)
            maximum_grad = max(maximum_grad, float(grad))
            if arm != "E0":
                norms = z[:, model.e_dim:].detach().norm(dim=1) * cfg["curvature"] ** .5
                max_radius = max(max_radius, float(norms.max()))
                boundary_cells += int((norms >= .979).sum())
                raw_norms.append((raw[:, model.e_dim:].detach().norm(dim=1)
                                  * cfg["curvature"]**.5).cpu().numpy())
                ball_norms.append(norms.cpu().numpy())
        row = dict(epoch=epoch+1, train_mse=loss_sum/len(x), grad_norm_max=maximum_grad,
                   max_ball_radius_fraction=max_radius, near_boundary_fraction=boundary_cells/len(x),
                   nonfinite_batches=0)
        if raw_norms:
            raw_values, radii = np.concatenate(raw_norms), np.concatenate(ball_norms)
            row.update(raw_tangent_radius_mean=float(raw_values.mean()),
                       raw_tangent_radius_p95=float(np.quantile(raw_values, .95)),
                       raw_tangent_radius_max=float(raw_values.max()),
                       ball_radius_fraction_median=float(np.median(radii)),
                       ball_radius_fraction_p95=float(np.quantile(radii, .95)),
                       ball_radius_fraction_std=float(radii.std()),
                       near_mapping_limit_fraction=float(np.mean(
                           radii >= np.tanh(.95*cfg["max_tangent_radius"]))))
        history.append(row)
        pd.DataFrame(history).to_csv(folder / "training.csv", index=False)
        if epoch == 0 or (epoch+1) % cfg["log_every"] == 0 or epoch+1 == cfg["epochs"]:
            print(f"{arm} seed={seed} epoch={epoch+1}/{cfg['epochs']} mse={row['train_mse']:.5f} "
                  f"boundary={row['near_boundary_fraction']:.3%}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timing = dict(train_seconds=time.perf_counter()-start,
                  peak_train_gpu_mb=torch.cuda.max_memory_allocated(device)/2**20 if device.type == "cuda" else 0,
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "n_genes": x.shape[1], "dim": cfg["dim"], "hidden": cfg["hidden"],
                "arm": arm, "curvature": cfg["curvature"],
                "max_tangent_radius": cfg["max_tangent_radius"]}, folder / "encoder.pt")
    return model, timing


@torch.no_grad()
def encode(model, x, device, batch_size=1024):
    model.eval()
    return np.concatenate([model.encode(torch.as_tensor(x[i:i+batch_size], device=device)).cpu().numpy()
                           for i in range(0, len(x), batch_size)])


def probe_coordinates(z, arm, curvature):
    """Linear probe uses tangent coordinates; retrieval uses true geodesic distances."""
    if arm == "E0":
        return z.copy()
    split = z.shape[1] // 2 if arm == "E2" else 0
    ball = geoopt.PoincareBall(c=curvature)
    with torch.no_grad():
        h = ball.logmap0(torch.as_tensor(z[:, split:], dtype=torch.float64)).numpy()
    return np.concatenate((z[:, :split], h), axis=1).astype("float32")


def pairwise_components(a, b, arm, curvature):
    """Matrices without an N x M x D broadcast; float64 for ball boundary stability."""
    split = a.shape[1] if arm == "E0" else a.shape[1] // 2 if arm == "E2" else 0
    a, b = a.double(), b.double()
    de = torch.cdist(a[:, :split], b[:, :split]) if split else None
    dh = None
    if arm != "E0":
        u, v = a[:, split:], b[:, split:]
        margin_u = (1-curvature*u.square().sum(1)).clamp_min(1e-12)
        margin_v = (1-curvature*v.square().sum(1)).clamp_min(1e-12)
        squared = torch.cdist(u, v, compute_mode="donot_use_mm_for_euclid_dist").square()
        # acosh(1+2t) = 2 asinh(sqrt(t)); gives exact zero for identical points.
        dh = 2 / curvature**.5 * torch.asinh(
            (curvature*squared/(margin_u[:, None]*margin_v[None, :])).clamp_min(0).sqrt())
    return de, dh


class Geometry:
    def __init__(self, arm, curvature, alpha=.5, scales=(1., 1.)):
        self.arm, self.curvature, self.alpha, self.scales = arm, curvature, alpha, tuple(scales)

    def calibrate(self, train_z, slice_ids, seed, device):
        """Robust medians of random cross-slice training pairs, never held-out cells."""
        rng = np.random.default_rng(seed)
        left = rng.integers(len(train_z), size=8192)
        right = rng.integers(len(train_z), size=8192)
        mask = slice_ids[left] != slice_ids[right]
        left, right = left[mask], right[mask]
        if not len(left):
            raise ValueError("Distance calibration requires multiple visible slices.")
        # Small diagonal blocks avoid constructing a large square calibration matrix.
        components = [[], []]
        for start in range(0, len(left), 128):
            a = torch.as_tensor(train_z[left[start:start+128]], device=device)
            b = torch.as_tensor(train_z[right[start:start+128]], device=device)
            for j, d in enumerate(pairwise_components(a, b, self.arm, self.curvature)):
                if d is not None:
                    components[j].extend(d.diag().cpu().tolist())
        scales = []
        for values in components:
            values = np.asarray(values)
            positive = values[values > 1e-12]
            scales.append(float(np.median(positive)) if len(positive) else 1.)
        self.scales = tuple(scales)
        return self

    def distances(self, a, b):
        de, dh = pairwise_components(a, b, self.arm, self.curvature)
        if self.arm == "E0":
            return de / self.scales[0]
        if self.arm == "E1":
            return dh / self.scales[1]
        return (self.alpha*(de/self.scales[0]).square()
                + (1-self.alpha)*(dh/self.scales[1]).square()).sqrt()

    @torch.no_grad()
    def neighbors(self, queries, reference, k, device, query_chunk=128, reference_chunk=2048,
                  exclude_same_slice=None):
        """Exact chunked top-k. Optional masks exclude ALL candidates in query's slice."""
        if k > len(reference) or k < 1:
            raise ValueError("Invalid k for reference size.")
        all_dist, all_idx = [], []
        for q in range(0, len(queries), query_chunk):
            a = torch.as_tensor(queries[q:q+query_chunk], device=device)
            best = torch.full((len(a), k), float("inf"), dtype=torch.float64, device=device)
            indices = torch.full((len(a), k), -1, dtype=torch.long, device=device)
            for r in range(0, len(reference), reference_chunk):
                b = torch.as_tensor(reference[r:r+reference_chunk], device=device)
                d = self.distances(a, b)
                if not torch.isfinite(d).all():
                    raise FloatingPointError("Nonfinite geometric distances.")
                if exclude_same_slice is not None:
                    qs, rs = exclude_same_slice
                    mask = qs[q:q+len(a), None] == rs[None, r:r+len(b)]
                    d.masked_fill_(torch.as_tensor(mask, device=device), float("inf"))
                candidates = torch.cat((best, d), dim=1)
                candidate_indices = torch.cat((indices, torch.arange(r, r+len(b), device=device)
                                                .expand(len(a), -1)), dim=1)
                best, positions = candidates.topk(k, largest=False, sorted=True)
                indices = candidate_indices.gather(1, positions)
            if not torch.isfinite(best).all() or (indices < 0).any():
                raise ValueError("Too few eligible cross-slice reference cells for requested k.")
            all_dist.append(best.cpu().numpy())
            all_idx.append(indices.cpu().numpy())
        return np.concatenate(all_dist), np.concatenate(all_idx)
