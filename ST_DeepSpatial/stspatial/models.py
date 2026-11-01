"""Three independent heads. No joint state ODE and no inherited cell labels."""
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .vendor.git import GiT


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mlp(inputs, outputs, width):
    return nn.Sequential(nn.Linear(inputs, width), nn.SiLU(), nn.Linear(width, width),
                         nn.SiLU(), nn.Linear(width, outputs))


class SpatialDensity(nn.Module):
    """A label-free Poisson intensity residual over interpolated endpoint density."""
    def __init__(self, config):
        super().__init__()
        self.net = mlp(7, 1, config['training']['width'])
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
        self.limit = config['spatial']['max_log_ratio']

    def forward(self, features, baseline):
        log_ratio = self.limit*torch.tanh(self.net(features).squeeze(-1)/self.limit)
        return baseline*log_ratio.exp(), log_ratio


class PositionType(nn.Module):
    """Predict type at a query position using only position and observed type geometry."""
    def __init__(self, nclasses, config):
        super().__init__()
        self.net = mlp(3+2*(nclasses+4), nclasses, config['training']['width'])
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, prior):
        return self.net(features)+prior.clamp_min(1e-5).log()


class ConditionalExpression(nn.Module):
    """A hurdle mean: presence probability times positive expression magnitude.

    This works with noninteger nonnegative X; no count likelihood is assumed for X.
    Predicted target-type probabilities are separate from observed niche composition.
    """
    def __init__(self, state, config):
        super().__init__()
        nc, ng = len(state['classes']), len(state['genes'])
        self.nclasses = nc
        self.git = GiT(gene_dim=ng, num_classes=nc, **config['git'])
        # Original spatial/type heads are not part of the stage-3 objective/output.
        self.git.x_head.requires_grad_(False)
        self.git.c_head.requires_grad_(False)
        self.net = mlp(3+2*(nc+4)+nc+4*ng, 2*ng, config['training']['width'])
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('max_log', torch.tensor(state['gene_max_log'], dtype=torch.float32))
        self.register_buffer('scale', torch.tensor(state['gene_scale'], dtype=torch.float32))
        self.register_buffer('global_mean', torch.tensor(state['gene_mean'], dtype=torch.float32))
        self.register_buffer('global_positive', torch.tensor(state['gene_positive'], dtype=torch.float32))

    def forward(self, features, types, gene_features, mean, positive, no_type=False, no_niche=False):
        if no_type:
            types = torch.zeros_like(types)
        if no_niche:
            # Keep observed niche type geometry; remove measured niche expression only.
            gene_features = torch.zeros_like(gene_features)
            mean = self.global_mean.expand_as(mean)
            positive = self.global_positive.expand_as(positive)
        raw = self.net(torch.cat((features, types, gene_features), -1))
        presence_delta, context_shift = raw.chunk(2, -1)
        # Preserve GiT's position token, gene patches, type/depth conditioning and
        # transformer blocks; reinterpret its gene head as a static magnitude
        # correction. Position/type heads are discarded. There is no joint ODE.
        _, magnitude_delta, _ = self.git(xt=features[:, :2], gt=torch.log1p(mean)+context_shift,
                                         ct=types, t=torch.ones(len(features), device=features.device),
                                         zt=features[:, 2],
                                         delta_z=features[:, -1]-features[:, self.nclasses+6])
        positive = positive.clamp(.005, .995)
        logits = torch.logit(positive)+presence_delta
        base_log = torch.log1p(mean/positive).clamp_min(1e-4)
        # Inverse-softplus baseline makes the zero residual reproduce the local mean.
        inverse = base_log+torch.log(-torch.expm1(-base_log))
        magnitude = torch.minimum(F.softplus(inverse+magnitude_delta), self.max_log)
        expected = logits.sigmoid()*torch.expm1(magnitude)
        return expected, logits, magnitude


def batches_predict(model, arrays, device, batch_size, all_outputs=False, **kwargs):
    outputs = None
    model.eval()
    with torch.no_grad():
        for start in range(0, len(arrays[0]), batch_size):
            values = [torch.as_tensor(a[start:start+batch_size], device=device) for a in arrays]
            out = model(*values, **kwargs)
            parts = out if isinstance(out, tuple) else (out,)
            parts = parts if all_outputs else parts[:1]
            if outputs is None:
                outputs = [[] for _ in parts]
            for bucket, value in zip(outputs, parts):
                bucket.append(value.cpu().numpy())
    result = tuple(np.concatenate(bucket) for bucket in outputs)
    return result if all_outputs else result[0]


def fit(model, arrays, loss_fn, config, folder, name, seed, device, resume=False):
    """Epoch-boundary resume with optimizer/loader/CPU/CUDA RNG state."""
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    seed_all(seed); model.to(device)
    cfg = config['training']; epochs = cfg['epochs'][name]
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    generator = torch.Generator().manual_seed(seed)
    checkpoint = folder/'last.pt'; history = []; start = 0; elapsed = 0.
    if checkpoint.exists():
        if not resume:
            raise FileExistsError(f'Checkpoint exists: {checkpoint}; use --resume.')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        history, start, elapsed = saved['history'], saved['epoch'], saved['elapsed_seconds']
        generator.set_state(saved['loader_rng']); torch.set_rng_state(saved['torch_rng'])
        if device.type == 'cuda' and saved['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
    loader = DataLoader(TensorDataset(*[torch.as_tensor(a) for a in arrays]),
                        batch_size=cfg['batch_size'], shuffle=True, generator=generator)
    for epoch in range(start, epochs):
        started = time.perf_counter(); totals = {}; n = 0; model.train()
        for batch in loader:
            batch = [x.to(device) for x in batch]
            optimizer.zero_grad(set_to_none=True)
            loss, parts = loss_fn(model, *batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite {name} loss; no completed stage recorded.')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step(); size = len(batch[0]); n += size
            for key, value in dict(loss=loss, **parts).items():
                totals[key] = totals.get(key, 0.)+float(value.detach())*size
        elapsed += time.perf_counter()-started
        row = dict(epoch=epoch+1, **{k: v/n for k, v in totals.items()})
        history.append(row)
        saved = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch+1,
                     history=history, elapsed_seconds=elapsed, loader_rng=generator.get_state(),
                     torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)
        tmp = folder/'last.pt.tmp'; torch.save(saved, tmp); os.replace(tmp, checkpoint)
        pd.DataFrame(history).to_csv(folder/'training.csv', index=False)
        print(f'{folder.parent.name}/{name}: {epoch+1}/{epochs} loss={row["loss"]:.6g}', flush=True)
    return model.eval()
