"""B: zero-initialized context residual. C: per-modality trust controller."""
import numpy as np
import torch
from torch import nn

from .vendor.git import GiT


class BaseFlow(nn.Module):
    """The unmodified upstream GiT, with a uniform experiment call signature."""
    def __init__(self, state, config):
        super().__init__()
        self.git = GiT(gene_dim=len(state['genes']), num_classes=len(state['classes']), **config['model'])

    def forward(self, xt, gt, ct, t, zt, delta_z, **unused):
        return self.git(xt=xt, gt=gt, ct=ct, t=t, zt=zt, delta_z=delta_z)


class TrustController(nn.Module):
    """A deterministic support/type gate and normalized RMS caps, NOT confidence probabilities."""
    def __init__(self, state, config, enabled):
        super().__init__()
        cfg = config['trust']
        self.enabled = bool(enabled)
        counts = np.asarray(state['counts'], float)
        rare = counts/counts.sum() <= config['evaluation']['rare_fraction']
        if not 0 <= cfg['common_gate'] <= 1:
            raise ValueError('common_gate must be in [0,1].')
        self.register_buffer('class_gates', torch.tensor(np.where(rare, 1., cfg['common_gate']), dtype=torch.float32))
        self.register_buffer('gene_scale', torch.tensor(state['gene_scale'], dtype=torch.float32))
        self.caps = tuple(float(cfg['max_rms'][m]) for m in ('x','g','c'))
        if not all(np.isfinite(self.caps)) or min(self.caps) <= 0:
            raise ValueError('Trust RMS budgets must be positive and finite.')

    def normalized(self, residual):
        return residual[0], residual[1]/self.gene_scale, residual[2]

    def forward(self, residual, source_c, support=None):
        if not self.enabled:
            return residual
        gate = self.class_gates[source_c.argmax(-1)].unsqueeze(-1)
        if support is not None:
            gate = gate*support.reshape(-1,1).to(gate)
        result = []
        for values, normed, cap in zip(residual, self.normalized(residual), self.caps):
            rms = normed.square().mean(-1, keepdim=True).add(1e-16).sqrt()
            result.append(gate*values*torch.clamp(cap/rms, max=1.))
        return tuple(result)


class ResidualFlow(nn.Module):
    """All residual arms have identical capacity; context switches zero inputs only."""
    def __init__(self, state, config, arm, base_state=None):
        super().__init__()
        self.base = BaseFlow(state, config)
        if base_state is not None:
            self.base.load_state_dict(base_state)
        self.base.requires_grad_(False)
        self.base.eval()
        self.context = bool(arm['B'])
        self.dual = bool(arm.get('dual', True))
        self.gdim, self.cdim = len(state['genes']), len(state['classes'])
        hdim, width = len(state['niche_features']), config['residual']['hidden_size']
        self.register_buffer('gene_scale', torch.tensor(state['gene_scale'], dtype=torch.float32))
        self.self_encoder = nn.Sequential(nn.Linear(2+self.gdim+2*self.cdim+3, width), nn.SiLU(), nn.Linear(width,width))
        self.context_encoder = nn.Sequential(nn.Linear(2*hdim,width), nn.SiLU(), nn.Linear(width,width))
        self.head = nn.Sequential(nn.SiLU(), nn.Linear(width, 2+self.gdim+self.cdim))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.controller = TrustController(state, config, arm['C'])

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, xt, gt, ct, t, zt, delta_z, h0, h1, source_c, support=None, return_parts=False):
        with torch.no_grad():
            base = self.base(xt=xt, gt=gt, ct=ct, t=t, zt=zt, delta_z=delta_z)
        h0 = h0 if self.context else torch.zeros_like(h0)
        h1 = h1 if self.context and self.dual else torch.zeros_like(h1)
        own = torch.cat((xt, gt/self.gene_scale, ct, source_c,
                         t.reshape(-1,1), zt.reshape(-1,1), delta_z.reshape(-1,1)), -1)
        hidden = self.self_encoder(own)+self.context_encoder(torch.cat((h0,h1),-1))
        raw = self.head(hidden)
        residual = (raw[:, :2], raw[:, 2:2+self.gdim], raw[:, 2+self.gdim:])
        residual = self.controller(residual, source_c, support)
        out = tuple(b+r for b,r in zip(base,residual))
        return (out, base, residual) if return_parts else out


def make_model(state, config, arm, base_state=None):
    return BaseFlow(state, config) if arm.get('base', False) else ResidualFlow(state, config, arm, base_state)
