"""External marker candidates and differentiable internal missing-slice supervision."""
import hashlib
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch

from .data import bracket
from .io import digest


def subclass_key(value):
    match = re.match(r'^\s*(\d+)(?:\s|$)', str(value))
    return str(int(match.group(1))) if match else str(value).strip()


def import_table(source, output, sheet=None, subclass_column=None, marker_columns=None, header=0):
    """Import Table 7 or an explicit subclass/gene CSV; inspect columns before guessing."""
    source = Path(source)
    if source.suffix.lower() == '.csv':
        table = pd.read_csv(source)
    else:
        if sheet is None:
            with pd.ExcelFile(source) as book:
                sheet = 'subclass_annotation' if 'subclass_annotation' in book.sheet_names else 0
        table = pd.read_excel(source, sheet_name=sheet or 0, header=header)
    normalized = {str(c).strip().lower().replace(' ', '_'): c for c in table.columns}
    subclass_column = subclass_column or next((normalized[k] for k in ('subclass_id_label', 'subclass', 'subclass_label', 'subclass_name', 'subclass_id') if k in normalized), None)
    if marker_columns is None:
        marker_columns = [c for c in table.columns if str(c).strip().lower().replace(' ', '_') in
                          ('gene', 'marker', 'markers', 'marker_genes', 'overall_marker_genes',
                           'subclass.markers.combo', 'subclass.tf.markers.combo')]
    if subclass_column not in table or not marker_columns or any(c not in table for c in marker_columns):
        raise ValueError(f'Cannot identify subclass/positive-marker columns. Available: {list(table.columns)}. Use --inspect then --subclass-column/--marker-columns/--header explicitly.')
    rows = []
    source_hash = digest(source)
    for _, row in table.iterrows():
        if pd.isna(row[subclass_column]):
            continue
        for column in marker_columns:
            if pd.isna(row[column]):
                continue
            for gene in re.split(r'[,;|\s]+', str(row[column]).strip()):
                # Keep symbols/Ensembl IDs; do not reinterpret plus/minus or annotation prose.
                if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.\-]*', gene) and re.search(r'[A-Za-z]', gene):
                    provenance = str(row.get('source', '')).strip()
                    provenance = provenance if provenance and provenance != 'nan' else ('Yao2023_Table7' if '41586_2023_6812_MOESM8' in source.stem else 'external_table_unverified')
                    rows.append(dict(subclass=str(row[subclass_column]).strip(), gene=gene,
                                     source=provenance,
                                     source_sha256=source_hash, source_sheet=str(sheet) if sheet is not None else '',
                                     source_column=str(column)))
    result = pd.DataFrame(rows).drop_duplicates(['subclass', 'gene'])
    if result.empty:
        raise ValueError('No positive marker records imported.')
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f'Marker output already exists: {output}')
    result.to_csv(output, index=False)
    return result


def load_markers(path, state, config):
    """External candidates only; deterministic per-type train/evaluation gene split."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Marker resource required for C: {path}. See resources/README.md; no hidden-data-derived fallback is used.')
    table = pd.read_csv(path, dtype=str).fillna('')
    if not {'subclass', 'gene', 'source'}.issubset(table.columns) or (table.source.str.len() == 0).any():
        raise ValueError('markers.csv needs subclass,gene,source columns with provenance.')
    lookup = {}
    for i, (gene, symbol) in enumerate(zip(state['genes'], state['symbols'])):
        lookup[str(gene)] = i
        if str(symbol) not in lookup:
            lookup[str(symbol)] = i
    classes = {subclass_key(k): i for i, k in enumerate(state['classes'])}
    aliases = {}
    for i, name in enumerate(state['classes']):
        alias = re.sub(r'^\s*\d+\s+', '', name).strip()
        aliases.setdefault(alias, []).append(i)
    classes.update({alias: items[0] for alias, items in aliases.items() if len(items) == 1})
    pools = {c: set() for c in range(len(state['classes']))}
    for row in table.itertuples(index=False):
        c, gene = classes.get(subclass_key(row.subclass)), lookup.get(row.gene)
        if c is not None and gene is not None:
            pools[c].add(gene)
    train, evaluation, rows = {}, {}, []
    for c, name in enumerate(state['classes']):
        cols = sorted(pools[c], key=lambda j: hashlib.sha256((name+'|'+state['genes'][j]).encode()).hexdigest())
        # Independent-gene evaluation is possible only with sufficient panel overlap.
        held = cols[::5] if len(cols) >= 5 else []
        available = [j for j in cols if j not in held]
        own = np.asarray(state['type_log_mean'][c])
        n, nc = sum(state['counts']), state['counts'][c]
        rest = (n*np.asarray(state['log_mean'])-nc*own)/max(n-nc, 1)
        contrast = own-rest
        available = sorted((j for j in available if contrast[j] > 0), key=lambda j: (-contrast[j], j))
        train[c] = available[:config['markers']['max_per_type']]
        if len(train[c]) < config['markers']['min_per_type']:
            train[c] = []
        evaluation[c] = held
        rows.append(dict(subclass=name, external_panel_n=len(cols), loss_marker_n=len(train[c]), heldout_C_marker_n=len(held),
                         loss_markers=';'.join(state['symbols'][j] for j in train[c]),
                         heldout_C_markers=';'.join(state['symbols'][j] for j in held)))
    if not any(train.values()):
        raise ValueError('External markers do not overlap the training taxonomy and measured panel.')
    return train, evaluation, pd.DataFrame(rows)


def tensor_inputs(left, right, row, col, state, device, reverse=False):
    src, dst = (right, left) if reverse else (left, right)
    i, j = (col, row) if reverse else (row, col)
    tensor = lambda value: torch.as_tensor(np.asarray(value).copy(), device=device, dtype=torch.float32)
    c0 = torch.nn.functional.one_hot(torch.as_tensor(np.asarray(src['y'][i]).copy(), device=device), len(state['classes'])).float()
    return dict(xt=tensor(src['x'][i]), gt=tensor(src['g'][i]), ct=c0,
                source_c=c0, h0=tensor(src['h'][i]), h1=tensor(dst['h'][j]),
                delta_z=torch.full((len(i), 1), dst['z']-src['z'], device=device),
                zt=torch.full((len(i), 1), src['z'], device=device), t=torch.zeros(len(i), device=device))


def rollout(model, inputs, fraction, steps):
    """Differentiable midpoint ODE, including derivatives through frozen base w.r.t. state."""
    values = dict(inputs)
    dt = float(fraction)/steps
    z0 = inputs['zt']
    for step in range(steps):
        values['t'] = torch.full_like(inputs['t'], step*dt)
        values['zt'] = z0+step*dt*inputs['delta_z']
        first = model(**values)
        middle = dict(values)
        for key, velocity in zip(('xt', 'gt', 'ct'), first):
            middle[key] = values[key]+.5*dt*velocity
        middle['t'] = values['t']+.5*dt
        middle['zt'] = values['zt']+.5*dt*inputs['delta_z']
        second = model(**middle)
        for key, velocity in zip(('xt', 'gt', 'ct'), second):
            values[key] = values[key]+dt*velocity
    return values['xt'], values['gt'], values['ct']


class MarkerTasks:
    """Each target is a TRAIN section removed from its own endpoint context.

    Target coordinates/expression enter the loss only. Validation/test sections
    are not loadable through SliceStore and cannot become marker targets.
    """
    def __init__(self, store, transports, state, config, markers, seed=42, transport_kind='local'):
        self.store, self.transports, self.state, self.config = store, transports, state, config
        self.markers = markers
        self.transport_kind = transport_kind
        fractions = np.asarray(state['counts'])/sum(state['counts'])
        self.rare = {c for c, p in enumerate(fractions) if p <= config['evaluation']['rare_fraction'] and markers[c]}
        rows = state['train']
        self.tasks = []
        for target in rows[1:-1]:
            endpoints = [r for r in rows if r['section'] != target['section']]
            left, right = bracket(endpoints, target['z_um'])
            self.tasks.append((left['section'], target['section'], right['section']))
        self.rng = np.random.default_rng(seed)
        width = config['C']['grid_um']
        lo, span = np.asarray(state['xy_min']), np.asarray(state['xy_range'])
        gx, gy = np.meshgrid(np.arange(lo[0]+width/2, lo[0]+span[0]+width/2, width),
                             np.arange(lo[1]+width/2, lo[1]+span[1]+width/2, width), indexing='ij')
        self.grid = np.column_stack((gx.ravel(), gy.ravel()))
        from collections import OrderedDict
        self.summaries = OrderedDict()

    def loss(self, model, device):
        if not self.tasks or not self.rare:
            raise ValueError('No train-internal marker tasks with supported rare marker types.')
        # Try tasks in a deterministic random order; no zero-loss silent fallback.
        for k in self.rng.permutation(len(self.tasks)):
            lname, tname, rname = self.tasks[int(k)]
            left, target, right = self.store[lname], self.store[tname], self.store[rname]
            if self.transport_kind == 'local':
                plan = self.transports.local(lname, rname).tocoo()
                supported = set(left['y'][plan.row]) if plan.nnz else set()
            else:
                supported = set(left['y']) & set(right['y'])
            candidates = sorted(self.rare & supported & set(target['y']))
            if candidates:
                break
        else:
            raise ValueError('No internal target has a supported rare type with measured external markers.')
        c = int(self.rng.choice(candidates))
        reverse = bool(self.rng.integers(2))
        if self.transport_kind == 'local':
            mask = left['y'][plan.row] == c
            edges = np.flatnonzero(mask)
            choice = self.rng.choice(edges, self.config['C']['cells'], p=plan.data[edges]/plan.data[edges].sum())
            row, col = plan.row[choice], plan.col[choice]
        else:
            # A is removed from C's candidate construction as well as main FM.
            # Internal skip-one gaps are distinct from the adjacent FM gaps;
            # retain one unused sample solely to cache exact global UOT factors.
            self.transports.raw(lname, rname, 1)
            src = right if reverse else left
            parents = self.rng.choice(np.flatnonzero(src['y'] == c), self.config['C']['cells'])
            targets = self.transports.targets(lname, rname, not reverse, parents, 'raw', self.rng)
            row, col = (targets, parents) if reverse else (parents, targets)
        inputs = tensor_inputs(left, right, row, col, self.state, device, reverse)
        fraction = (target['z']-left['z'])/(right['z']-left['z'])
        xy, g, _ = rollout(model, inputs, 1-fraction if reverse else fraction, self.config['C']['rollout_steps'])
        columns = self.markers[c]
        xy = xy*xy.new_tensor(self.state['xy_range'])+xy.new_tensor(self.state['xy_min'])
        grid = xy.new_tensor(self.grid)
        bandwidth = self.config['C']['bandwidth_um']
        kernel = torch.exp(-torch.cdist(xy, grid).square()/(2*bandwidth**2))
        expression = torch.log1p(g[:, columns].clamp_min(0))
        field = kernel.T@expression/len(g)
        key = tname, c
        summary = self.summaries.get(key)
        if summary is None:
            rows = np.flatnonzero(target['y'] == c)
            tx = target['physical'][rows]
            tg = np.log1p(target['g'][rows][:, columns])
            truth_field = np.zeros((len(self.grid), len(columns)))
            for start in range(0, len(rows), 256):
                d = tx[start:start+256, None, :]-self.grid[None, :, :]
                kern = np.exp(-np.sum(d*d, axis=-1)/(2*bandwidth**2))
                truth_field += kern.T@tg[start:start+256]/len(rows)
            summary = dict(field=truth_field.astype(np.float32), mean=tg.mean(0).astype(np.float32))
            self.summaries[key] = summary
            while len(self.summaries) > 64:
                self.summaries.popitem(last=False)
        truth = field.new_tensor(summary['field'])
        a, b = field-field.mean(0), truth-truth.mean(0)
        # Smooth zero norms before sqrt: a zero generated field must not produce NaN gradients.
        norm = (a.square().sum(0)+1e-12).sqrt()*(b.square().sum(0)+1e-12).sqrt()
        valid = b.square().sum(0) > 1e-12
        correlation = (a*b).sum(0)/norm.clamp_min(1e-8)
        spatial = (1-correlation[valid]).mean() if valid.any() else field.sum()*0
        expression_loss = (expression.mean(0)-expression.new_tensor(summary['mean'])).square().mean()
        cfg = self.config['C']
        loss = cfg['expression_weight']*expression_loss+cfg['spatial_weight']*spatial
        return loss, dict(target=tname, subclass=self.state['classes'][c], marker_expression=float(expression_loss.detach()),
                          marker_spatial=float(spatial.detach()), spatial_genes=int(valid.sum()))
