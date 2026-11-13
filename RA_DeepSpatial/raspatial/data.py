"""Section-level isolation and disk-backed, train-only statistics and neighborhoods."""
from collections import OrderedDict
import glob
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree

from .io import digest, identity, read_json, write_json


def h5_values(group, key):
    item = group[key]
    if isinstance(item, h5py.Group):
        categories = item['categories'].asstr()[:]
        codes = item['codes'][:]
        if (codes < 0).any():
            raise ValueError(f'Missing category: {key}')
        return categories[codes]
    return item.asstr()[:] if item.dtype.kind in 'OSU' else item[:]


def read_section(path, config, expression=True):
    """Read original coordinates; convert mm to µm exactly once, never use CCF."""
    cfg = config['data']
    with h5py.File(path) as f:
        zs = np.asarray(f['obs'][cfg['z_key']], dtype=np.float64) * cfg['coordinate_to_um']
        if not len(zs) or not np.isfinite(zs).all() or np.ptp(zs) > 1e-5:
            raise ValueError(f'Require a finite constant physical section depth: {path}')
        result = dict(name=Path(path).stem, n_cells=len(zs), z_um=float(zs[0]))
        if not expression:
            return result
        xy = np.asarray(f['obsm'][cfg['spatial_key']], dtype=np.float64) * cfg['coordinate_to_um']
        if xy.shape != (len(zs), 2) or not np.isfinite(xy).all():
            raise ValueError(f'Require finite N by 2 physical XY: {path}')
        result['physical'] = xy
        result['labels'] = h5_values(f['obs'], cfg['label_key']).astype(str)
        result['ids'] = h5_values(f['obs'], 'cell_label' if 'cell_label' in f['obs'] else f['obs'].attrs['_index']).astype(str)
        result['genes'] = h5_values(f['var'], 'ensembl' if 'ensembl' in f['var'] else f['var'].attrs['_index']).astype(str)
        result['symbols'] = h5_values(f['var'], 'gene_symbol').astype(str) if 'gene_symbol' in f['var'] else result['genes'].copy()
        x = f['X']
        if isinstance(x, h5py.Dataset):
            g = x[:]
        else:
            shape = tuple(x.attrs['shape'])
            cls = sparse.csr_matrix if str(x.attrs['encoding-type']) == 'csr_matrix' else sparse.csc_matrix
            g = cls((x['data'][:], x['indices'][:], x['indptr'][:]), shape=shape).toarray()
        result['g'] = np.asarray(g, dtype=np.float32)
    if (result['g'].shape != (len(xy), len(result['genes'])) or not np.isfinite(result['g']).all()
            or (result['g'] < 0).any() or len(set(result['genes'])) != len(result['genes'])
            or len(set(result['ids'])) != len(result['ids'])):
        raise ValueError(f'Invalid expression or duplicate identifiers: {path}')
    return result


def make_split(project, config):
    """Only filenames, physical geometry and row counts are inspected for split planning."""
    paths = sorted(Path(p).resolve() for p in glob.glob(str(Path(project) / config['input_glob'])))
    if len(paths) < 5 or len({p.stem for p in paths}) != len(paths):
        raise ValueError('Need unique section filenames and at least five sections.')
    validation, test = config['split']['validation'], config['split']['test']
    if len(set(validation + test)) != len(validation + test):
        raise ValueError('Validation and test section names must be unique and disjoint.')
    rows = []
    for path in paths:
        section = read_section(path, config, expression=False)
        rows.append(dict(section=path.stem, filename=path.name, z_um=section['z_um'], n_cells=section['n_cells'],
                         role='validation' if path.stem in validation else 'test' if path.stem in test else 'train'))
    rows.sort(key=lambda row: row['z_um'])
    if not set(validation + test).issubset({row['section'] for row in rows}) or np.any(np.diff([r['z_um'] for r in rows]) <= 0):
        raise ValueError('Missing held-out section or duplicate/non-increasing physical depths.')
    train = [r for r in rows if r['role'] == 'train']
    for row in rows:
        if row['role'] != 'train':
            left, right = bracket(train, row['z_um'])
            row.update(context_left=left['section'], context_right=right['section'])
    return rows, {p.stem: p for p in paths}


def bracket(rows, z):
    left = [r for r in rows if r['z_um'] < z]
    right = [r for r in rows if r['z_um'] > z]
    if not left or not right:
        raise ValueError(f'Depth {z} is not bracketed by observed training sections.')
    return max(left, key=lambda r: r['z_um']), min(right, key=lambda r: r['z_um'])


def neighborhood(physical, y, g, state, config):
    cfg = config['niche']
    n, cd = len(y), len(state['classes'])
    tree = cKDTree(physical)
    edges = tree.sparse_distance_matrix(tree, cfg['radius_um'], output_type='coo_matrix')
    keep = edges.row != edges.col
    row, col, distances = edges.row[keep], edges.col[keep], edges.data[keep]
    w = np.exp(-0.5 * (distances / cfg['sigma_um'])**2)
    graph = sparse.csr_matrix((w, (row, col)), shape=(n, n))
    mass = np.asarray(graph.sum(1)).ravel()
    inverse = np.divide(1., mass, out=np.zeros(n), where=mass > 0)
    onehot = sparse.csr_matrix((np.ones(n), (np.arange(n), y)), shape=(n, cd))
    composition = (graph @ onehot).toarray() * inverse[:, None]
    cols = state['niche_gene_indices']
    values = (np.log1p(g[:, cols]) - np.asarray(state['log_mean'])[cols]) / np.asarray(state['log_scale'])[cols]
    mean = (graph @ values) * inverse[:, None]
    std = np.sqrt(np.maximum((graph @ (values**2)) * inverse[:, None] - mean**2, 0))
    delta = (physical[col] - physical[row]) / cfg['radius_um']
    moments = np.column_stack([np.bincount(row, weights=w*v, minlength=n)*inverse
                               for v in (delta[:, 0], delta[:, 1], delta[:, 0]**2, delta[:, 1]**2)])
    mass2 = np.asarray(graph.power(2).sum(1)).ravel()
    effective = np.divide(mass**2, mass2, out=np.zeros(n), where=mass2 > 0)
    return np.column_stack((composition, mean, std, moments, np.log1p(mass), np.log1p(effective), mass == 0)).astype(np.float32)


def prepare(project, config, cache):
    """No validation/test expression, labels or counts fit any model statistic."""
    rows, paths = make_split(project, config)
    train = [r for r in rows if r['role'] == 'train']
    cache = Path(cache)
    from .io import source_identity
    stamp = dict(config=config, train_inputs={r['section']: digest(paths[r['section']]) for r in train},
                 source_sha256=source_identity(project))
    identity(cache, stamp)
    write_json(cache/'split.json', rows)
    pd.DataFrame(rows).to_csv(cache/'split.csv', index=False)
    state_path = cache/'state.json'
    if (cache/'prepared.json').exists():
        record = read_json(cache/'prepared.json')
        if digest(state_path) != record['state_sha256']:
            raise ValueError('Prepared state changed after creation.')
        return read_json(state_path), rows, paths
    classes, statistics, raw = {}, None, []
    xymin, xymax = np.full(2, np.inf), np.full(2, -np.inf)
    genes = symbols = None
    for row in train:
        section = read_section(paths[row['section']], config)
        if genes is None:
            genes, symbols = section['genes'], section['symbols']
            statistics = np.zeros((4, len(genes)), np.float64)
        elif not np.array_equal(section['genes'], genes) or not np.array_equal(section['symbols'], symbols):
            raise ValueError('Training gene identifiers/order changed between sections.')
        folder = cache/row['section']; folder.mkdir(exist_ok=True)
        for key in ('g', 'physical', 'ids', 'labels'):
            np.save(folder/(key+'.npy'), section[key], allow_pickle=False)
        for label, count in zip(*np.unique(section['labels'], return_counts=True)):
            classes[str(label)] = classes.get(str(label), 0) + int(count)
        g = section['g'].astype(np.float64)
        lg = np.log1p(g)
        statistics += np.stack((g.sum(0), (g*g).sum(0), lg.sum(0), (lg*lg).sum(0)))
        xymin = np.minimum(xymin, section['physical'].min(0)); xymax = np.maximum(xymax, section['physical'].max(0))
        raw.append(dict(section=row['section'], z_um=row['z_um'], n_cells=len(g)))
        print(f'Prepared raw {row["section"]}: {len(g):,} cells', flush=True)
    count = sum(classes.values())
    mean, log_mean = statistics[0]/count, statistics[2]/count
    scale = np.sqrt(np.maximum(statistics[1]/count - mean**2, 1e-6))
    log_scale = np.sqrt(np.maximum(statistics[3]/count - log_mean**2, 1e-6))
    niche_cols = np.argsort(-log_scale, kind='stable')[:config['data']['niche_genes']]
    state = dict(genes=genes.tolist(), symbols=symbols.tolist(), classes=sorted(classes),
                 counts=[classes[k] for k in sorted(classes)], gene_scale=scale.tolist(),
                 log_mean=log_mean.tolist(), log_scale=log_scale.tolist(), niche_gene_indices=niche_cols.tolist(),
                 xy_min=xymin.tolist(), xy_range=np.maximum(xymax-xymin, 1).tolist(),
                 z_min=raw[0]['z_um'], z_range=raw[-1]['z_um']-raw[0]['z_um'], train=raw,
                 niche_dim=len(classes)+2*len(niche_cols)+7, input_expression='provided_X_identity', coordinate_units='um')
    lookup = {name: i for i, name in enumerate(state['classes'])}
    compositions = np.zeros((len(classes), len(classes)), np.float64)
    type_log_sums = np.zeros((len(classes), len(genes)), np.float64)
    for row in raw:
        folder = cache/row['section']
        g, xy, labels = (np.load(folder/(k+'.npy'), mmap_mode='r') for k in ('g', 'physical', 'labels'))
        y = np.array([lookup[str(k)] for k in labels], np.int64)
        membership = sparse.csr_matrix((np.ones(len(y)), (y, np.arange(len(y)))), shape=(len(classes), len(y)))
        type_log_sums += membership @ np.log1p(g)
        np.save(folder/'y.npy', y, allow_pickle=False)
        np.save(folder/'x.npy', ((xy-xymin)/state['xy_range']).astype(np.float32), allow_pickle=False)
        h = neighborhood(xy, y, g, state, config)
        np.add.at(compositions, y, h[:, :len(classes)])
        np.save(folder/'h.npy', h, allow_pickle=False)
        print(f'Prepared niche {row["section"]}: {h.shape}', flush=True)
    enrichment = compositions / np.maximum(np.asarray(state['counts'])[:, None], 1)
    enrichment /= np.maximum(np.asarray(state['counts'])[None, :]/count, 1e-8)
    state['preferred_neighbor_code'] = np.where(compositions.sum(1) > 0, enrichment.argmax(1), -1).tolist()
    state['type_log_mean'] = (type_log_sums/np.asarray(state['counts'])[:, None]).tolist()
    write_json(state_path, state)
    write_json(cache/'prepared.json', dict(state_sha256=digest(state_path)))
    return state, rows, paths


class SliceStore:
    """Memory maps with a bounded number of open sections; no all-brain tensor."""
    def __init__(self, cache, state, capacity=8):
        self.cache, self.state, self.capacity = Path(cache), state, capacity
        self.loaded = OrderedDict()
        self.rows = {r['section']: r for r in state['train']}

    def __getitem__(self, name):
        if name not in self.rows:
            raise ValueError(f'Attempt to use a non-training section as context: {name}')
        if name not in self.loaded:
            row = self.rows[name]
            values = {k: np.load(self.cache/name/(k+'.npy'), mmap_mode='r', allow_pickle=False)
                      for k in ('x', 'g', 'y', 'h', 'physical', 'ids')}
            values.update(name=name, z_um=row['z_um'], z=float((row['z_um']-self.state['z_min'])/self.state['z_range']))
            self.loaded[name] = values
            while len(self.loaded) > self.capacity:
                self.loaded.popitem(last=False)
        self.loaded.move_to_end(name)
        return self.loaded[name]
