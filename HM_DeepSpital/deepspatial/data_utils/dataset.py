import gc
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from tqdm import tqdm

from .uot_solver import compute_uot_coupling
from .sampling import sample_coupling_indices
from .local_matching import compute_local_uot, sample_local_pairs

class DeepSpatialDataset(Dataset):
    """
    DeepSpatial Global Trajectory Dataset.
    Constructs cross-slice cell pairs using Unbalanced Optimal Transport (UOT) 
    to provide continuous training samples for Flow Matching.
    """
    def __init__(self, 
                 adata_list: list, 
                 spatial_key: str = 'spatial_norm', 
                 z_key: str = 'z_norm',
                 label_key: str = 'cell_class',
                 n_samples_base: int = 50000, 
                 alpha_spatial: float = 0.5,
                 uot_reg: float = 0.8, 
                 uot_tau: float = 0.05,
                 mode: str = 'fit',
                 balanced_fraction: float = 0.0,
                 niche_key: str = None,
                 local_matching_config: dict = None,
                 physical_spatial_key: str = 'spatial',
                 physical_z_key: str = 'z_coord'):
        """
        Args:
            adata_list: List of AnnData objects.
            spatial_key: Key in .obsm containing normalized XY coordinates.
            z_key: Key in .obs containing normalized Z coordinates.
            label_key: Key in .obs for cell labels/types.
            n_samples_base: Target total number of trajectory pairs.
            alpha_spatial: Balance between spatial and gene distances for UOT.
            uot_reg: Entropy regularization for UOT solver.
            uot_tau: Marginal relaxation for UOT solver.
            mode: 'fit' for full multi-slice training, 'predict' for limited slice pairs.
            balanced_fraction: Fraction sampled uniformly across valid same-type UOT blocks.
            niche_key: Precomputed source-neighborhood features; None disables conditioning.
            local_matching_config: Optional physical same-type candidate/cost scales (G).
            physical_spatial_key: Raw XY in micrometers for G; never normalized coordinates.
            physical_z_key: Raw depth in micrometers for G's gap-dependent scales.
        """
        self.adata_list = adata_list
        self.spatial_key = spatial_key
        self.z_key = z_key
        self.label_key = label_key
        self.n_samples_base = n_samples_base
        self.alpha_spatial = alpha_spatial
        self.uot_reg = uot_reg
        self.uot_tau = uot_tau
        if not 0 <= balanced_fraction <= 1:
            raise ValueError("balanced_fraction must lie in [0, 1].")
        self.balanced_fraction = balanced_fraction
        self.niche_key = niche_key
        self.sampling_stats = []
        self.local_matching_config = local_matching_config
        self.physical_spatial_key = physical_spatial_key
        self.physical_z_key = physical_z_key
        self.matching_diagnostics = []

        # --- 1. Global Label Encoding ---
        self.label_encoder = LabelEncoder()
        all_labels = []
        for adata in adata_list:
            if label_key in adata.obs:
                all_labels.extend(adata.obs[label_key].astype(str).tolist())
        
        if all_labels:
            self.label_encoder.fit(all_labels)
            self.num_classes = len(self.label_encoder.classes_)
            self.id2label = {i: label for i, label in enumerate(self.label_encoder.classes_)}
        else:
            self.num_classes = 1
            self.id2label = {0: "unknown"}

        if mode != 'fit':
            self.adata_list = self.adata_list[:2]

        # Containers for trajectory pairs
        self.trajectory_pairs = {
            'x0': [], 'g0': [], 'c0': [], 'z0': [],
            'x1': [], 'g1': [], 'c1': [], 'z1': [],
            'delta_z': [],              
            'slice0': [], 'cell0': [], 'slice1': [], 'cell1': [],
        }
        if niche_key is not None:
            self.trajectory_pairs.update({'h0': [], 'h1': []})
        
        self._build_trajectory_dataset()
        self._convert_to_tensors()

    def _get_data_arrays(self, adata):
        """Extracts spatial, gene expression, and one-hot labels from AnnData."""
        x = adata.obsm[self.spatial_key].astype(np.float32)
        g = adata.X.toarray().astype(np.float32) if sp.issparse(adata.X) else adata.X.astype(np.float32)
            
        if self.label_key in adata.obs:
            raw_labels = adata.obs[self.label_key].astype(str).values
            indices = self.label_encoder.transform(raw_labels)
            c_onehot = np.eye(self.num_classes)[indices].astype(np.float32)
        else:
            c_onehot = np.zeros((adata.n_obs, self.num_classes), dtype=np.float32)
            
        # Z coordinate is assumed uniform across a single slice
        z = float(adata.obs[self.z_key].iloc[0])
            
        return x, g, c_onehot, z

    def _build_trajectory_dataset(self):
        """Pairs cells across adjacent slices using UOT coupling."""
        num_slices = len(self.adata_list)
        
        # Allocate samples based on slice product weights
        pair_sizes = [self.adata_list[k].n_obs * self.adata_list[k+1].n_obs for k in range(num_slices - 1)]
        total_weight = sum(pair_sizes)
        sampling_counts = [int(self.n_samples_base * (w / total_weight)) for w in pair_sizes]

        for k in tqdm(range(num_slices - 1), desc="DeepSpatial: Building Trajectories"):
            n_to_sample = sampling_counts[k]
            if n_to_sample <= 0:
                continue

            x0, g0, c0, z0 = self._get_data_arrays(self.adata_list[k])
            x1, g1, c1, z1 = self._get_data_arrays(self.adata_list[k+1])
            delta_z = z1 - z0

            # Compute Unbalanced Optimal Transport (UOT)
            if self.local_matching_config is not None:
                a0, a1 = self.adata_list[k:k + 2]
                physical_z = [float(a.obs[self.physical_z_key].iloc[0]) for a in (a0, a1)]
                pi, support = compute_local_uot(
                    a0.obsm[self.physical_spatial_key], g0, c0.argmax(1),
                    a1.obsm[self.physical_spatial_key], g1, c1.argmax(1),
                    physical_z[1] - physical_z[0], self.local_matching_config,
                    self.alpha_spatial, self.uot_reg, self.uot_tau,
                )
                for side, (a, diagnostic) in enumerate(zip((a0, a1), support)):
                    self.matching_diagnostics.append(dict(
                        slice_pair=f'{k}-{k + 1}', side=side,
                        source_z_um=physical_z[side], target_z_um=physical_z[1 - side],
                        cell_id=a.obs_names.to_numpy(), cell_class=a.obs[self.label_key].astype(str).to_numpy(),
                        **diagnostic,
                    ))
                if pi.sum() <= 0:
                    raise ValueError(f'No supported local UOT edges in gap {physical_z}; no synthetic pairs added.')
            else:
                pi = compute_uot_coupling(
                    x0, g0, c0, x1, g1, c1,
                    alpha_spatial=self.alpha_spatial,
                    uot_reg=self.uot_reg,
                    uot_tau=self.uot_tau,
                )
            
            # Sampling logic
            pi_sum = float(pi.sum())
            
            if pi_sum > 0:
                sampler = sample_local_pairs if self.local_matching_config is not None else sample_coupling_indices
                idx0, idx1, stats = sampler(
                    pi, c0.argmax(axis=1), c1.argmax(axis=1), n_to_sample,
                    self.balanced_fraction,
                )
                stats.update({
                    'slice_pair': [k, k + 1], 'sampled_pairs': n_to_sample,
                    'source_type_counts': np.bincount(c0[idx0].argmax(axis=1), minlength=self.num_classes).tolist(),
                    'target_type_counts': np.bincount(c1[idx1].argmax(axis=1), minlength=self.num_classes).tolist(),
                    'cross_type_pairs': int((c0[idx0].argmax(axis=1) != c1[idx1].argmax(axis=1)).sum()),
                    'unique_source_cells': int(len(np.unique(idx0))),
                    'unique_target_cells': int(len(np.unique(idx1))),
                    'transport_total_mass': float(pi_sum),
                    'source_type_transport_mass': np.bincount(
                        c0.argmax(axis=1), weights=np.asarray(pi.sum(axis=1)).ravel(), minlength=self.num_classes,
                    ).tolist(),
                    'target_type_transport_mass': np.bincount(
                        c1.argmax(axis=1), weights=np.asarray(pi.sum(axis=0)).ravel(), minlength=self.num_classes,
                    ).tolist(),
                })
                self.sampling_stats.append(stats)
                if self.local_matching_config is not None:
                    distances = np.linalg.norm(a0.obsm[self.physical_spatial_key][idx0] - a1.obsm[self.physical_spatial_key][idx1], axis=1)
                    stats['sampled_distance_um'] = dict(mean=float(distances.mean()), maximum=float(distances.max()))

                # Store trajectory endpoints
                self.trajectory_pairs['x0'].append(x0[idx0])
                self.trajectory_pairs['g0'].append(g0[idx0])
                self.trajectory_pairs['c0'].append(c0[idx0])
                self.trajectory_pairs['z0'].append(np.full((n_to_sample, 1), z0, dtype=np.float32))
                
                self.trajectory_pairs['x1'].append(x1[idx1])
                self.trajectory_pairs['g1'].append(g1[idx1])
                self.trajectory_pairs['c1'].append(c1[idx1])
                self.trajectory_pairs['z1'].append(np.full((n_to_sample, 1), z1, dtype=np.float32))
                self.trajectory_pairs['delta_z'].append(np.full((n_to_sample, 1), delta_z, dtype=np.float32))
                # 保留切片内端点索引，便于检查稀有来源是否真正进入训练。
                for suffix, cell_indices, slice_index in [('0', idx0, k), ('1', idx1, k + 1)]:
                    self.trajectory_pairs['cell' + suffix].append(cell_indices.astype(np.int64))
                    self.trajectory_pairs['slice' + suffix].append(np.full(n_to_sample, slice_index, dtype=np.int64))
                    if self.niche_key is not None:
                        features = self.adata_list[slice_index].obsm[self.niche_key]
                        self.trajectory_pairs['h' + suffix].append(np.asarray(features[cell_indices], dtype=np.float32))

            del pi, x0, g0, c0, x1, g1, c1
            gc.collect()

    def _convert_to_tensors(self):
        """Aggregates list of arrays into final PyTorch tensors."""
        self.tensors = {}
        for key in self.trajectory_pairs:
            if self.trajectory_pairs[key]:
                # Use np.concatenate for efficiency before converting to tensor
                concatenated = np.concatenate(self.trajectory_pairs[key], axis=0)
                self.tensors[key] = torch.from_numpy(concatenated)
            
        self.trajectory_pairs.clear()
        gc.collect()
        
        if 'x0' in self.tensors:
            self.num_samples = self.tensors['x0'].shape[0]
        else:
            self.num_samples = 0

    def __len__(self):
        '返回可用训练轨迹的数量。'
        return self.num_samples

    def __getitem__(self, idx):
        '按索引返回保存的数据。'
        return {k: v[idx] for k, v in self.tensors.items()}

    def decode_label(self, one_hot_vec):
        """Converts model one-hot output back to original string label."""
        idx = torch.argmax(one_hot_vec, dim=-1).item()
        return self.id2label[idx]
