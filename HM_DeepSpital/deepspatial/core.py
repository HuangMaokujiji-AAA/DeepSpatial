import os
import json
import torch
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse
import pytorch_lightning as pl
from tqdm import tqdm
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from .data_utils import DeepSpatialDataset
from .models import GiT
from .module import DeepSpatialModule
from .bio_prior import NicheFeaturizer
from .data_utils.sampling import stratified_source_indices


class DeepSpatial:
    """
    DeepSpatial: Reconstructing True 3D Spatial Omics at Single-Cell Resolution.
    """
    def __init__(self):
        """
        Initializes an empty DeepSpatial instance. 
        Configurations and paths are injected during functional method calls.
        """
        # Data and Dataloader
        self.dataset = None
        self.train_loader = None
        
        # Model and Training state
        self.gene_dim = None
        self.num_classes = None
        self.model = None
        self.module = None
        
        # Metadata for persistence and restoration
        self.categories = None
        self.spatial_stats = None
        self.spatial_key = None
        self.z_key = None
        self.label_key = None
        
        # Configuration dictionaries
        self.model_config = {}
        self.train_config = {}
        self.data_config = {}
        self.gene_names = None
        self.niche_featurizer = None
        self.source_sampling = 'random'

    def _normalize_spatial(self, adata_list: list[ad.AnnData]) -> None:
        """
        Extracts coordinates from user keys and computes normalization stats.
        Stores normalized data in internal 'spatial_norm' and 'z_norm' keys.
        """
        z_raw_list = []
        all_spatial = []
        
        for i, adata in enumerate(adata_list):
            if self.spatial_key not in adata.obsm:
                raise KeyError(f"Spatial key '{self.spatial_key}' not found in slice {i}.")
            if self.z_key not in adata.obs:
                raise KeyError(f"Z-coord key '{self.z_key}' not found in slice {i}.")
            
            all_spatial.append(adata.obsm[self.spatial_key])
            z_raw_list.append(adata.obs[self.z_key].iloc[0])
            
        all_spatial = np.vstack(all_spatial)
        z_raw_arr = np.array(z_raw_list)
        
        # Store stats for physical space restoration
        self.spatial_stats = {
            'x_min': float(all_spatial[:, 0].min()),
            'x_range': float(all_spatial[:, 0].max() - all_spatial[:, 0].min() + 1e-8),
            'y_min': float(all_spatial[:, 1].min()),
            'y_range': float(all_spatial[:, 1].max() - all_spatial[:, 1].min() + 1e-8),
            'z_min': float(z_raw_arr.min()),
            'z_range': float(z_raw_arr.max() - z_raw_arr.min() + 1e-8)
        }

        norm_z_arr = (z_raw_arr - self.spatial_stats['z_min']) / self.spatial_stats['z_range']
    
        # Perform non-destructive normalization
        for i, adata in enumerate(adata_list):
            coords = np.asarray(adata.obsm[self.spatial_key]).copy()
            if not np.issubdtype(coords.dtype, np.floating):
                coords = coords.astype(np.float64)
            coords[:, 0] = (coords[:, 0] - self.spatial_stats['x_min']) / self.spatial_stats['x_range']
            coords[:, 1] = (coords[:, 1] - self.spatial_stats['y_min']) / self.spatial_stats['y_range']
            
            adata.obsm['spatial_norm'] = coords
            adata.obs['z_norm'] = norm_z_arr[i]

    def setup_data(self, 
                   adata_list: list[ad.AnnData], 
                   spatial_key: str = 'spatial',
                   z_key: str = 'z_coord',
                   label_key: str = 'cell_class',
                   batch_size: int = 128, 
                   num_workers: int = 4,
                   n_samples_base: int = 50000,
                   alpha_spatial: float = 0.5,
                   uot_reg: float = 0.8,
                   uot_tau: float = 0.05,
                   mode: str = 'fit',
                   niche_config: dict = None,
                   balanced_fraction: float = 0.0,
                   source_sampling: str = 'random'):
        """
        Prepares the data pipeline and calculates physical normalization statistics.
        
        Args:
            adata_list: List of AnnData objects (slices).
            spatial_key: Key in `.obsm` for XY coordinates.
            z_key: Key in `.obs` for the physical Z coordinate.
            label_key: Key in `.obs` for cell type annotations.
            batch_size: Number of samples per training batch.
            num_workers: Multi-process data loading workers.
            n_samples_base: Base number of cell pairs to sample per slice pair.
            alpha_spatial: UOT spatial distance weight.
            uot_reg: Entropy regularization for UOT.
            uot_tau: Marginal relaxation for UOT.
            mode: Dataset mode ('fit' for training, 'predict' for inference).
            niche_config: 邻域半径、程序权重等；None 保留无生态位条件基线。
            balanced_fraction: 有效同类 UOT 配对的混合抽样比例。
            source_sampling: 'random' 或保持来源期望比例的 'stratified'。
        """
        self.spatial_key = spatial_key
        self.z_key = z_key
        self.label_key = label_key
        if mode not in ('fit', 'predict'):
            raise ValueError("mode must be 'fit' or 'predict'.")
        if mode == 'predict':
            adata_list = adata_list[:2]
        if source_sampling not in ('random', 'stratified'):
            raise ValueError("source_sampling must be 'random' or 'stratified'.")
        self.source_sampling = source_sampling
        self._validate_slices(adata_list, require_order=True)
        self.gene_names = adata_list[0].var_names.tolist()
        self.data_config = {'balanced_fraction': balanced_fraction, 'source_sampling': source_sampling}
        self.niche_featurizer = None
        if niche_config is not None:
            self.niche_featurizer = NicheFeaturizer(**niche_config).fit(adata_list, label_key)
            for adata in adata_list:
                adata.obsm['niche_cond'] = self.niche_featurizer.transform(adata, spatial_key, label_key)
        
        self._normalize_spatial(adata_list)
        
        self.dataset = DeepSpatialDataset(
            adata_list=adata_list, 
            spatial_key='spatial_norm', 
            z_key='z_norm',
            label_key=label_key,
            n_samples_base=n_samples_base,
            alpha_spatial=alpha_spatial,
            uot_reg=uot_reg,
            uot_tau=uot_tau,
            mode=mode,
            balanced_fraction=balanced_fraction,
            niche_key='niche_cond' if self.niche_featurizer is not None else None,
        )
        if not len(self.dataset):
            raise ValueError("No valid UOT trajectories were sampled; check data and sampling budget.")

        self.categories = pd.Index(self.dataset.label_encoder.classes_) 
        self.train_loader = DataLoader(
            self.dataset, batch_size=batch_size, 
            shuffle=True, num_workers=num_workers
        )

        # Infer dimensions from the first batch
        batch = next(iter(self.train_loader))
        self.gene_dim = batch['g0'].shape[1]
        self.num_classes = batch['c0'].shape[1]

    def _validate_slices(self, adatas, require_order=False):
        """检查基因顺序、标签、物理坐标及单张切片深度，尽早报告输入错误。"""
        if len(adatas) < 2:
            raise ValueError("At least two nonempty observed slices are required.")
        genes = self.gene_names if self.gene_names is not None else adatas[0].var_names.tolist()
        depths = []
        for adata in adatas:
            if not adata.n_obs or not adata.var_names.is_unique or adata.var_names.tolist() != genes:
                raise ValueError("Observed slices must be nonempty and have identical, unique gene names in order.")
            if self.label_key not in adata.obs or adata.obs[self.label_key].isna().any():
                raise ValueError(f"Observed cells need nonmissing {self.label_key!r} labels.")
            if self.categories is not None and not adata.obs[self.label_key].astype(str).isin(self.categories).all():
                raise ValueError("Inference slice contains cell types absent from the trained label vocabulary.")
            xy = np.asarray(adata.obsm[self.spatial_key])
            z = np.asarray(adata.obs[self.z_key], dtype=float)
            if xy.shape != (adata.n_obs, 2) or not np.isfinite(xy).all() or not np.isfinite(z).all():
                raise ValueError("Observed slices require finite N x 2 XY coordinates and finite Z values.")
            if not np.allclose(z, z[0], rtol=0, atol=1e-8):
                raise ValueError("Each input slice must have one physical Z coordinate.")
            depths.append(z[0])
        if require_order and not np.all(np.diff(depths) > 0):
            raise ValueError("Pass observed slices in strictly increasing physical Z order.")

    def _prepare_inference_slices(self, adatas):
        """用训练统计量变换可见切片；新切片不能重新拟合归一化或程序尺度。"""
        if self.module is None or self.spatial_stats is None:
            raise ValueError("Build or load a trained model before reconstruction.")
        self._validate_slices(adatas, require_order=True)
        for adata in adatas:
            xy = np.asarray(adata.obsm[self.spatial_key]).copy()
            if not np.issubdtype(xy.dtype, np.floating):
                xy = xy.astype(np.float64)
            for column, axis in enumerate(('x', 'y')):
                xy[:, column] = (xy[:, column] - self.spatial_stats[axis + '_min']) / self.spatial_stats[axis + '_range']
            adata.obsm['spatial_norm'] = xy
            adata.obs['z_norm'] = (float(adata.obs[self.z_key].iloc[0]) - self.spatial_stats['z_min']) / self.spatial_stats['z_range']
            if self.niche_featurizer is not None:
                adata.obsm['niche_cond'] = self.niche_featurizer.transform(adata, self.spatial_key, self.label_key)

    def build_model(self, 
                    patch_size: int = 8,       
                    hidden_size: int = 256,
                    depth: int = 6,
                    num_heads: int = 8,
                    mlp_ratio: float = 4.0,
                    path_type: str = "Linear",
                    lr: float = 2e-4,
                    weight_decay: float = 1e-5,
                    lambda_g: float = 0.1,  
                    lambda_c: float = 10.0,
                    sampling_method: str = "dopri5", 
                    atol: float = 1e-5, 
                    rtol: float = 1e-5,
                    bidirectional_training: bool = False):
        """
        Instantiates the GiT network architecture and Flow Matching logic.
        
        Args:
            patch_size: Tokenization patch size for spatial coordinates.
            hidden_size: Transformer embedding dimension.
            depth: Number of transformer layers.
            num_heads: Number of attention heads.
            mlp_ratio: Expansion ratio for MLP layers.
            path_type: Probability path type for Flow Matching.
            lr: Learning rate for training.
            weight_decay: L2 regularization weight.
            lambda_g: Loss weight for gene expression reconstruction.
            lambda_c: Loss weight for cell type classification.
            sampling_method: ODE solver for inference (e.g., 'dopri5', 'euler').
            atol: Absolute tolerance for ODE solver.
            rtol: Relative tolerance for ODE solver.
            bidirectional_training: 随机交换训练端点及源条件，覆盖两个生成方向。
        """
        if self.gene_dim is None or self.num_classes is None:
            raise ValueError("Dimensions unknown. Call `setup_data()` first.")

        self.model_config = {
            "patch_size": patch_size,
            "hidden_size": hidden_size,
            "depth": depth,
            "num_heads": num_heads,
            "mlp_ratio": mlp_ratio,
            "niche_dim": len(self.niche_featurizer.feature_names) if self.niche_featurizer else 0,
        }
        
        self.train_config = {
            "path_type": path_type,
            "prediction": "velocity",
            "train_eps": 0.02,
            "sample_eps": 0.02,
            "ema_decay": 0.999,
            "lr": lr,
            "weight_decay": weight_decay,
            "lambda_g": lambda_g,
            "lambda_c": lambda_c,
            "sampling_method": sampling_method,
            "atol": atol,
            "rtol": rtol,
            "bidirectional_training": bidirectional_training,
        }

        self.model = GiT(
            gene_dim=self.gene_dim,
            num_classes=self.num_classes,
            **self.model_config
        )

        self.module = DeepSpatialModule(self.train_config, self.model)

    def _save_config(self, save_dir: str):
        """Internal helper to persist metadata and configurations."""
        os.makedirs(save_dir, exist_ok=True)
        config = {
            'gene_dim': self.gene_dim,
            'num_classes': self.num_classes,
            'spatial_stats': self.spatial_stats,
            'categories': self.categories.tolist() if self.categories is not None else None,
            'spatial_key': self.spatial_key,
            'z_key': self.z_key,
            'label_key': self.label_key,
            'model_config': self.model_config,
            'train_config': self.train_config,
            'data_config': self.data_config,
            'gene_names': self.gene_names,
            'niche_featurizer': self.niche_featurizer.to_dict() if self.niche_featurizer else None,
            'trajectory_sampling': self.dataset.sampling_stats if self.dataset is not None else [],
        }
        with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    def load_checkpoint(self, ckpt_path: str, config_path: str = None, sampling_method: str = None):
        """
        Loads model weights and metadata for inference or resuming.
        
        Args:
            ckpt_path: Path to the `.ckpt` file.
            config_path: Path to `config.json`. Defaults to the same folder as ckpt_path.
            sampling_method: Overrides the ODE solver for this inference session.
        """
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            
        if config_path is None:
            config_path = os.path.join(os.path.dirname(ckpt_path), 'config.json')
            
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.gene_dim = config['gene_dim']
            self.num_classes = config['num_classes']
            self.spatial_stats = config['spatial_stats']
            self.spatial_key = config.get('spatial_key', 'spatial')
            self.z_key = config.get('z_key', 'z_coord')
            self.label_key = config['label_key']
            self.categories = pd.Index(config['categories'])
            self.model_config = config.get('model_config', {})
            self.train_config = config.get('train_config', {})
            self.data_config = config.get('data_config', {})
            self.source_sampling = self.data_config.get('source_sampling', 'random')
            self.gene_names = config.get('gene_names')
            niche_state = config.get('niche_featurizer')
            self.niche_featurizer = NicheFeaturizer.from_dict(niche_state) if niche_state else None
        else:
             raise ValueError("Metadata 'config.json' not found. Cannot rebuild model.")

        # 模型参数与运输配置分别恢复，避免将 prediction/train_eps 等误传给 build_model。
        if sampling_method is not None:
            self.train_config['sampling_method'] = sampling_method
        self.model = GiT(gene_dim=self.gene_dim, num_classes=self.num_classes, **self.model_config)
        self.module = DeepSpatialModule(self.train_config, self.model)
            
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)
        self.module.load_state_dict(state_dict)

    def save_model(self, save_dir):
        """保存可独立加载的模型权重及训练期生态位元数据。"""
        if self.module is None:
            raise ValueError("No model is available to save.")
        self._save_config(save_dir)
        torch.save({'state_dict': self.module.state_dict()}, os.path.join(save_dir, 'model.ckpt'))

    def fit(self, 
            max_epochs: int = 100, 
            save_dir: str = "./checkpoints", 
            accelerator: str = 'auto', 
            devices: str = 'auto', 
            save_ckpt: bool = False, 
            resume_ckpt_path: str = None):
        """
        Executes the training loop using PyTorch Lightning.
        
        Args:
            max_epochs: Max training epochs.
            save_dir: Directory to save checkpoints and metadata.
            accelerator: Hardware accelerator ('auto', 'gpu', 'cpu', 'mps').
            devices: Number of devices or indices ('auto', 1, [0, 1]).
            save_ckpt: Whether to save progress.
            resume_ckpt_path: Path to resume full training state.
        """
        if save_ckpt:
            self._save_config(save_dir)
            
        callbacks = []
        if save_ckpt:
            callbacks.append(ModelCheckpoint(
                dirpath=save_dir, 
                monitor="loss",
                filename="deepspatial-{epoch:02d}-{loss:.4f}",
                save_top_k=1, 
                save_last=True,
                mode="min"
            ))

        trainer = pl.Trainer(
            max_epochs=max_epochs, 
            accelerator=accelerator, 
            devices=devices,
            callbacks=callbacks,
            enable_checkpointing=save_ckpt,
            logger=CSVLogger(save_dir, name='logs') if save_ckpt else False,
        )
        trainer.fit(self.module, self.train_loader, ckpt_path=resume_ckpt_path)

    def _restore_3d_physical_coords(self, adata_3d: ad.AnnData) -> ad.AnnData:
        """Internal helper to map [0,1] coordinates back to physical scale."""
        if self.spatial_stats is None:
            raise ValueError("Physical stats missing. Ensure setup_data or load_checkpoint was called.")
        
        # Restore XY in obsm[spatial_key]
        adata_3d.obsm[self.spatial_key][:, 0] = adata_3d.obsm[self.spatial_key][:, 0] * self.spatial_stats['x_range'] + self.spatial_stats['x_min']
        adata_3d.obsm[self.spatial_key][:, 1] = adata_3d.obsm[self.spatial_key][:, 1] * self.spatial_stats['y_range'] + self.spatial_stats['y_min']
        
        # Restore Z in obs[z_key]
        adata_3d.obs[self.z_key] = adata_3d.obs[self.z_key] * self.spatial_stats['z_range'] + self.spatial_stats['z_min']
        
        adata_3d.obs_names = [f"cell_3d_z{z:.4f}_{i}" for i, z in enumerate(adata_3d.obs[self.z_key])]
        adata_3d.obsm['spatial_3d'] = np.column_stack([
            adata_3d.obsm[self.spatial_key], adata_3d.obs[self.z_key].to_numpy(),
        ])
        return adata_3d

    @torch.no_grad()
    def reconstruct_between_slices(self, 
                                   adata0: ad.AnnData, 
                                   adata1: ad.AnnData, 
                                   thickness: float,
                                   steps: int = 20, 
                                   chunk_size: int = 2048,
                                   device: str = "auto",
                                   source_sampling: str = None,
                                   save_intermediates: bool = False) -> ad.AnnData:
        """
        Generates a 3D volume segment between two specific AnnData slices.

        Args:
            adata0: Source slice AnnData (must contain 'spatial_norm' and 'z_norm').
            adata1: Target slice AnnData (must contain 'spatial_norm' and 'z_norm').
            steps: Number of integration steps for the ODE solver.
            thickness: Physical distance (um) between generated cells, controlling density.
            chunk_size: Batch size for ODE integration to manage VRAM usage.
            device: Computing device ('auto', 'cuda', 'cpu', or specific 'cuda:n').

        Returns:
            An AnnData object containing the interpolated 3D segment in physical coordinates.
        """
        if not np.isfinite(thickness) or thickness <= 0 or steps < 2 or chunk_size < 1:
            raise ValueError("Use positive thickness/chunk_size and at least two ODE output steps.")
        self._prepare_inference_slices([adata0, adata1])
        source_sampling = source_sampling or self.source_sampling
        if source_sampling not in ('random', 'stratified'):
            raise ValueError("source_sampling must be 'random' or 'stratified'.")
        # Device management
        if device == "auto":
            dev = self.module.device if self.module.device.type != 'cpu' else \
                  torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            dev = torch.device(device)

        if self.module.device != dev:
            self.module.to(dev)
        
        self.module.eval()
        
        # Extract features and normalized Z coordinates
        x0, z0, g0, c0, x1, z1, g1, c1, target_cells, total_cells = self._setup_and_extract(
            adata0, adata1, thickness, dev
        )

        # Execute chunked ODE integration
        mix_data = self._generate_and_prune_optimized(
            x0, g0, c0, x1, g1, c1, z0, z1, steps, 
            target_cells, total_cells, dev, chunk_size,
            niche0=adata0.obsm.get('niche_cond') if self.niche_featurizer else None,
            niche1=adata1.obsm.get('niche_cond') if self.niche_featurizer else None,
            source_sampling=source_sampling, save_intermediates=save_intermediates,
        )

        # Assemble and restore to physical coordinates
        adata_segment = self._assemble_fast_anndata(adata0, adata1, mix_data)
        return self._restore_3d_physical_coords(adata_segment)

    @torch.no_grad()
    def reconstruct_full_volume(self, 
                                adata_list: list,
                                thickness: float, 
                                steps: int = 100, 
                                chunk_size: int = 2048,
                                device: str = "auto",
                                source_sampling: str = None,
                                save_intermediates: bool = False) -> ad.AnnData:
        """
        High-level API to reconstruct the entire 3D volume from a list of slices.

        Args:
            adata_list: Ordered list of AnnData slices.
            thickness: Target physical distance (um) between cells in the Z-axis.
            steps: Number of ODE integration steps per gap.
            chunk_size: Processing batch size to prevent OOM on large datasets.
            device: Target device for the entire reconstruction process.

        Returns:
            A single merged AnnData object representing the continuous 3D volume.
        """
        from tqdm import tqdm
        
        segment_list = []
        num_pairs = len(adata_list) - 1
        
        if num_pairs < 1:
            raise ValueError("adata_list must contain at least 2 slices.")
        
        # Initialize progress bar
        pbar = tqdm(range(num_pairs), desc="DeepSpatial: 3D Reconstruct", unit="gap")
        
        for i in pbar:
            ad0, ad1 = adata_list[i], adata_list[i+1]
            
            # Display current physical range in progress bar
            z_start = ad0.obs[self.z_key].iloc[0]
            z_end = ad1.obs[self.z_key].iloc[0]
            pbar.set_postfix({"range": f"{z_start:.1f}-{z_end:.1f}um"})

            # Generate segment
            segment = self.reconstruct_between_slices(
                adata0=ad0,
                adata1=ad1,
                steps=steps,
                thickness=thickness,
                chunk_size=chunk_size,
                device=device,
                source_sampling=source_sampling,
                save_intermediates=save_intermediates,
            )
            
            segment_list.append(segment)

        # Concatenate all generated segments
        full_volume = ad.concat(segment_list, join='outer', uns_merge='first', index_unique='-')
        
        # Inherit metadata from the reference slice
        if hasattr(adata_list[0], 'uns'):
            full_volume.uns.update(adata_list[0].uns)
        
        return full_volume

    # =========================================================================
    # Internal Computation Helpers
    # =========================================================================

    def _setup_and_extract(self, adata0, adata1, thickness, dev):
        """Calculates sampling density and maps normalized input to device."""
        avg_n_ref = (adata0.n_obs + adata1.n_obs) / 2

        z0, z1 = adata0.obs[self.z_key].iloc[0], adata1.obs[self.z_key].iloc[0]
        physical_gap = abs(z1 - z0)
        
        target_cells = max(1, int(avg_n_ref * (physical_gap / thickness)))
        total_cells = int(target_cells)

        def extract(adata):
            # Extract from normalized keys
            '提取归一化位置、表达和类型，供分块生成使用。'
            x = torch.tensor(adata.obsm['spatial_norm'], dtype=torch.float32, device=dev)
            z = adata.obs['z_norm'].iloc[0]
            g_arr = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
            g = torch.tensor(g_arr, dtype=torch.float32, device=dev)
            c_idx = torch.tensor(pd.Categorical(adata.obs[self.label_key], categories=self.categories).codes.astype(np.int64), device=dev)
            return x, z, g, c_idx
            
        x0, z0, g0, c0 = extract(adata0)
        x1, z1, g1, c1 = extract(adata1)
        
        return x0, z0, g0, c0, x1, z1, g1, c1, target_cells, total_cells

    def _generate_and_prune_optimized(self, x0, g0, c0, x1, g1, c1, z0, z1, steps, target_cells, total_cells, dev, chunk_size,
                                     niche0=None, niche1=None, source_sampling='random', save_intermediates=False):
        """保持原深度/方向分布，按开关抽取来源并分块积分、保留追溯结果。"""
        N0, N1 = float(x0.shape[0]), float(x1.shape[0])
        u = torch.rand(total_cells, device=dev)
        
        # Inverse transform sampling for time distribution
        if abs(N0 - N1) < 1e-5:
            t_vals = u
        else:
            t_vals = (-N0 + torch.sqrt(N0**2 * (1 - u) + N1**2 * u)) / (N1 - N0)

        target_zs = t_vals * (z1 - z0) + z0
        is_fwd = torch.rand(total_cells, device=dev) > t_vals
        fwd_ids, bwd_ids = torch.where(is_fwd)[0], torch.where(~is_fwd)[0]

        if source_sampling == 'stratified':
            src_fwd = torch.as_tensor(stratified_source_indices(c0.cpu().numpy(), len(fwd_ids)), device=dev)
            src_bwd = torch.as_tensor(stratified_source_indices(c1.cpu().numpy(), len(bwd_ids)), device=dev)
        else:
            src_fwd = torch.randint(0, x0.shape[0], (len(fwd_ids),), device=dev)
            src_bwd = torch.randint(0, x1.shape[0], (len(bwd_ids),), device=dev)

        final_x = torch.zeros((total_cells, 2), device=dev)
        final_g = torch.zeros((total_cells, g0.shape[1]), device=dev)
        final_c = torch.zeros(total_cells, device=dev, dtype=torch.long)
        continuous_c = torch.zeros((total_cells, self.num_classes), device=dev) if save_intermediates else None

        def process_direction(is_forward, src_indices, child_ids, x_ref, g_ref, c_ref, z_start, z_end, niche_ref):
            """同一 parent 的固定源条件贯穿完整轨迹，多个目标深度复用该轨迹。"""
            if len(child_ids) == 0: return
            unique_parents, inverse_indices = torch.unique(src_indices, return_inverse=True)
            c_onehot_ref = torch.nn.functional.one_hot(c_ref, num_classes=self.num_classes).float()

            for i in range(0, len(unique_parents), chunk_size):
                end_idx = min(i + chunk_size, len(unique_parents))
                chunk_parents = unique_parents[i:end_idx]
                mask_in_chunk = (inverse_indices >= i) & (inverse_indices < end_idx)
                chunk_child_ids = child_ids[mask_in_chunk]
                if len(chunk_child_ids) == 0: continue
                
                local_parent_idx = inverse_indices[mask_in_chunk] - i
                batch = {
                    'x0': x_ref[chunk_parents], 'g0': g_ref[chunk_parents], 'c0': c_onehot_ref[chunk_parents],
                    'z0': torch.full((len(chunk_parents), 1), z_start, device=dev),
                    'z1': torch.full((len(chunk_parents), 1), z_end, device=dev),
                    'delta_z': torch.full((len(chunk_parents), 1), z_end - z_start, device=dev)
                }
                if niche_ref is not None:
                    batch['niche_cond'] = torch.as_tensor(niche_ref[chunk_parents.cpu().numpy()], device=dev)
                res = self.module.sample(batch, mode="ODE", steps=steps)
                
                if not is_forward:
                    res['x_traj'] = torch.flip(res['x_traj'], dims=[0])
                    res['g_traj'] = torch.flip(res['g_traj'], dims=[0])
                    res['c_traj_discrete'] = torch.flip(res['c_traj_discrete'], dims=[0])
                    res['c_traj_continuous'] = torch.flip(res['c_traj_continuous'], dims=[0])

                chunk_t_vals = t_vals[chunk_child_ids] * (steps - 1)
                idx_low = chunk_t_vals.long()
                idx_high = torch.clamp(idx_low + 1, max=steps - 1)
                w_high = (chunk_t_vals - idx_low.float()).unsqueeze(1)

                final_x[chunk_child_ids] = (1 - w_high) * res['x_traj'][idx_low, local_parent_idx] + w_high * res['x_traj'][idx_high, local_parent_idx]
                final_g[chunk_child_ids] = (1 - w_high) * res['g_traj'][idx_low, local_parent_idx] + w_high * res['g_traj'][idx_high, local_parent_idx]
                final_c[chunk_child_ids] = res['c_traj_discrete'][torch.clamp(chunk_t_vals.round().long(), max=steps-1), local_parent_idx]
                if continuous_c is not None:
                    continuous_c[chunk_child_ids] = (
                        (1 - w_high) * res['c_traj_continuous'][idx_low, local_parent_idx]
                        + w_high * res['c_traj_continuous'][idx_high, local_parent_idx]
                    )

                del res, batch
                torch.cuda.empty_cache()

        process_direction(True, src_fwd, fwd_ids, x0, g0, c0, z0, z1, niche0)
        process_direction(False, src_bwd, bwd_ids, x1, g1, c1, z1, z0, niche1)

        
        fwd_src_keep, bwd_src_keep = src_fwd, src_bwd
        fwd_keep_indices, bwd_keep_indices = fwd_ids, bwd_ids
        fwd_mask = is_fwd
        before_sparsification = final_g.clone() if save_intermediates else None

        # Sparsity enforcement
        nnz_parent = torch.zeros(target_cells, device=dev, dtype=torch.long)
        nnz_parent[fwd_mask] = (g0 > 0).sum(dim=1)[fwd_src_keep]
        nnz_parent[~fwd_mask] = (g1 > 0).sum(dim=1)[bwd_src_keep]
        max_k = int(nnz_parent.max().item())

        if max_k > 0:
            for i in range(0, target_cells, chunk_size):
                end_idx = min(i + chunk_size, target_cells)
                g_chunk, nnz_chunk = final_g[i:end_idx], nnz_parent[i:end_idx]
                _, sorted_indices = torch.sort(g_chunk, dim=1, descending=True)
                top_indices = sorted_indices[:, :max_k]
                mask = torch.arange(max_k, device=dev).unsqueeze(0) < nnz_chunk.unsqueeze(1)
                final_mask = torch.zeros_like(g_chunk, dtype=torch.bool).scatter_(1, top_indices, mask)
                g_chunk[~final_mask] = 0.0
                final_g[i:end_idx] = g_chunk
        
        return {
            'x': final_x.cpu().numpy(), 'g': final_g, 'c': final_c.cpu().numpy(), 
            'zs': target_zs.cpu().numpy(), 'cells': target_cells,
            'fwd_src': fwd_src_keep.cpu().numpy(), 'bwd_src': bwd_src_keep.cpu().numpy(),
            'fwd_idx': fwd_keep_indices.cpu().numpy(), 'bwd_idx': bwd_keep_indices.cpu().numpy(),
            'continuous_c': continuous_c.cpu().numpy() if continuous_c is not None else None,
            'before_sparsification': before_sparsification.cpu().numpy() if before_sparsification is not None else None,
            'source_sampling': source_sampling,
        }

    def _assemble_fast_anndata(self, adata0, adata1, mix_data):
        """组装生成结果；源注释只用于追溯，parent_id 明确标识真实来源。"""
        nz_indices = mix_data['g'].nonzero(as_tuple=True)
        res_g_sparse = scipy.sparse.csr_matrix(
            (mix_data['g'][nz_indices].cpu().numpy(), (nz_indices[0].cpu().numpy(), nz_indices[1].cpu().numpy())), 
            shape=(mix_data['cells'], mix_data['g'].shape[1])
        )
        

        obs_fwd = adata0.obs.iloc[mix_data['fwd_src']].copy()
        obs_fwd['parent_id'] = [f"{float(adata0.obs[self.z_key].iloc[0]):.12g}:{name}" for name in obs_fwd.index]
        obs_fwd['source_cell_id'] = obs_fwd.index.astype(str)
        obs_fwd['source_z'] = float(adata0.obs[self.z_key].iloc[0])
        obs_fwd['source_cell_index'] = mix_data['fwd_src']
        obs_fwd['source_direction'] = 'forward'
        obs_fwd.index = mix_data['fwd_idx']
        obs_bwd = adata1.obs.iloc[mix_data['bwd_src']].copy()
        obs_bwd['parent_id'] = [f"{float(adata1.obs[self.z_key].iloc[0]):.12g}:{name}" for name in obs_bwd.index]
        obs_bwd['source_cell_id'] = obs_bwd.index.astype(str)
        obs_bwd['source_z'] = float(adata1.obs[self.z_key].iloc[0])
        obs_bwd['source_cell_index'] = mix_data['bwd_src']
        obs_bwd['source_direction'] = 'backward'
        obs_bwd.index = mix_data['bwd_idx']
        
        mixed_obs = pd.concat([obs_fwd, obs_bwd]).sort_index()
        mixed_obs['source_cell_class'] = mixed_obs[self.label_key].astype(str)
        mixed_obs['is_observed'] = False
        mixed_obs['annotation_origin'] = 'source_cell'
        mixed_obs[self.z_key] = mix_data['zs'] 
        mixed_obs[self.label_key] = pd.Categorical.from_codes(mix_data['c'], categories=self.categories)

        mixed_obs.index = mixed_obs.index.astype(str)

        result = ad.AnnData(
            X=res_g_sparse, 
            obs=mixed_obs, 
            var=adata0.var.copy(), 
            obsm={self.spatial_key: mix_data['x']}
        )
        if mix_data['before_sparsification'] is not None:
            result.layers['expression_before_sparsification'] = mix_data['before_sparsification']
            result.obsm['cell_class_continuous'] = mix_data['continuous_c']
        result.uns['hm_reconstruction'] = {
            'source_sampling': mix_data['source_sampling'],
            'generated_cells': int(result.n_obs),
            'unique_parents': int(result.obs['parent_id'].nunique()),
            'niche_conditioning': self.niche_featurizer is not None,
        }
        return result
