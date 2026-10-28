"""Minimal standalone check: python tests/test_visualize_reconstruction.py."""
from pathlib import Path
import sys
import tempfile

import h5py
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from visualize_reconstruction import read_coordinates, sample_indices, virtual_slices


def main():
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'tiny.h5ad'
        with h5py.File(path,'w') as f:
            f['obsm/spatial']=np.array([[0,1],[2,3],[4,5],[6,7]],dtype=float)
            f['obs/z_coord']=[0.,10.,20.,30.]
            labels=f.create_group('obs/cell_class')
            labels.attrs['encoding-type']='categorical'
            labels['categories']=np.array(['common','rare'],dtype=h5py.string_dtype())
            labels['codes']=[0,0,0,1]
        coords,labels=read_coordinates(path,'cell_class')
        assert coords.shape==(4,3) and labels.tolist()==['common','common','common','rare']
        idx=sample_indices(labels,2,42)
        assert len(idx)==2 and set(labels[idx])=={'common','rare'}
        assert np.array_equal(idx,sample_indices(labels,2,42))
        assert len(sample_indices(labels,0,42))==4
        try:
            sample_indices(labels,1,42)
        except ValueError:
            pass
        else:
            raise AssertionError('Display budget smaller than class count must be rejected.')
        fig=virtual_slices(coords,labels,['common','rare'],{'common':'blue','rare':'red'},10)
        assert sum(len(trace.x) for trace in fig.frames[0].data)==1  # Z=0 endpoint included
        assert sum(len(trace.x) for trace in fig.frames[-1].data)==1  # Z=30 endpoint included
        with h5py.File(path,'a') as f:
            f['obs/cell_class/codes'][3]=-1
        try:
            read_coordinates(path,'cell_class')
        except ValueError:
            pass
        else:
            raise AssertionError('Missing labels must be rejected.')
    print('Visualization checks passed.')


if __name__=='__main__':
    main()
