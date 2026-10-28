"""Run: python tests/test_compare_spatialz.py"""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from compare_spatialz import expression_metrics, align_genes, coverage


def main():
    result=expression_metrics([0.,1.,2.],[0.,2.,4.])
    assert np.isclose(result['MSE'],5/3) and np.isclose(result['MAE'],1) and np.isclose(result['PCC'],1)
    assert np.isnan(expression_metrics([0,0],[1,1])['PCC'])
    truth={'genes':np.array(['a','b']),'labels':np.array(['rareA','rareB']),
           'xy':np.array([[0.,0.],[200.,0.]])}
    pred={'genes':np.array(['b','a']),'x':np.array([[10.,20.]]),
          'labels':np.array(['rareA']),'xy':np.array([[50.,0.]])}
    assert np.array_equal(align_genes(truth,pred),[[20,10]])
    table,summary=coverage(truth,pred,{'common':1000,'rareA':2,'rareB':3})
    assert summary['rare_classes']==2 and summary['rare_missing_classes']==1
    assert summary['rare_recall_50um']==.5 and summary['rare_recall_100um']==.5
    missing=table.set_index('cell_class').loc['rareB']
    assert missing.recall_50um==0 and missing.recall_100um==0  # must stay in macro denominator
    bad={**pred,'genes':np.array(['a','c'])}
    try:
        align_genes(truth,bad)
    except ValueError:
        pass
    else:
        raise AssertionError('Different gene sets must not be silently intersected.')
    print('Unified metric checks passed.')


if __name__=='__main__':
    main()
