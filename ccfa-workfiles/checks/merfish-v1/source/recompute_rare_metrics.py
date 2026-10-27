"""Re-evaluate saved original/V1/S predictions with the notebook's inclusive rarity policy."""
import ast
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
DEST=ROOT/'HM_DeepSpital/outputs/s_holdouts_1_2_3/analysis'
os.environ['MPLCONFIGDIR']=str(DEST/'.matplotlib')

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from analyze_v1 import md_table

METHODS=('DeepSpatial','V1 reg=0.05','Experiment S')
RADII=(25,50,100)


def evaluation_functions():
    """Read the canonical notebook policy without executing any training cells."""
    nb=json.loads((ROOT/'merfish_leave_one_slice_out.ipynb').read_text(encoding='utf-8'))
    config=ast.parse(next(''.join(c['source']) for c in nb['cells'] if c.get('id')=='configuration'))
    threshold=next(float(ast.literal_eval(node.value)) for node in config.body if isinstance(node,ast.Assign)
                   and any(isinstance(target,ast.Name) and target.id=='RARE_TRAIN_FRACTION' for target in node.targets))
    source=next(''.join(c['source']) for c in nb['cells'] if c.get('id')=='metric-helpers')
    scope=dict(np=np,pd=pd,ad=ad,sp=sp,cKDTree=cKDTree,LABEL_KEY='cell_class',
               RARE_TRAIN_FRACTION=threshold,COVERAGE_RADII_UM=RADII)
    exec(compile(source,'notebook:metric-helpers','exec'),scope)
    return scope['evaluate_holdout'],threshold


def main():
    evaluate,threshold=evaluation_functions()
    assert threshold==.05,'This refresh is for the requested <=5% policy.'
    # Exercise the actual evaluator at exactly 5%, above it, and unseen training support.
    a=ad.AnnData(X=np.array([[1.,0,2],[0,2,1],[2,1,0],[1,2,0]]),
                 obs=pd.DataFrame({'cell_class':['boundary','above','common','unseen']},index=['a','b','c','d']),
                 obsm={'spatial':np.array([[0.,0],[10,0],[20,0],[30,0]])})
    test,_=evaluate(a,a,pd.Series({'boundary':5,'above':6,'common':89}))
    flags=test.set_index('cell_class').is_rare_train
    assert flags['boundary'] and not flags['above'] and not flags['unseen']
    original={i:ad.read_h5ad(next((ROOT/'data/merfish_mouse_hypothalamus').rglob(f'merfish_{i}.h5ad'))) for i in range(5)}
    old_s=pd.read_csv(DEST/'cell_type_metrics.csv')
    old_v1=pd.read_csv(ROOT/'HM_DeepSpital/outputs/v1_holdouts_1_2_3/analysis/cell_type_metrics.csv')
    summaries,tables,class_rows=[],[],[]
    for fold in (1,2,3):
        train=pd.concat([original[i].obs.cell_class.astype(str) for i in range(5) if i!=fold]).value_counts()
        for label,n in train.items():
            class_rows.append(dict(heldout_id=fold,cell_class=label,train_n=int(n),train_total=int(train.sum()),
                                   train_fraction=float(n/train.sum()),is_rare_train=bool(0<n and n/train.sum()<=threshold)))
        local={}
        for method in METHODS:
            if method=='DeepSpatial':
                path=ROOT/f'output/merfish_holdout/holdout_{fold}/predicted_holdout.h5ad'
                pred=ad.read_h5ad(path)
                previous=old_v1[(old_v1.heldout_id==fold)&(old_v1.run==method)]
            else:
                group='v1_holdouts_1_2_3' if method=='V1 reg=0.05' else 's_holdouts_1_2_3'
                volume=ad.read_h5ad(ROOT/f'HM_DeepSpital/outputs/{group}/holdout_{fold}/generated.h5ad')
                pred=volume[abs(volume.obs.z_coord-fold*50)<=5].copy()
                previous=old_s[(old_s.heldout_id==fold)&(old_s.run==method)]
                del volume
            table,summary=evaluate(original[fold],pred,train)
            # Changing the rare-class definition must not change any per-class measurements.
            old=previous.set_index('cell_class').sort_index()
            now=table.set_index('cell_class').sort_index()
            assert old.index.equals(now.index)
            columns=['train_n','train_fraction','truth_n','predicted_n','mean_log_expression_pcc',
                     *[f'coverage_{kind}_{r}um' for r in RADII for kind in ('recall','precision')]]
            np.testing.assert_allclose(old[columns],now[columns],atol=1e-7,rtol=1e-7,equal_nan=True)
            rare=table[table.is_rare_train&(table.truth_n>0)]
            summary.update(heldout_id=fold,run=method,rare_train_fraction=threshold,rare_fraction_inclusive=True,
                           rare_true_n=int(rare.truth_n.sum()),rare_predicted_n=int(rare.predicted_n.sum()),
                           rare_truth_fraction=float(rare.truth_n.sum()/original[fold].n_obs),
                           rare_expression_classes=int(rare.mean_log_expression_pcc.notna().sum()),
                           rare_present_class_expression_pcc=float(rare.mean_log_expression_pcc.mean()))
            for radius in RADII:
                summary[f'rare_covered_truth_n_{radius}um']=int(np.rint(rare.truth_n*rare[f'coverage_recall_{radius}um']).sum())
                valid=rare.predicted_n>0
                summary[f'rare_covered_prediction_n_{radius}um']=int(np.rint(rare.loc[valid,'predicted_n']*rare.loc[valid,f'coverage_precision_{radius}um']).sum())
                summary[f'rare_micro_coverage_recall_{radius}um']=summary[f'rare_covered_truth_n_{radius}um']/summary['rare_true_n']
                summary[f'rare_micro_coverage_precision_{radius}um']=summary[f'rare_covered_prediction_n_{radius}um']/summary['rare_predicted_n']
            local[method]=(table,summary)
            tables.append(table.assign(heldout_id=fold,run=method))
        common=set.intersection(*[set(t.loc[t.predicted_n>0,'cell_class']) for t,s in local.values()])
        for table,summary in local.values():
            rare=table[table.is_rare_train&table.cell_class.isin(common)&(table.truth_n>0)]
            summary['rare_common_present_expression_classes']=len(rare)
            summary['rare_common_present_expression_pcc']=float(rare.mean_log_expression_pcc.mean())
            summaries.append(summary)
        print('FOLD',fold,json.dumps(local['Experiment S'][1]),flush=True)
    result=pd.DataFrame(summaries)
    table=pd.concat(tables,ignore_index=True)
    classes=pd.DataFrame(class_rows).sort_values(['heldout_id','train_fraction'])
    metric_cols=[f'rare_{average}_coverage_{kind}_{r}um' for average in ('macro','micro') for kind in ('recall','precision') for r in RADII]
    metric_cols+=['rare_common_present_expression_pcc']
    averages=result.groupby('run')[metric_cols].agg(['mean','std'])
    averages.columns=['_'.join(c) for c in averages.columns]
    pooled=[]
    for method,frame in result.groupby('run'):
        row=dict(run=method,rare_truth_n=int(frame.rare_true_n.sum()),rare_predicted_n=int(frame.rare_predicted_n.sum()))
        for r in RADII:
            row[f'pooled_recall_{r}um']=float(frame[f'rare_covered_truth_n_{r}um'].sum()/frame.rare_true_n.sum())
            row[f'pooled_precision_{r}um']=float(frame[f'rare_covered_prediction_n_{r}um'].sum()/frame.rare_predicted_n.sum())
        pooled.append(row)
    for name,frame in [('rare_metrics',result),('rare_cell_type_metrics',table),('rare_classes',classes),
                       ('rare_metrics_mean_std',averages.reset_index()),('rare_metrics_pooled',pd.DataFrame(pooled))]:
        frame.to_csv(DEST/f'{name}.csv',index=False)
    protocol=dict(threshold=threshold,inclusive=True,abundance_source='pooled cells in each fold visible training slices',
                  unseen_training_classes_excluded_from_rare=True,folds=[1,2,3],evaluation_slab_half_width_um=5,
                  coverage_radii_um=list(RADII),methods=list(METHODS),boundary_5_percent_verified=True,
                  per_class_metrics_unchanged_verified=True)
    (DEST/'rare_metrics_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    make_plot(result)
    write_report(result,table,classes,averages,pd.DataFrame(pooled))
    print('MEANS',averages.to_string(),flush=True)


def make_plot(result):
    plt.rcParams.update({'font.family':'Times New Roman','font.size':11,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for ax,key,title in zip(axes,['rare_macro_coverage_recall_100um','rare_micro_coverage_recall_100um'],
                            ['Macro same-type recall @100 µm (%)','Cell-weighted same-type recall @100 µm (%)']):
        for shift,method,color in [(-.25,METHODS[0],'#8496a4'),(0,METHODS[1],'#357ba8'),(.25,METHODS[2],'#d15e32')]:
            values=result[result.run==method].sort_values('heldout_id')[key].to_numpy()*100
            ax.bar(np.arange(3)+shift,values,width=.24,color=color,label=method)
            for x,y in zip(np.arange(3)+shift,values):ax.text(x,y,f'{y:.1f}',ha='center',va='bottom',fontsize=9)
        ax.set_xticks(range(3),['Holdout 1','Holdout 2','Holdout 3'])
        ax.set_title(title,fontsize=11)
        ax.set_ylim(0,100)
        ax.spines[['right','top']].set_visible(False)
    axes[0].legend(frameon=False,fontsize=9,loc='upper left')
    fig.suptitle('Rare class = visible training abundance ≤5% (11 classes per fold)',fontsize=12)
    for ext in ('png','svg'):fig.savefig(DEST/f'rare_coverage.{ext}',dpi=180)
    plt.close(fig)


def write_report(result,table,classes,averages,pooled):
    active=classes[classes.is_rare_train]
    members=active.groupby('cell_class').agg(minimum_fraction=('train_fraction','min'),maximum_fraction=('train_fraction','max')).reset_index()
    members['原1%名单']=np.where(members.maximum_fraction<.01,'原有','新增')
    members['训练丰度范围']=members.apply(lambda r:f'{r.minimum_fraction:.3%}–{r.maximum_fraction:.3%}',axis=1)
    rows=[]
    metrics=[('稀有类宏平均覆盖 @25 µm ↑','rare_macro_coverage_recall_25um','.2%'),
             ('稀有类宏平均覆盖 @50 µm ↑','rare_macro_coverage_recall_50um','.2%'),
             ('稀有类宏平均覆盖 @100 µm ↑','rare_macro_coverage_recall_100um','.2%'),
             ('稀有类细胞加权覆盖 @100 µm ↑','rare_micro_coverage_recall_100um','.2%'),
             ('稀有类细胞加权 precision @100 µm ↑','rare_micro_coverage_precision_100um','.2%'),
             ('共同存在稀有类均值表达 PCC ↑','rare_common_present_expression_pcc','.4f')]
    for title,key,fmt in metrics:
        rows.append({'指标':title,**{method:format(averages.loc[method,key+'_mean'],fmt) for method in METHODS}})
    fold_rows=[]
    for fold in (1,2,3):
        row={'留出':str(fold)}
        for method in METHODS:
            a=result[(result.heldout_id==fold)&(result.run==method)].iloc[0]
            row[method]=f'{a.rare_macro_coverage_recall_100um:.2%}'
        fold_rows.append(row)
    cohort_rows=[]
    for label,selector in [('原 <1% 六类',table.train_fraction<.01),('新增 1%–5% 五类',table.train_fraction>=.01),('新 ≤5% 十一类',pd.Series(True,index=table.index))]:
        row={'类别集合':label}
        for method in METHODS:
            a=table[(table.run==method)&table.is_rare_train&selector]
            row[method]=f'{a.coverage_recall_100um.mean():.2%}'
        cohort_rows.append(row)
    rare_rows=[]
    for label in sorted(active.cell_class.unique()):
        row={'类别':label}
        for method in METHODS:
            a=table[(table.cell_class==label)&(table.run==method)]
            row[method]=' / '.join(f'{x:.1%}' for x in a.sort_values('heldout_id').coverage_recall_100um)
        rare_rows.append(row)
    s=averages.loc['Experiment S'];v=averages.loc['V1 reg=0.05']
    delta=100*(s.rare_macro_coverage_recall_100um_mean-v.rare_macro_coverage_recall_100um_mean)
    report=f'''# 稀有细胞定义更新：可见训练丰度 ≤5%

本文件为当前稀有类评价口径。旧 `report.md` 及其图表保留原 <1% 的历史分析，不与新口径混用。

## 定义与复算范围

- 每折将可见四张训练切片的细胞合并，计算 `该 cell_class 的训练数量 / 全部可见训练细胞数`。
- 满足 `train_n > 0` 且丰度 **≤0.05** 的类型定义为稀有类，包含恰好 5% 的边界。训练未见类型仍单列，不混入稀有组。
- 不使用隐藏真值丰度，也不使用预测生成比例选择名单；每折各方法使用同一名单。
- 逐类最近邻仍要求相同原始 cell_class，不把十一类合并后寻找任意“稀有细胞”邻居。
- 复算覆盖原版 DeepSpatial、V1 reg=.05、实验 S 的留出 1/2/3；评价仍为 Z±5 µm，半径仍为 25/50/100 µm。
- 从原始 H5AD 重新计算，并确认除 `is_rare_train` 外逐类计数、训练丰度、覆盖及表达 PCC 与先前结果在 1e-7 容差内一致。模型、生成结果和训练参数不变，无需重训。
- 已在实际 evaluator 上验证 5/100 被纳入、6/100 被排除；训练未见类型不纳入。

## 新名单：三折均为 11 类

{md_table(members[['cell_class','训练丰度范围','原1%名单']].rename(columns={'cell_class':'类别'}))}

新增 Endothelial 3、Microglia、OD Immature 1、OD Mature 2、Ependymal。Endothelial 1 在三折训练丰度约 5.22%–5.31%，仍不计入；Astrocyte、Excitatory、Inhibitory 也不计入。

三个真实评价薄层中的这十一类分别包含 1035、1026、1034 个细胞，共 3095 个。原六类共 372 个，新增五类共 2723 个，占新稀有组真实细胞约 88%。

## 三折等权平均

{md_table(pd.DataFrame(rows))}

各切片间标准差见 `rare_metrics_mean_std.csv`，不是多种子标准差。宏平均对十一类等权；细胞加权覆盖按各类真实数量加权，precision 按预测数量加权。

V1 留出 1 缺少 OD Mature 3，召回仍按 0 计入。该类表达 PCC 未定义，为公平比较，表达表仅对三方法共同有预测的稀有类取平均，分母依次为 10、11、11 类。原协议的宏 precision 对零预测类未定义，所以主表使用细胞加权 precision。

![新口径覆盖率](rare_coverage.png)

## 100 µm 宏平均覆盖率分折结果

{md_table(pd.DataFrame(fold_rows))}

按新定义，S 相对 V1 的 100 µm 宏平均覆盖由 45.86% 到 46.01%，增加约 {delta:.2f} 个百分点；细胞加权覆盖由 76.31% 到 76.80%。但 50 µm 宏平均覆盖从 25.10% 变为 24.07%，且 100 µm 宏平均在留出 1/2 下降、留出 3 上升。当前仍不能据此声称 S 带来跨划分一致的空间收益。

相对原版 DeepSpatial，新 V1 和 S 的宽口径 100 µm 宏平均覆盖均提高；该比较同时包含训练配对、双向训练和生态位等差异，不能单独归因于生态位条件。

## 数值为何高于原报告

以下在同一批逐类空间指标上，仅改变纳入的类别集合，均为 100 µm 宏平均覆盖：

{md_table(pd.DataFrame(cohort_rows))}

新增加的五类比原六类更丰富，空间覆盖也较高。因此扩大稀有类范围后，所有方法的聚合指标均上升；这不是模型在定义改变后变得更准确。应明确标注 ≤5% 口径，避免将 46.01% 与旧六类的 19.90% 直接比较为方法收益。

OD Mature 3 和 OD Immature 2 在实验 S 的三个切片中 100 µm 覆盖仍全部为 0。新定义没有改变这些极稀有类的定位问题；建议保留逐类结果而不只报告总平均。

## 十一类逐类覆盖率

每个单元格按留出 1 / 2 / 3 列出 100 µm 同类覆盖率。

{md_table(pd.DataFrame(rare_rows))}

## 文件与复算

- `rare_metrics.csv`：每折、每方法的整体与新稀有类指标。
- `rare_metrics_mean_std.csv`：三折等权均值和切片间标准差。
- `rare_metrics_pooled.csv`：合并三折真实/预测细胞的加权指标，区别于先算每折再平均。
- `rare_classes.csv`：所有十五类的各折训练丰度、数量及新名单标记。
- `rare_cell_type_metrics.csv`：九个方法×切片组合的逐类指标（135 行），其中稀有类为 99 行。
- `rare_metrics_protocol.json`：阈值、包含边界、训练丰度来源、空间半径及核验记录。
- `rare_coverage.png/svg`：新口径的分折对照图。

默认评价定义已经同步到仓库根目录 `merfish_leave_one_slice_out.ipynb`。仅重算现有结果可在仓库根目录执行：

```bash
python ccfa-workfiles/checks/merfish-v1/source/recompute_rare_metrics.py
```

历史分析脚本中的显式 .01 参数仍用于复现旧报告；当前 ≤5% 分析使用此新入口及 Notebook 默认定义，避免混用。
'''
    assert '\ufffd' not in report
    (DEST/'rare_metrics.md').write_text(report,encoding='utf-8')


if __name__=='__main__':
    main()
