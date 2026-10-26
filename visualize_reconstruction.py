"""Visualize saved H5AD coordinates and cell classes without loading a model.

Run from any directory:
    python visualize_reconstruction.py
    python visualize_reconstruction.py --input output/deepspatial_3d.h5ad --open

Uses the existing environment's numpy, h5py, matplotlib, and plotly packages.
Only coordinates/labels are read; the expression matrix is left on disk.
"""
import argparse
import csv
import html
import json
from pathlib import Path
import webbrowser

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parent


def read_coordinates(path, label_key):
    with h5py.File(path, 'r') as f:
        xy = np.asarray(f['obsm/spatial'], dtype=float)
        z = np.asarray(f['obs/z_coord'], dtype=float)
        node = f[f'obs/{label_key}']
        if isinstance(node, h5py.Group) and node.attrs.get('encoding-type') == 'categorical':
            codes = node['codes'][...]
            categories = node['categories'].asstr()[...]
            if np.any(codes < 0) or np.any(codes >= len(categories)):
                raise ValueError(f'{label_key} contains missing or invalid category codes.')
            labels = categories[codes]
        elif isinstance(node, h5py.Dataset) and h5py.check_string_dtype(node.dtype):
            labels = node.asstr()[...]
        else:
            raise ValueError(f'obs/{label_key} must contain categorical or string labels.')
    if z.ndim != 1 or xy.shape != (len(z), 2) or labels.shape != z.shape or len(z) == 0:
        raise ValueError('Expected nonempty spatial (N, 2), z_coord (N,), and labels (N,).')
    coords = np.column_stack((xy, z))
    if not np.isfinite(coords).all():
        raise ValueError('Coordinates contain NaN or Inf.')
    return coords, labels.astype(str)


def sample_indices(labels, limit, seed):
    """Keep at least one point per class when limiting the global point cloud."""
    if limit == 0 or len(labels) <= limit:
        return np.arange(len(labels))
    categories = np.unique(labels)
    if limit < len(categories):
        raise ValueError('--max-points must be 0 (all) or at least the number of classes.')
    rng = np.random.default_rng(seed)
    anchors = np.array([rng.choice(np.flatnonzero(labels == cat)) for cat in categories])
    remaining = np.setdiff1d(np.arange(len(labels)), anchors)
    return np.sort(np.concatenate((anchors, rng.choice(remaining, limit-len(anchors), replace=False))))


def scatter3d(coords, labels, categories, palette, title, z_stretch):
    fig = go.Figure()
    for cat in categories:
        points = coords[labels == cat]
        fig.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2], mode='markers', name=cat,
            marker=dict(color=palette[cat], size=2, opacity=.7),
            hovertemplate=html.escape(cat)+'<br>X=%{x:.1f}<br>Y=%{y:.1f}<br>Z=%{z:.1f} µm<extra></extra>'))
    spans = np.maximum(np.ptp(coords, axis=0), 1e-6)
    ratio = spans / spans.max()
    ratio[2] *= z_stretch
    fig.update_layout(
        title=title, template='plotly_white', height=660, margin=dict(l=0, r=0, b=0, t=95),
        scene=dict(xaxis_title='X (µm)', yaxis_title='Y (µm)', zaxis_title='Z (µm)',
                   aspectmode='manual', aspectratio=dict(zip(('x', 'y', 'z'), ratio))),
        legend=dict(itemsizing='constant'),
        updatemenus=[dict(type='buttons', direction='right', x=0, y=1.10, buttons=[
            dict(label='Physical aspect', method='relayout', args=[{'scene.aspectmode':'data'}]),
            dict(label=f'Z display ×{z_stretch:g}', method='relayout', args=[{
                'scene.aspectmode':'manual', 'scene.aspectratio':dict(zip(('x','y','z'),ratio))}]),
        ])])
    return fig


def virtual_slices(coords, labels, categories, palette, width):
    """Use all cells for XY slabs; global 3D display sampling does not affect them."""
    centers = np.unique(np.linspace(coords[:, 2].min(), coords[:, 2].max(), 21))
    frames, steps = [], []
    for i, center in enumerate(centers):
        slab = np.abs(coords[:, 2] - center) <= width/2
        traces = []
        for cat in categories:
            points = coords[slab & (labels == cat)]
            traces.append(go.Scattergl(x=points[:, 0], y=points[:, 1], mode='markers', name=cat,
                                      marker=dict(color=palette[cat], size=3, opacity=.75),
                                      hovertemplate=html.escape(cat)+'<br>X=%{x:.1f}<br>Y=%{y:.1f} µm<extra></extra>'))
        title = f'XY slab: Z = {center:.1f} ± {width/2:g} µm | {slab.sum():,} cells'
        frames.append(go.Frame(name=str(i), data=traces, layout=go.Layout(title_text=title)))
        steps.append(dict(label=f'{center:.0f}',method='animate',args=[[str(i)],dict(
            mode='immediate',frame=dict(duration=0,redraw=True),transition=dict(duration=0))]))
    middle = len(frames)//2
    fig = go.Figure(data=frames[middle].data, frames=frames)
    fig.update_layout(title=frames[middle].layout.title, template='plotly_white', height=660,
                      xaxis=dict(title='X (µm)',range=[coords[:,0].min(),coords[:,0].max()]),
                      yaxis=dict(title='Y (µm)',range=[coords[:,1].min(),coords[:,1].max()],
                                 scaleanchor='x',scaleratio=1),
                      legend=dict(itemsizing='constant'),
                      sliders=[dict(active=middle,currentvalue=dict(prefix='Z center (µm): '),steps=steps)])
    return fig


def static_figures(coords, labels, indices, categories, palette, out, width, z_stretch):
    shown, shown_labels = coords[indices], labels[indices]
    colors = [palette[cat] for cat in shown_labels]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'savefig.facecolor':'white'})
    fig = plt.figure(figsize=(13, 7.6),layout='constrained')
    grid = fig.add_gridspec(2,2,height_ratios=[3,1])
    ax = fig.add_subplot(grid[0,0],projection='3d')
    ax.scatter(*shown.T,c=colors,s=.5,alpha=.65,linewidths=0,rasterized=True)
    spans = np.maximum(np.ptp(coords,axis=0),1e-6)
    ax.set_box_aspect((spans[0],spans[1],spans[2]*z_stretch))
    ax.view_init(elev=25,azim=-60)
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
        axis.set_major_locator(MaxNLocator(4))
    ax.set(xlabel='X (um)',ylabel='Y (um)',zlabel='Z (um)',title=f'3D volume | Z display x{z_stretch:g}')
    for position,(a,b) in zip([grid[0,1],grid[1,0],grid[1,1]],[(0,1),(0,2),(1,2)]):
        ax = fig.add_subplot(position)
        ax.scatter(shown[:,a],shown[:,b],c=colors,s=.5,alpha=.6,linewidths=0,rasterized=True)
        ax.set(xlabel=f'{"XYZ"[a]} (um)',ylabel=f'{"XYZ"[b]} (um)',title=f'{"XYZ"[a]}{"XYZ"[b]} projection')
        ax.set_aspect('equal',adjustable='box')
    handles=[Line2D([],[],marker='o',linestyle='',color=palette[cat],label=cat,markersize=5) for cat in categories]
    fig.legend(handles=handles,loc='outside lower center',ncol=5,frameon=False,fontsize=8)
    fig.suptitle(f'Reconstructed cells | {len(coords):,} total, {len(shown):,} displayed')
    fig.savefig(out/'reconstruction_overview.png',dpi=200,bbox_inches='tight',pad_inches=.15)
    plt.close(fig)

    centers = coords[:,2].min()+np.ptp(coords[:,2])*np.array([.25,.5,.75])
    fig,axes=plt.subplots(1,3,figsize=(13,4.6),layout='constrained')
    for ax,center in zip(axes,centers):
        slab=np.abs(coords[:,2]-center)<=width/2
        for cat in categories:
            points=coords[slab & (labels==cat)]
            ax.scatter(points[:,0],points[:,1],s=1.5,c=palette[cat],alpha=.8,linewidths=0)
        ax.set(title=f'Z = {center:.1f} +/- {width/2:g} um\nn = {slab.sum():,}',
               xlabel='X (um)',ylabel='Y (um)',xlim=(coords[:,0].min(),coords[:,0].max()),
               ylim=(coords[:,1].min(),coords[:,1].max()),aspect='equal')
    fig.legend(handles=handles,loc='outside lower center',ncol=5,frameon=False,fontsize=8)
    fig.savefig(out/'virtual_slices.png',dpi=200,bbox_inches='tight',pad_inches=.15)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input',type=Path,default=ROOT/'output/deepspatial_3d.h5ad')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'output/reconstruction_visualization')
    parser.add_argument('--label-key',default='cell_class')
    parser.add_argument('--max-points',type=int,default=60000,help='Global display limit; 0 uses all cells.')
    parser.add_argument('--rare-threshold',type=float,default=.01,help='Rare class fraction in this generated file.')
    parser.add_argument('--slab-width',type=float,default=10,help='Full Z slab width in micrometers.')
    parser.add_argument('--z-stretch',type=float,default=5,help='Display aspect only; coordinates stay unchanged.')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--open',action='store_true',help='Open the generated HTML in a browser.')
    args=parser.parse_args()
    if (args.max_points<0 or not 0<args.rare_threshold<1 or
        not np.isfinite(args.slab_width) or args.slab_width<=0 or
        not np.isfinite(args.z_stretch) or args.z_stretch<=0):
        parser.error('Require max-points >= 0, 0 < rare-threshold < 1, and positive finite widths/scales.')
    coords,labels=read_coordinates(args.input,args.label_key)
    categories,counts=np.unique(labels,return_counts=True)
    palette={cat:to_hex(plt.get_cmap('tab20')(i%20)) for i,cat in enumerate(categories)}
    indices=sample_indices(labels,args.max_points,args.seed)
    rare=categories[counts/len(labels)<args.rare_threshold]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    out=args.output_dir.resolve()
    summary=[dict(cell_class=cat,count=int(n),fraction=float(n/len(labels)),
                  displayed_count=int((labels[indices]==cat).sum()),is_rare=bool(cat in rare))
             for cat,n in zip(categories,counts)]
    with (out/'cell_class_counts.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    metadata=dict(input=str(args.input.resolve()),label_key=args.label_key,n_cells=len(coords),
                  displayed_cells=len(indices),rare_definition=f'fraction < {args.rare_threshold:g} in generated file',
                  rare_classes=rare.tolist(),slab_width_um=args.slab_width,z_display_stretch=args.z_stretch,
                  seed=args.seed,coordinate_min=coords.min(0).tolist(),coordinate_max=coords.max(0).tolist())
    (out/'visualization_parameters.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
    static_figures(coords,labels,indices,categories,palette,out,args.slab_width,args.z_stretch)
    volume=scatter3d(coords[indices],labels[indices],categories,palette,
                     f'3D reconstruction | {len(indices):,} displayed / {len(coords):,} cells',args.z_stretch)
    slabs=virtual_slices(coords,labels,categories,palette,args.slab_width)
    rare_mask=np.isin(labels,rare)
    rare_html='<p>当前阈值下没有稀有类。</p>'
    if rare_mask.any():
        rare_fig=scatter3d(coords[rare_mask],labels[rare_mask],rare,palette,
                          f'Rare classes | all {rare_mask.sum():,} generated cells',args.z_stretch)
        rare_html=rare_fig.to_html(full_html=False,include_plotlyjs=False)
    rows=''.join(f'<tr><td>{html.escape(row["cell_class"])}</td><td>{row["count"]:,}</td>'
                 f'<td>{row["fraction"]:.3%}</td><td>{"是" if row["is_rare"] else ""}</td></tr>' for row in summary)
    document=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>DeepSpatial 重建结果</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;margin:24px auto;padding:0 20px;max-width:1300px;color:#172b3a}}
h1,h2{{color:#172b3a}}p{{max-width:1000px}}table{{border-collapse:collapse;width:700px;max-width:100%}}
th,td{{border:1px solid #dbe2e8;padding:7px 12px;text-align:right}}th:first-child,td:first-child{{text-align:left}}
th{{background:#edf3f7}}.plotly-graph-div{{max-width:100%}}</style></head><body>
<h1>DeepSpatial 三维重建结果</h1>
<p>文件：{html.escape(args.input.name)} · 生成细胞 {len(coords):,} 个 · 细胞类别 {len(categories)} 类。</p>
<p>拖动旋转，滚轮缩放，单击图例隐藏类别，双击图例单独查看类别。坐标单位沿用实验配置的 µm。
三维视图默认仅在显示比例上将 Z 拉伸 {args.z_stretch:g} 倍，可点击 Physical aspect 恢复物理比例；坐标数值不变。
这些图展示生成结果本身，不构成对重建准确性的验证。</p>
<h2>整体三维分布</h2><p>整体点云显示 {len(indices):,} 个细胞，固定随机种子抽样并保留每类至少一个点；下方表格统计全部细胞。</p>
{volume.to_html(full_html=False,include_plotlyjs=True)}
<h2>沿 Z 查看虚拟切片</h2><p>滑动下方 Z 滑块查看厚度为 {args.slab_width:g} µm 的 XY 薄层。
薄层使用全部细胞，端点处只包含文件范围内的部分。每帧标题显示实际细胞数。</p>
{slabs.to_html(full_html=False,include_plotlyjs=False)}
<h2>稀有细胞分布</h2><p>这里按当前生成文件占比小于 {args.rare_threshold:.1%} 筛选，显示全部稀有类生成点；
该口径区别于留出评估中按训练集占比定义稀有类。类别出现或沿 Z 连续分布不等于真实位置恢复准确。</p>
{rare_html}<h2>全部生成细胞的类别统计</h2>
<table><thead><tr><th>细胞类型</th><th>数量</th><th>占比</th><th>稀有类</th></tr></thead><tbody>{rows}</tbody></table>
</body></html>'''
    page=out/'reconstruction_viewer.html'
    page.write_text(document,encoding='utf-8')
    print(f'Input: {args.input.resolve()}\nCells: {len(coords):,}; classes: {len(categories)}; rare classes: {len(rare)}')
    print(f'Interactive viewer: {page}\nStatic overview: {out/"reconstruction_overview.png"}\nVirtual slices: {out/"virtual_slices.png"}')
    if args.open:
        webbrowser.open(page.as_uri())


if __name__=='__main__':
    main()
