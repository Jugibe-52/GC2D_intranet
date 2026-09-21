"""Render saved cloud Poincare products without loading an integrator."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable
from study_io import ROOT, load_calculation
from poincare_selector import export_poincare_selector


def render_saved(run_id=None):
    """Create static plots and a self-contained selector with original IDs."""
    directory, meta, arrays, positions, initial = load_calculation(run_id)
    output = ROOT/'figuras'/directory.name
    output.mkdir(parents=True,exist_ok=True)
    n, cycles, spc = meta['particle_count'],meta['cycles'],meta['steps_per_cycle']
    ids, colors = arrays['particle_ids'],arrays['colors_rgba']
    xy = arrays['cycle_positions_wrapped'][1:]/meta['field_provenance']['grid']['period']
    assert xy.shape == (cycles,n,2) and len(positions)==n*cycles
    np.testing.assert_array_equal(np.unique(positions['particle']), ids)
    for i,color in zip(ids,arrays['colors_hex']):
        assert (positions.loc[positions.particle==i,'color']==color).all()
    title=f'Hamiltonian Poincare section | {meta["method"]}'
    subtitle=f'{n} particles · {cycles} cycles · {spc} steps/cycle · {meta["process_count"]} processes'
    boundaries = np.r_[ids[0]-.5,(ids[:-1]+ids[1:])/2,ids[-1]+.5]
    cmap,norm = ListedColormap(colors),BoundaryNorm(boundaries,len(ids))
    for name,points in [('poincare_section',xy),('initial_positions',(arrays['initial_positions']/meta['field_provenance']['grid']['period'])[None,:,:])]:
        fig, ax = plt.subplots(figsize=(10,8),layout='constrained')
        for i in range(n): ax.scatter(points[:,i,0],points[:,i,1],color=colors[i],s=10 if name=='poincare_section' else 35,alpha=.85,linewidths=0)
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='x / L',ylabel='y / L',title=title+'\n'+subtitle if name=='poincare_section' else 'Original initial positions and particle IDs')
        ax.set_aspect('equal');ax.grid(alpha=.16)
        fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=ax,label='Original particle ID (fixed color)',ticks=ids[::max(1,n//8)])
        fig.savefig(output/(name+'.png'),dpi=160,bbox_inches='tight')
        if name=='poincare_section': fig.savefig(output/(name+'.svg'),bbox_inches='tight')
        plt.close(fig)
    if meta['method']=='BM4Implicit':
        series=[arrays['nonlinear_iterations'], arrays['nonlinear_residuals']/arrays['nonlinear_tolerances'],arrays['projection_multiplier_norms']]
        labels=['Newton corrections','Final residual / tolerance','Implicit multiplier infinity norm']
    elif meta['method']=='BM4Midpoint':
        series=[arrays['copy_separation_norms']]; labels=['Copy separation infinity norm']
    else:
        series=[];labels=[]
    if series:
        fig, axes=plt.subplots(len(series),1,figsize=(11,3*len(series)),squeeze=False,layout='constrained')
        step=100;time=arrays['times'][1:];size=len(time)//step
        for ax,values,label in zip(axes[:,0],series,labels):
            if size:
                values=values[:,:size*step].reshape(len(values),size,step).max(axis=2);t=time[:size*step].reshape(size,step)[:,-1]
            else:t=time
            for row in values: ax.plot(t,row,lw=.6)
            ax.set(xlabel='Forcing cycles',ylabel=label,title=label+(' (maxima over 100 steps)' if size else ' (every step)'))
        fig.savefig(output/'diagnostics.png',dpi=160,bbox_inches='tight');plt.close(fig)
    html=export_poincare_selector(output/'poincare_selector.html',xy,ids,arrays['colors_hex'],title=title,subtitle=subtitle)
    print(f'Saved plots and particle selector: {output}')
    return html
