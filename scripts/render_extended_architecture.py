"""Render per-model architecture diagrams from one explicit graph specification.

Run with the project's visualization dependencies. Each model receives matching
PlantUML/Graphviz source and SVG/PNG views; Matplotlib uses fixed node positions
so regeneration does not require an external Graphviz executable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Model:
    """Public model identity and its concrete shared-engine recipe."""
    directory: str
    stem: str
    name: str
    recipe: str
    placement: str
    implicit: bool = True


MODELS = (
    Model('extended', 'extended-simulation-architecture', 'ABBA and BM4',
          'ABBA2: 2 | ABBA4: 6 | ABBA6: 14\nBM4: 12 signed direct/adjoint stages',
          'One outer projection per implicit step\nNo intermediate projections'),
    Model('abba', 'abba-numerical-architecture', 'ABBA family',
          'ABBA2 pair; ABBA4 triple jump\nABBA6: seven signed ABBA2 steps',
          'ABBA4 and ABBA6 share one outer solve\nOnly the unprojected recipe differs'),
    Model('abba2-implicit', 'abba2-implicit-simulation-architecture', 'ABBA2Implicit',
          'adjoint(h/2), direct(h/2)\nNo harmonic coupling',
          'One spatial projection per step\nReduced or simultaneous equation'),
    Model('abba4-implicit', 'abba4-implicit-simulation-architecture', 'ABBA4Implicit',
          'Three ABBA pairs; six signed stages\nTriple-jump coefficients; no coupling',
          'One projection around all six stages\nNo intermediate projection'),
    Model('abba6-implicit', 'abba6-implicit-simulation-architecture', 'ABBA6Implicit',
          'Seven unprojected pairs; fourteen stages\nYoshida order-six coefficients',
          'One projection around all fourteen stages\nSame execution structure as ABBA4'),
    Model('bm4-implicit', 'bm4-simulation-architecture', 'BM4Implicit',
          'Twelve alternating signed stages\nOptional harmonic coupling',
          'One reduced projection per cycle\nAnalytic or finite-difference Newton'),
    Model('abba2-midpoint', 'abba2-midpoint-simulation-architecture', 'ABBA2Midpoint',
          'adjoint(h/2), direct(h/2)\nNo harmonic coupling',
          'Arithmetic mean after one pair\nNo nonlinear solve', False),
    Model('bm4-midpoint', 'bm4-midpoint-simulation-architecture', 'BM4Midpoint',
          'Twelve alternating signed stages\nOptional harmonic coupling',
          'Arithmetic mean after the full cycle\nNo nonlinear solve', False),
)


def render(model: Model) -> None:
    """Generate source and readable views of the same numerical dependencies."""
    folder = ROOT / 'docs/models' / model.directory / 'simulation'
    folder.mkdir(parents=True, exist_ok=True)
    projection = ('solve_projection', 'core/projection.py\nSpatial Hairer equation') if model.implicit else ('midpoint_step', 'core/midpoint.py\nOne arithmetic diagonal average')
    nodes = {
        'recipe': (2.3, 8.0, 'Composition recipe', model.recipe, '#fff0d9'),
        'method': (7.0, 8.0, model.name, 'Exports: simulation / methods.extended\nClasses: extended/abba.py and bm4.py', '#e5ecfa'),
        'driver': (11.7, 8.0, 'Simulation + integration', 'src/simulation/runner.py and src/integration/\nMain steps, output sampling, collection', '#e5ecfa'),
        'state': (2.3, 6.5, 'DoubledFormulation', 'src/formulations/state.py\nTwo copies; optional time and kappa', '#e5ecfa'),
        'advance': (7.0, 6.5, 'advance', 'Bound recipe and projection configuration\nOne accepted numerical result', '#e5ecfa'),
        'energy': (11.7, 6.5, 'Passive energy and outputs', 'core/energy.py: accepted shear quadrature\nH + kappa; no feedback into the root', '#e1f0e6'),
        'observe': (2.3, 5.0, 'Optional observations', 'extended/observations.py: event adapters\nPublic records: contracts/observation.py', '#f0e5f5'),
        'projection': (7.0, 5.0, projection[0], projection[1], '#e1f0e6'),
        'solver': (11.7, 5.0, 'Newton / Broyden' if model.implicit else 'Explicit projection',
                   '_nonlinear.py: shared convergence and work\nTime and momentum excluded from roots' if model.implicit else 'Average the two final copies\nNo Newton / Broyden calls', '#fff0d9'),
        'placement': (2.3, 3.5, 'Method identity', model.placement, '#fff0d9'),
        'compose': (7.0, 3.5, 'compose', 'core/composition.py: signed stage clock\nAccepted trace reused by all consumers', '#e1f0e6'),
        'tangent': (11.7, 3.5, 'Analytic tangents' if model.implicit else 'Accepted stage trace',
                    'core/jacobians.py: ordered particle tangents\nBroyden does not evaluate Jacobians' if model.implicit else 'Signed times, durations and shear inputs\nEnergy evaluated after the spatial map', '#e1f0e6'),
        'maps': (7.0, 2.0, 'GCDoubledMaps', 'direct_map / adjoint_map\nSpatial shears and optional coupling', '#e1f0e6'),
        'dynamics': (7.0, .5, 'Physical dynamics', 'Vector field, analytic derivatives, H and partial_t H\nExisting dynamics and potential contracts', '#e5ecfa'),
    }
    if model.directory in ('extended', 'abba'):
        nodes['projection'] = (7.0, 5.0, 'Spatial projection',
            'solve_projection / midpoint_step\nHairer equation or arithmetic mean', '#e1f0e6')
        nodes['solver'] = (11.7, 5.0, 'Newton / Broyden',
            'Implicit variants only; shared convergence\nTime and momentum excluded from roots', '#fff0d9')
    edges = [('recipe','method'),('method','driver'),('driver','advance'),('state','advance'),
             ('advance','projection'),('advance','energy'),('advance','observe'),
             ('projection','compose'),('projection','solver'),('compose','tangent'),
             ('compose','maps'),('maps','dynamics')]
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor('#f8fafc'); ax.set_facecolor('#f8fafc')
    ax.set(xlim=(0,14), ylim=(-.3,9.4)); ax.axis('off')
    ax.text(.35,9.15,f'{model.name} | shared extended-space architecture',fontsize=19,weight='bold',color='#18324d')
    ax.text(.35,8.78,'Blue: lifecycle and state    Green: numerical operations    Orange: recipe and solver    Purple: observation',fontsize=10,color='#53657a')
    width, height = 4.05, 1.04
    for source, target in edges:
        x1,y1,*_ = nodes[source]; x2,y2,*_ = nodes[target]
        dx,dy=x2-x1,y2-y1
        if abs(dx)>abs(dy):
            start=(x1 + (width/2 if dx>0 else -width/2), y1)
            end=(x2 - (width/2 if dx>0 else -width/2), y2)
        else:
            start=(x1,y1 + (height/2 if dy>0 else -height/2))
            end=(x2,y2 - (height/2 if dy>0 else -height/2))
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=13,
                                    linewidth=1.25,color='#64778b',zorder=1))
    for x,y,title,body,color in nodes.values():
        ax.add_patch(FancyBboxPatch((x-width/2,y-height/2),width,height,
            boxstyle='round,pad=0.04,rounding_size=0.08',facecolor=color,edgecolor='#b8c6d4',linewidth=.9,zorder=2))
        ax.text(x,y+.26,title,ha='center',va='center',fontsize=12,weight='bold',color='#18324d',zorder=3)
        ax.text(x,y-.13,body,ha='center',va='center',fontsize=9.1,linespacing=1.5,color='#34495e',zorder=3)
    fig.subplots_adjust(left=.01,right=.99,top=.99,bottom=.01)
    with matplotlib.rc_context({'svg.fonttype':'none', 'svg.hashsalt':'gc2d-extended'}):
        fig.savefig(folder/(model.stem+'.svg'),facecolor=fig.get_facecolor(),metadata={'Date':None})
    svg = folder/(model.stem+'.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    fig.savefig(folder/(model.stem+'.png'),dpi=150,facecolor=fig.get_facecolor())
    plt.close(fig)
    source = ['@startdot', 'digraph architecture {', '  graph [rankdir=TB, bgcolor="#f8fafc"];',
              '  node [shape=box, style="rounded,filled", fontname="DejaVu Sans"];']
    for key,(_,_,title,body,color) in nodes.items():
        label=(title+'\n'+body).replace('"','\\"').replace('\n','\\n')
        source.append(f'  {key} [label="{label}", fillcolor="{color}"];')
    source.extend(f'  {a} -> {b};' for a,b in edges)
    source += ['}', '@enddot', '']
    (folder/(model.stem+'.puml')).write_text('\n'.join(source))


if __name__ == '__main__':
    for model in MODELS:
        render(model)
