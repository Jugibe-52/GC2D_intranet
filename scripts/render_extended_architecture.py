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
    doubled: bool = True


MODELS = (
    Model('extended', 'extended-simulation-architecture', 'ABBA and BM4',
          'Explicit A/B/T operations\nGeneric recipes may also include C',
          'One outer projection per implicit step\nNo intermediate projections'),
    Model('abba', 'abba-numerical-architecture', 'ABBA family',
          'ABBA2 pair; ABBA4 triple jump\nABBA6: seven signed ABBA2 steps',
          'ABBA4 and ABBA6 share one outer solve\nOnly the unprojected recipe differs'),
    Model('abba2-implicit', 'abba2-implicit-simulation-architecture', 'ABBA2Implicit',
          'A/2, B/2, T/2, T/2, B/2, A/2\nNo harmonic coupling',
          'One spatial projection per step\nReduced or simultaneous equation'),
    Model('abba4-implicit', 'abba4-implicit-simulation-architecture', 'ABBA4Implicit',
          'Three ABBA pairs; six signed stages\nTriple-jump coefficients; no coupling',
          'One projection around all six stages\nNo intermediate projection'),
    Model('abba6-implicit', 'abba6-implicit-simulation-architecture', 'ABBA6Implicit',
          'Seven unprojected pairs; fourteen stages\nYoshida order-six coefficients',
          'One projection around all fourteen stages\nSame execution structure as ABBA4'),
    Model('bm4-implicit', 'bm4-simulation-architecture', 'BM4Implicit',
          'Twelve alternating signed stages\nNo harmonic coupling',
          'One reduced projection per cycle\nAnalytic or finite-difference Newton'),
    Model('abba2-midpoint', 'abba2-midpoint-simulation-architecture', 'ABBA2Midpoint',
          'A/2, B/2, T/2, T/2, B/2, A/2\nNo harmonic coupling',
          'Arithmetic mean after one pair\nNo nonlinear solve', False),
    Model('bm4-midpoint', 'bm4-midpoint-simulation-architecture', 'BM4Midpoint',
          'Twelve alternating signed stages\nNo harmonic coupling',
          'Arithmetic mean after the full cycle\nNo nonlinear solve', False),
    Model('explicit-euler', 'explicit-euler-simulation-architecture', 'ExplicitEuler',
          'One physical slope at the interval start', 'Explicit first-order update', False, False),
    Model('rk4', 'rk4-simulation-architecture', 'RK4',
          'Four physical stages; classical weights', 'Explicit fourth-order update', False, False),
    Model('gauss-legendre4', 'gauss-legendre4-simulation-architecture', 'GaussLegendre4',
          'Two Gauss collocation stages', 'One joint Newton system per step', True, False),
    Model('sdirk4', 'sdirk4-simulation-architecture', 'SDIRK4',
          'Five diagonally implicit stages', 'One Newton system per stage', True, False),
    Model('hbvm42', 'hbvm42-simulation-architecture', 'HBVM42',
          'Two Legendre modes; four Gauss nodes', 'Newton with matrix norm and backtracking', True, False),
    Model('dop853', 'dop853-simulation-architecture', 'DOP853',
          'SciPy adaptive explicit stages', 'Live physical solver and dense output', False, False),
    Model('radau', 'radau-simulation-architecture', 'Radau',
          'SciPy adaptive Radau IIA stages', 'Live physical solver and dense output', True, False),
)


def render(model: Model) -> None:
    """Generate source and readable views of the same numerical dependencies."""
    folder = ROOT / 'docs/models' / model.directory / 'simulation'
    folder.mkdir(parents=True, exist_ok=True)
    projection = ('solve_projection', 'core/projection.py\nSpatial Hairer equation') if model.implicit else ('midpoint_step', 'core/midpoint.py\nOne arithmetic diagonal average')
    nodes = {
        'recipe': (2.3, 8.0, 'Composition recipe', model.recipe, '#fff0d9'),
        'method': (7.0, 8.0, model.name, 'Exports: simulation / methods.extended\nFixed step belongs to the method', '#e5ecfa'),
        'driver': (11.7, 8.0, 'Simulation + execution', 'Bind f / derivatives and observer functions\nAccepted intervals, samples and collection', '#e5ecfa'),
        'state': (2.3, 6.5, 'DoubleFormulation', 'Physical size only; lift / join / split\nTemporary copies; no time or momentum', '#e5ecfa'),
        'advance': (7.0, 6.5, 'Common advance', 'Prepare copies; solve or average\nPhysical Z, work and optional quadrature', '#e5ecfa'),
        'energy': (11.7, 6.5, 'EnergyObserver', 'Own momentum; consume accepted quadrature\nBind H and -partial_t H through execution', '#e1f0e6'),
        'observe': (2.3, 5.0, 'Optional observations', 'extended/observations.py: event adapters\nPublic records: contracts/methods/', '#f0e5f5'),
        'projection': (7.0, 5.0, projection[0], projection[1], '#e1f0e6'),
        'solver': (11.7, 5.0, 'Newton / Broyden' if model.implicit else 'Explicit projection',
                   'Own tolerances, derivatives and iterations\nBacktracking and iteration callbacks' if model.implicit else 'Average the two final copies\nNo Newton / Broyden calls', '#fff0d9'),
        'placement': (2.3, 3.5, 'Method identity', model.placement, '#fff0d9'),
        'compose': (7.0, 3.5, 'compose', 'core/composition.py: explicit A/B/T/C\nAccepted trace reused by all consumers', '#e1f0e6'),
        'tangent': (11.7, 3.5, 'Analytic tangents' if model.implicit else 'Accepted stage trace',
                    'core/jacobians.py: ordered particle tangents\nBroyden does not evaluate Jacobians' if model.implicit else 'Signed times, durations and shear inputs\nEnergy evaluated after the spatial map', '#e1f0e6'),
        'maps': (7.0, 2.0, 'GCDoubledMaps', 'apply_spatial(A / B / C)\nOnly T changes the clock\nBM4 has no C operation', '#e1f0e6'),
        'dynamics': (7.0, .5, 'Injected physical functions', 'f(t, Z) and optional Jacobian\nMethod and maps retain no dynamics object', '#e5ecfa'),
    }
    if model.directory in ('extended', 'abba'):
        nodes['projection'] = (7.0, 5.0, 'Spatial projection',
            'solve_projection / midpoint_step\nHairer equation or arithmetic mean', '#e1f0e6')
        nodes['solver'] = (11.7, 5.0, 'Newton / Broyden',
            'Implicit variants only; own solver controls\nTolerances, derivatives and callbacks', '#fff0d9')
    edges = [('recipe','method'),('method','driver'),('driver','advance'),('state','advance'),
             ('advance','projection'),('driver','energy'),('advance','energy'),('advance','observe'),
             ('projection','compose'),('projection','solver'),('compose','tangent'),
             ('compose','maps'),('maps','dynamics')]
    if not model.doubled:
        adaptive = model.directory in ('dop853', 'radau')
        nodes['method'] = (7.0, 8.0, model.name, 'Exports: simulation / methods\nAdaptive tolerances and step bounds' if adaptive else 'Exports: simulation / methods\nFixed integration step belongs to the method', '#e5ecfa')
        nodes['recipe'] = (2.3, 8.0, 'Numerical scheme', model.recipe, '#fff0d9')
        nodes['state'] = (2.3, 6.5, 'Physical state', 'InitialConditions: values and layout\nNo formulation or auxiliary coordinates', '#e5ecfa')
        nodes['advance'] = (7.0, 6.5, 'Physical advance', 'Evaluate injected f and optional derivatives\nReturn Z, work and optional quadrature', '#e5ecfa')
        nodes['projection'] = (7.0, 5.0, 'Adaptive step' if adaptive else 'Stage equations', 'One retained SciPy numerical session' if adaptive else 'Method owns tableau and equation assembly', '#e1f0e6')
        nodes['compose'] = (7.0, 3.5, 'Physical evaluations', 'Physical vector-field and optional derivative calls\nNo Hamiltonian or momentum evaluation', '#e1f0e6')
        nodes['maps'] = (7.0, 2.0, 'Accepted numerical data', 'Dense interpolant for passive quadrature' if adaptive else 'Retained stage times, states and weights', '#e1f0e6')
        nodes['tangent'] = (11.7, 3.5, 'Sampling', 'Accepted dense output' if adaptive else 'Independent shortened physical steps\nShadow quadrature never commits observer state', '#e1f0e6')
        nodes['observe'] = (2.3, 5.0, 'Step callbacks', 'Passed through simulate(observers=...)\nExecution supplies physical provenance', '#f0e5f5')
        if adaptive:
            nodes['solver'] = (11.7, 5.0, 'SciPy adaptive control', 'Method owns atol / rtol and step bounds\nPhysical-only error and acceptance', '#fff0d9')
        elif model.implicit:
            nodes['solver'] = (11.7, 5.0, 'NewtonSolver', 'Own tolerances, derivatives and iterations\nBacktracking and iteration callbacks', '#fff0d9')
        elif not model.implicit:
            nodes['solver'] = (11.7, 5.0, 'Explicit update', 'No nonlinear solver\nMethod owns its fixed integration step', '#fff0d9')
        edges = [('recipe','method'),('method','driver'),('driver','advance'),('state','advance'),
                 ('advance','projection'),('driver','energy'),('advance','energy'),('advance','observe'),
                 ('projection','compose'),('projection','solver'),('compose','tangent'),
                 ('compose','maps'),('maps','dynamics')]
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor('#f8fafc'); ax.set_facecolor('#f8fafc')
    ax.set(xlim=(0,14), ylim=(-.3,9.4)); ax.axis('off')
    ax.text(.35,9.15,f'{model.name} | functions, numerical method and observers',fontsize=19,weight='bold',color='#18324d')
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
