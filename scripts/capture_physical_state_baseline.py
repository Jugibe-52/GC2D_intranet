"""Capture the physical-state migration matrix on either side of the refactor.

Use --help for backend and output prefix. Numerical arrays exclude changed
workspace dimension metadata; JSON records controls and scalar diagnostics.
"""

from pathlib import Path
import argparse, json
import numpy as np
from potential import Potential
from dynamics import GuidingCenterDynamics, FullCyclotronDynamics
from simulation import (InitialValueProblem, SimulationRequest, simulate, ExplicitEuler, RK4,
    GaussLegendre4, SDIRK4, HBVM42, DOP853, Radau, ABBA2Midpoint, ABBA2Implicit,
    ABBA4Implicit, ABBA6Implicit, BM4Implicit, BM4Midpoint, NewtonSolver, BroydenSolver,
    ReducedProjection, SimultaneousProjection)
from execution.options import ExecutionOptions
from diagnostics.methods.energy import EnergyObserver

parser = argparse.ArgumentParser()
parser.add_argument('output')
parser.add_argument('--backend', default='scipy')
args = parser.parse_args()
field = Potential.random(A=0.05, M=2, nx=16, ny=16, seed=43)
models = [('gc', GuidingCenterDynamics(field, rho=0.1, B=1.5), np.array([.25,.65,.35,.7])),
          ('fc', FullCyclotronDynamics(field, rho=.2, eta=.1), np.array([.25,.65,.35,.7,.1,-.2,.2,.1]))]
request = SimulationRequest((.15,.255), output_times=np.array([.15,.153,.17,.181,.21,.23,.25,.255]))
arrays = {}; records = {}
for model, dynamics, initial in models:
    problem = InitialValueProblem(dynamics, initial)
    for energy in (False, True):
        options = dict(step=.02)
        cases = [(cls.__name__, cls(**options)) for cls in (ExplicitEuler,RK4,GaussLegendre4,SDIRK4,HBVM42)]
        cases += [(cls.__name__, cls(max_step=.02)) for cls in (DOP853,Radau)]
        if model == 'gc':
            cases += [(cls.__name__, cls(**options)) for cls in (ABBA2Midpoint,BM4Midpoint,ABBA6Implicit)]
            for cls in (ABBA2Implicit,ABBA4Implicit,BM4Implicit):
                for solver in (NewtonSolver(absolute_tolerance=1e-12,relative_tolerance=1e-11,max_iterations=40,jacobian_method="analytic"),
                               BroydenSolver(absolute_tolerance=1e-12,relative_tolerance=1e-11,max_iterations=40,jacobian_method="analytic")):
                    projections = (ReducedProjection(), SimultaneousProjection()) if cls != BM4Implicit else (None,)
                    for projection in projections:
                        extra = dict(solver=solver)
                        if projection is not None: extra['projection_formulation'] = projection
                        cases.append((f'{cls.__name__}_{solver.name}_{projection.name if projection else "reduced"}',cls(**options,**extra)))
        for name, method in cases:
            key = f'{model}_{name}_energy{int(energy)}'
            result = simulate(problem, method, request, options=ExecutionOptions(backend=args.backend),
                observers=(EnergyObserver(),) if energy else ())
            arrays[key + '/values'] = result.values
            arrays[key + '/times'] = result.times
            record = {}
            for k,v in result.diagnostics.items():
                if isinstance(v,np.ndarray): arrays[key + '/' + k] = v
                elif isinstance(v,(float,int,str,bool)): record[k] = v
            records[key] = record
            print(key, 'ok', flush=True)
np.savez_compressed(args.output + '.npz', **arrays)
Path(args.output + '.json').write_text(json.dumps(records, indent=2))
print('cases',len(records),'arrays',len(arrays),flush=True)
