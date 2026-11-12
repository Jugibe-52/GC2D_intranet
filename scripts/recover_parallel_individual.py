"""Recover the failed independent particle block without changing numerical controls."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import replace,asdict
import json
from time import time
from types import SimpleNamespace
import numpy as np
from run_poincare_parallel import BASE, OUTPUT, SOURCE, APP
from run_horizontal_transport import billing, write_json
from studies.poincare_parallel import prepare_parallel
from studies.poincare_rho_sweep import _rho_star_method, validate_rho_solution, folded_rho_positions
from execution.execution_modal import Execution_Modal
from execution.options import ExecutionOptions
from diagnostics.io.persistence import load_solution, save_solution
from diagnostics.execution.horizontal_transport import HorizontalTransportObserver
from problem import InitialValueProblem
from dynamics.gc import GuidingCenterDynamics
from diagnostics.execution.horizontal_transport import HORIZONTAL_METRIC_ARRAYS
from states.history import States


def main() -> None:
    """Run and validate independent trajectories, then merge complete results only."""
    if (OUTPUT/'runs/bm4_implicit_01/manifest.json').exists():
        raise ValueError('The combined archive already exists; use the audit command.')
    failures=json.loads((OUTPUT/'recovery_01_failed.json').read_text())
    if failures.get('status')!='failed' or failures.get('method')!='BM4Implicit' or failures.get('rho')!=.05:
        raise ValueError('Independent recovery requires the confirmed failed damped attempt.')
    seeds=json.loads((OUTPUT/'seeds.json').read_text())
    original=json.loads((BASE/'horizontal_transport/runs/bm4_implicit_01_star/metadata.json').read_text())
    prepared=prepare_parallel(SOURCE,original,seeds)
    options=ExecutionOptions(backend='jax',device='cpu')
    current=billing();initial=json.loads((OUTPUT/'billing_before.json').read_text())
    ceiling=current['max_worker_hour_usd']*36*2400/3600
    if current['metered_usd']-initial['metered_usd']+ceiling>=6:raise ValueError('Study budget cannot cover bounded single-particle recovery.')
    root=OUTPUT/'individual_recovery/bm4_implicit_01'
    report={'status':'running','started':time(),'reason':'Joint undamped and damped Newton both failed; independent particles retain original fixed step and solver controls.','jobs':[]}

    def one(i: int) -> dict:
        pid=49+i;xy=np.asarray(seeds['particles'][i]['position_m'])
        problem=InitialValueProblem(prepared.problem.dynamics,xy,layout=GuidingCenterDynamics.coordinates)
        # A literal single seed has no radial-star geometry or minimum star size.
        config=SimpleNamespace(**{**asdict(prepared.config),'particles':1,'arms':1})
        single=replace(prepared,problem=problem,config=config)
        folder=root/f'particle_{pid:02d}'
        if (folder/'manifest.json').exists():
            stored=load_solution(folder)
            if (stored.metadata['particle_id']!=[pid] or stored.metadata['seed_manifest']!=seeds
                    or stored.metadata['config']!=json.loads(json.dumps(vars(config)))
                    or stored.metadata['source_sha256']!=prepared.metadata['source_sha256']):
                raise ValueError('Saved single-particle inputs differ.')
            data=stored.solution
        else:
            receipt_dir=OUTPUT/'receipts'/f'bm4_implicit_01_particle_{pid:02d}'
            receipts=list(receipt_dir.glob('*.json'))
            if len(receipts)>1:raise ValueError('Multiple receipts require inspection.')
            executor=Execution_Modal.from_record(receipts[0]) if receipts else Execution_Modal(app_name=APP,function_name='integrate',record_directory=receipt_dir)
            data=executor.run(problem.with_request(prepared.request),_rho_star_method(config),options=options,observers=(HorizontalTransportObserver(),))
            validate_rho_solution(data,single,options=options)
            metadata={**prepared.metadata,'config':vars(config),'particle_id':[pid],'initial_positions':xy[None,:],
                      'initial_positions_m':xy[None,:],'initial_positions_cell_fraction':np.asarray([seeds['particles'][i]['cell_fraction']]),
                      'execution_executor':'modal','modal_call_id':executor.last_call_id,'execution_options':asdict(options),
                      'grouping_recovery':'Independent particle after confirmed joint nonconvergence'}
            save_solution(data,folder,metadata=metadata)
        validate_rho_solution(data,single,options=options)
        d=data.diagnostics
        if d['horizontal_observed_steps']!=250000:raise ValueError('Incomplete observer.')
        np.testing.assert_allclose(d['horizontal_transport_rate'],d['horizontal_travel']/(5000*d['horizontal_width']),rtol=1e-14)
        return dict(particle=pid,status='completed',call_id=d.get('execution_call_id'),worker_seconds=float(d['worker_wall_seconds']))

    with ThreadPoolExecutor(max_workers=36) as pool:
        futures={pool.submit(one,i):49+i for i in range(36)}
        for f in as_completed(futures):
            try:row=f.result()
            except Exception as exc:row=dict(particle=futures[f],status='failed',error_type=type(exc).__name__,error=str(exc))
            report['jobs'].append(row);write_json(OUTPUT/'individual_recovery_progress.json',report);print(json.dumps(row),flush=True)
    report.update(status='failed' if any(j['status']!='completed' for j in report['jobs']) else 'completed',finished=time())
    write_json(OUTPUT/'individual_recovery_progress.json',report)
    if report['status']!='completed':raise RuntimeError('Independent recovery contains failed particles; no combined archive written.')
    records=[load_solution(root/f'particle_{pid:02d}') for pid in range(49,85)]
    values=np.concatenate([np.column_stack([r.solution.states[:,component] for r in records]) for component in (0,1)],axis=1)
    first=records[0].solution
    run=_rho_star_method(prepared.config).new_run(prepared.problem.dynamics.vector_field,prepared.problem.initial_conditions,jacobian=prepared.problem.dynamics.particle_vector_field_jacobians)
    diagnostics={**dict(first.diagnostics),**run.metadata}
    for key in HORIZONTAL_METRIC_ARRAYS:diagnostics[key]=np.concatenate([r.solution.diagnostics[key] for r in records])
    for key in ['execution_call_id']:diagnostics.pop(key,None)
    diagnostics.update(worker_wall_seconds=sum(r.solution.diagnostics['worker_wall_seconds'] for r in records),
        worker_started_unix_seconds=min(r.solution.diagnostics['worker_started_unix_seconds'] for r in records),
        worker_finished_unix_seconds=max(r.solution.diagnostics['worker_finished_unix_seconds'] for r in records))
    solution=States(first.times,values,diagnostics,layout=GuidingCenterDynamics.coordinates,initial_state=prepared.problem.initial_state)
    validate_rho_solution(solution,prepared,options=options)
    wrapped,fractions=folded_rho_positions(solution,prepared.metadata)
    solution=solution.with_diagnostics({'cycle_positions_wrapped':wrapped,'cycle_positions_cell_fraction':fractions})
    reference=load_solution(OUTPUT/'benchmarks/bm4_implicit_01_20c_50s').solution
    comparison=float(np.max(np.abs(solution.states[:21]-reference.states)))
    write_json(OUTPUT/'individual_recovery_comparison.json',dict(cycles=20,max_state_difference_m=comparison))
    save_solution(solution,OUTPUT/'runs/bm4_implicit_01',metadata={**prepared.metadata,'execution_options':asdict(options),
        'execution_executor':'modal','modal_call_ids':[r.metadata['modal_call_id'] for r in records],
        'horizontal_metric':'integral_abs_vx/(cycles*(max_x-min_x))','saved_folded_returns':True,
        'grouping_recovery':{'reason':'Joint undamped and damped Newton failed','particle_groups':36,
            'unchanged':'seeds, field, duration, fixed step, Newton max iterations and tolerances',
            'short_comparison_max_difference_m':comparison}})
    print('Merged 36 validated independent trajectories.',flush=True)


if __name__=="__main__":main()
