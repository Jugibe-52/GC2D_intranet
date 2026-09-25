"""Compare resumed trajectories against uninterrupted frozen public integrators."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import json
import numpy as np
from study_io import ROOT,load_calculation,load_snapshot,atomic_json,utc_now

def baseline(indices,xy,meta):
    from threadpoolctl import threadpool_limits
    potential,_,_=load_snapshot()
    from dynamics import GuidingCenterDynamics
    from initial_conditions import GCInitialConfiguration
    from methods.extended.bm4 import BM4Implicit
    from methods.extended.bm4_midpoint import BM4Midpoint
    from contracts.problem import InitialValueProblem
    from contracts.request import SimulationRequest
    from simulation.runner import simulate
    initial=GCInitialConfiguration.from_components(x=xy[indices,0],y=xy[indices,1])
    problem=InitialValueProblem(GuidingCenterDynamics(potential,rho=meta['rho']),initial)
    request=SimulationRequest.uniform(t_span=tuple(meta['t_span']),max_step=meta['step'],sample_count=meta['complete_steps']+1)
    if meta['method']=='BM4Implicit':
        method=BM4Implicit(coupling_frequency=meta['coupling_frequency'],newton_absolute_tolerance=meta['newton_atol'],newton_relative_tolerance=meta['newton_rtol'],newton_max_iterations=meta['newton_max_iterations'],newton_jacobian_relative_step=meta['jacobian_relative_step'],newton_jacobian_method='analytic',nonlinear_solver='newton',progress=False)
    else:
        method=BM4Midpoint(coupling_frequency=meta['coupling_frequency'],progress=False)
    with threadpool_limits(limits=1): solution=simulate(problem,method,request)
    return solution.t,np.stack(solution.positions(),axis=-1).transpose(1,0,2),dict(solution.diagnostics)

def main():
    folder,meta,a,positions,initial=load_calculation('checkpoint_resume_test')
    n=meta['particle_count']; steps=meta['complete_steps']
    assert meta['validation_run'] and meta['cycles']==2 and meta['process_count']==16
    assert all(w['resumed_steps']==10 and w['new_steps']==steps-10 for w in meta['workers'])
    original=json.loads((ROOT/'assets/original_particles.json').read_text())['particles']
    np.testing.assert_array_equal(a['initial_positions'],[[original[i-1]['x'],original[i-1]['y']] for i in a['particle_ids']])
    np.testing.assert_array_equal(a['colors_rgba'],[original[i-1]['rgba'] for i in a['particle_ids']])
    assert len(positions)==2*n and len(set(a['colors_hex']))==n
    assert [w['particle_ids'] for w in meta['workers']]==[a['particle_ids'][w::16].tolist() for w in range(16)]
    if meta['method']=='BM4Implicit':
        np.testing.assert_array_equal(np.diff(a['newton_history_offsets'],axis=1),a['nonlinear_iterations']+1)
    else:
        assert not any('newton' in key or 'multiplier' in key for key in a)
    load_snapshot()
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        jobs=[pool.submit(baseline,np.arange(w,n,16),a['initial_positions'],meta) for w in range(16)]
        for w,job in enumerate(jobs):
            t,xy,d=job.result(); indices=np.arange(w,n,16)
            np.testing.assert_array_equal(t,a['times'])
            np.testing.assert_array_equal(xy,np.stack((a['states'][indices].T,a['states'][n+indices].T),axis=-1))
            keys=[('nonlinear_iterations','nonlinear_iterations'),('nonlinear_residual_norms','nonlinear_residuals'),('nonlinear_tolerances','nonlinear_tolerances'),('projection_multiplier_norms','projection_multiplier_norms')] if meta['method']=='BM4Implicit' else [('copy_separation_norms','copy_separation_norms')]
            for source,target in keys: np.testing.assert_array_equal(d[source],a[target][w])
    report=dict(verified_utc=utc_now(),method=meta['method'],particles=n,workers=16,resumed_steps_per_worker=10,total_steps_per_worker=steps,all_states_and_diagnostics_bitwise_equal=True,original_ids_positions_colors_preserved=True,baseline='unmodified frozen public simulate',cycle_rows=len(positions))
    atomic_json(ROOT/'aws/validation_restart.json',report)
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
