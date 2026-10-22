"""Test exact restart and compare against unmodified public RK4."""

from formulations.layout import GCStateLayout
import json,tempfile
import numpy as np
from study_io import ROOT,load_snapshot,load_calculation,atomic_json,utc_now
from rk4_checkpoints import calculate_checkpointed_group

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from potential.potential import Potential
    from states.history import States
    from typing import Any


def main() -> None:
    m: dict[str, Any]
    a: dict[str, np.ndarray]
    _,m,a,_,_=load_calculation('checkpoint_resume_test');assert all(w['resumed_steps']==10 and w['new_steps']==30 for w in m['workers'])
    settings={k:m[k] for k in ('rho','coupling_frequency','newton_atol','newton_rtol','newton_max_iterations','jacobian_relative_step','t_span','step','method','particle_ids','checkpoint_steps')};settings.update(n_steps=40,record_newton_history=False,test_stop_after_chunks=0)
    potential: Potential
    potential,_,_=load_snapshot()
    from dynamics import GuidingCenterDynamics
    import numpy as np
    from methods.classical.rk4 import RK4
    from problem import InitialValueProblem
    from request import SimulationRequest
    from simulation.runner import simulate
    discrepancies=[]
    with tempfile.TemporaryDirectory() as directory:
        settings['checkpoint_directory']=directory
        for w in range(4):
            ids: np.ndarray=np.arange(w,48,4);xy=a['initial_positions'][ids]
            baseline: Any=calculate_checkpointed_group(ids,a['initial_positions'],settings)
            expected: np.ndarray=np.stack((a['states'][ids].T,a['states'][48+ids].T),axis=-1)
            np.testing.assert_array_equal(baseline['xy'],expected)
            problem: InitialValueProblem=InitialValueProblem(GuidingCenterDynamics(potential,rho=.3), GCStateLayout().pack_components(xy[:,0], xy[:,1]), layout=GCStateLayout())
            solution: States=simulate(problem,RK4(),SimulationRequest.uniform(t_span=(0.,2.),max_step=.05,sample_count=41))
            full: np.ndarray=np.stack(solution.positions(),axis=-1).transpose(1,0,2)
            discrepancies.append(float(np.max(np.abs(full-expected))))
            np.testing.assert_allclose(full,expected,rtol=0,atol=1e-12)
    report: dict[str, Any]=dict(verified_utc=utc_now(),restart_bitwise_identical_to_uninterrupted_chunk_schedule=True,all_four_groups_checked=True,maximum_difference_from_unchunked_public_RK4=max(discrepancies),comparison_absolute_tolerance=1e-12,resumed_steps_per_worker=10)
    atomic_json(ROOT/'validation/restart.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
