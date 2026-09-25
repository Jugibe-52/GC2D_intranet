"""Checkpoint orchestration using the frozen public classical RK4 integrator."""
from pathlib import Path
import hashlib,json,os,time
import numpy as np
from study_io import load_snapshot, digest, atomic_json, utc_now

class CheckpointTestInterruption(RuntimeError):
    """Intentional interruption for validation only."""

def calculate_checkpointed_group(indices,initial_xy,settings):
    """Resume immutable chunks without changing their public-integrator schedule."""
    from threadpoolctl import threadpool_limits,threadpool_info
    potential,_,snapshot=load_snapshot()
    from dynamics import GuidingCenterDynamics
    from initial_conditions import GCInitialConfiguration
    from methods.classical.rk4 import RK4
    from contracts.problem import InitialValueProblem
    from contracts.request import SimulationRequest
    from simulation.runner import simulate
    import parallel_calculation
    indices=np.asarray(indices,dtype=int);ids=np.asarray(settings['particle_ids'])[indices]
    xy=np.asarray(initial_xy)[indices];count=len(indices);value=np.r_[xy[:,0],xy[:,1]]
    field=GuidingCenterDynamics(potential,rho=settings['rho'])
    total=settings['n_steps'];h=settings['step'];t0=settings['t_span'][0]
    folder=Path(settings['checkpoint_directory'])/f'worker_{indices[0]+1:02d}';folder.mkdir(parents=True,exist_ok=True)
    contract=dict(settings={k:v for k,v in settings.items() if k not in ('checkpoint_directory','test_stop_after_chunks')},ids=ids.tolist(),initial_xy=xy.tolist(),snapshot=snapshot,driver=digest(Path(__file__)),numpy_version=np.__version__)
    fingerprint=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    parts=[];completed=0
    with threadpool_limits(limits=1):
        pools=threadpool_info();assert all(p['num_threads']==1 for p in pools)
        if parallel_calculation._START_BARRIER is not None:parallel_calculation._START_BARRIER.wait(timeout=180)
        started=time.perf_counter();cpu=time.process_time();start_utc=utc_now()
        for marker in sorted(folder.glob('chunk_*.json')):
            record=json.loads(marker.read_text());file=marker.with_suffix('.npz')
            assert record['fingerprint']==fingerprint and record['start_step']==completed
            assert record['file']==file.name and digest(file)==record['sha256']
            assert completed<record['end_step']<=total
            with np.load(file,allow_pickle=False) as a:part=a['states'].copy()
            assert part.shape==(2*count,record['end_step']-completed+1)
            np.testing.assert_array_equal(part[:,0],value)
            parts.append(part);value=part[:,-1].copy();completed=record['end_step']
        resumed=completed;new_chunks=0
        while completed<total:
            end=min(total,completed+settings['checkpoint_steps']);length=end-completed
            initial=GCInitialConfiguration.from_components(x=value[:count],y=value[count:])
            problem=InitialValueProblem(field,initial)
            request=SimulationRequest.uniform(t_span=(t0+completed*h,t0+end*h),max_step=h,sample_count=length+1)
            solution=simulate(problem,RK4(progress=False),request)
            assert solution.diagnostics['step_count']==length
            part=solution.states;assert part.shape==(2*count,length+1) and np.isfinite(part).all()
            np.testing.assert_array_equal(part[:,0],value)
            file=folder/f'chunk_{end:08d}.npz';temporary=file.with_suffix('.npz.tmp')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream,states=part);stream.flush();os.fsync(stream.fileno())
            temporary.replace(file)
            atomic_json(file.with_suffix('.json'),dict(fingerprint=fingerprint,file=file.name,sha256=digest(file),start_step=completed,end_step=end,committed_utc=utc_now(),particle_ids=ids.tolist(),pid=os.getpid()))
            parts.append(part);value=part[:,-1].copy();completed=end;new_chunks+=1
            atomic_json(folder/'progress.json',dict(completed_steps=completed,total_steps=total,updated_utc=utc_now(),pid=os.getpid(),resumed_steps=resumed))
            print(f'Worker {indices[0]+1}: checkpoint {completed}/{total} ({100*completed/total:.1f}%)',flush=True)
            if settings.get('test_stop_after_chunks')==new_chunks:raise CheckpointTestInterruption('Simulated interruption after a durable checkpoint.')
        finished=time.perf_counter();finish_utc=utc_now();cpu_seconds=time.process_time()-cpu
    states=np.concatenate([parts[0]]+[p[:,1:] for p in parts[1:]],axis=1)
    return dict(indices=indices,times=t0+np.arange(total+1)*h,xy=np.stack((states[:count].T,states[count:].T),axis=-1),diagnostics={},newton_history=None,
        record=dict(pid=os.getpid(),particle_ids=ids.tolist(),particle_count=count,started_utc=start_utc,finished_utc=finish_utc,started_monotonic_s=started,finished_monotonic_s=finished,integration_seconds=finished-started,cpu_seconds=cpu_seconds,snapshot_sha256=snapshot,threadpools=pools,resumed_steps=resumed,new_steps=total-resumed,checkpoint_chunks=len(parts),checkpoint_fingerprint=fingerprint,field_evaluations_per_new_step=4))
