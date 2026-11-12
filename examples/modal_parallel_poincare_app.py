"""Bounded CPU JAX workers for the 36-neighbour Poincare study."""
from pathlib import Path
import sys

import modal
from execution._modal_worker import execute_cycle_payload
from states.history import States

ROOT = Path(__file__).resolve().parents[1]
image = (modal.Image.debian_slim(python_version=f"{sys.version_info.major}.{sys.version_info.minor}")
    .add_local_file(ROOT / "constraints.txt", "/tmp/gc2d-constraints.txt", copy=True)
    .pip_install_from_pyproject(str(ROOT / "pyproject.toml"), optional_dependencies=["jax"],
                               extra_options="-c /tmp/gc2d-constraints.txt")
    .env({"JAX_ENABLE_X64": "true", "JAX_PLATFORMS": "cpu"})
    .add_local_python_source("contracts", "diagnostics", "dynamics", "execution", "formulations",
                            "states", "methods", "potential", "solvers", "problem", "request"))
app = modal.App("gc2d-parallel-poincare")


@app.function(image=image, cpu=(2.0, 2.0), memory=(2048, 4096), timeout=2400,
              retries=0, max_containers=44, scaledown_window=5)
def integrate(payload: bytes) -> tuple[int, str, States]:
    """Integrate one rho/method case with no automatic retry."""
    return execute_cycle_payload(payload)


@app.function(image=image, cpu=(2.0, 2.0), memory=(2048, 4096), timeout=600,
              retries=0, max_containers=4, scaledown_window=5)
def benchmark(payload: bytes) -> tuple[int, str, States]:
    """Validate short cases under a separate time/concurrency ceiling."""
    return execute_cycle_payload(payload)


@app.function(image=image, cpu=(2.0, 2.0), memory=(2048, 4096), timeout=1200,
              retries=0, max_containers=1, scaledown_window=5)
def diagnose_failure(payload: bytes) -> dict:
    """Stop at the first rejected step; never export failed trajectory metrics."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from execution._modal_worker import decode_job
    from execution.execution import _prepare_backend, _prepare_numerical_run
    from execution._jax_step import prepare_step
    from execution._plan import build_fixed_step_plan

    problem,method,options,_=decode_job(payload)
    dynamics,device,_,_=_prepare_backend(problem.dynamics,method,options)
    run,_=_prepare_numerical_run(problem.initial_conditions,method,dynamics,track_energy=False)
    kernel=prepare_step(run,device,initial_conditions=problem.initial_conditions)
    plan=build_fixed_step_plan(problem.request,method.step)
    starts=jnp.asarray(plan.step_starts);sizes=jnp.asarray(plan.step_sizes)

    @jax.jit
    def advance(initial,index,limit):
        def condition(c):return (c[2]<limit)&c[3]
        def step(c):
            state,_,i,_,_,_,_=c
            after,stats,ok=kernel(starts[i],state,sizes[i])
            return (after,state,i+1,ok,stats['nonlinear_iterations'],
                    stats['nonlinear_residual_norms'],stats['nonlinear_tolerances'])
        return jax.lax.while_loop(condition,step,(initial,initial,index,jnp.array(True),
            jnp.array(0),jnp.array(0.),jnp.array(0.)))

    state=jnp.asarray(problem.initial_state);index=0
    while index<len(plan.step_sizes):
        after,before,index,ok,iterations,residual,tolerance=jax.device_get(
            advance(state,jnp.asarray(index),jnp.asarray(min(index+12500,len(plan.step_sizes)))))
        index=int(index)
        print(f'Diagnostic reached cycle {index/50:.2f}; converged={bool(ok)}',flush=True)
        if not ok:
            return dict(status='nonconvergence',step=index-1,time=float(plan.step_starts[index-1]),
                duration=float(plan.step_sizes[index-1]),before=before.tolist(),after=after.tolist(),
                iterations=int(iterations),residual=float(residual),tolerance=float(tolerance))
        state=jnp.asarray(after)
    return dict(status='all_steps_converged_in_diagnostic_loop',steps=index)
