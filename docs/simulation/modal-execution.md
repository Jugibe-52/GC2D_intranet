# Modal integration execution

`Execution_Modal` inherits from `Execution` and executes one complete integration
on a deployed Modal Function. It returns `IntegrationData` to the local machine;
`simulate` then constructs and validates the immutable NumPy `Solution`.
The notebook or study can subsequently save it through the existing persistence
API to the configured Backblaze bucket. The remote worker does not upload results
or receive bucket credentials.

The adapter keeps endpoint-option validation and remote-result validation in
private functions in `execution.execution_modal`. The constructor checks names
and waiting controls without contacting Modal. Retrieval checks the response
shape, payload version, result type, and exact serialized-job digest before
adding executor diagnostics. Receipt writes, submission order, recovery, and
exception context remain owned by `Execution_Modal.run`.

The canonical class is `execution.execution_modal.Execution_Modal`, explicitly
re-exported by `execution`. Importing the package or using local execution does
not import Modal. Install the optional client before using remote execution:

```bash
python -m pip install -c constraints.txt -e '.[modal]'
```

## Deploy the worker separately

The deployable example is [`examples/modal_execution_app.py`](../../examples/modal_execution_app.py).
Authenticate the Modal CLI for the intended account, then deploy explicitly from
the repository root:

```bash
modal deploy examples/modal_execution_app.py
```

The example names the App `gc2d-execution` and its Function `integrate`. Its image
uses the local Python major/minor version and pinned numerical dependencies from
`constraints.txt`, and includes the required source packages. No notebooks, HDF5
files, results or credentials are mounted. Physical potential samples travel
inside each job and interpolation resources are reconstructed in the worker.
Deploy the same project revision and use matching Python/dependency versions on
both machines. Custom classes must be importable in the worker image; notebook
closures and objects containing live resources are not supported automatically.

The deployment example sets a one-hour execution limit and `retries=0`. Edit the
deployment to choose other limits or resources; the executor does not build or
deploy images, change CPU affinity, choose worker counts or configure JAX.
`retries=0` disables configured retries of function failures, but Modal has
separate container-crash recovery behavior. See Modal's
[failure and retry policy](https://modal.com/docs/guide/retries).

## Execute, retrieve locally, then persist

```python
from execution import Execution_Modal
from simulation.runner import simulate
from diagnostics.paths import solution_destination
from diagnostics.persistence import save_solution

executor = Execution_Modal(
    app_name="gc2d-execution",
    function_name="integrate",
    record_directory="logs/execution_modal",
)
solution = simulate(problem, method, request, execution=executor)

# Use the actual notebook experiment directory and its explicit run identifier.
destination = solution_destination(EXPERIMENT_PATH, RUN_ID)
save_solution(
    solution, destination,
    metadata={"modal_call_id": executor.last_call_id},
)
```

`EXPERIMENT_PATH` is relative to `notebooks/`, for example
`developements/my_study`. `solution_destination` preserves that hierarchy and
defaults to `gc2d_data:gc2d-notebooks-data`. Select `storage="local"` explicitly
only when local result storage is desired. A failed upload remains a failure;
it does not invalidate the already retrieved local `Solution` or cause another
remote integration. Keep all scientific parameters in the notebook and its
saved study metadata as before.

Remote execution accepts NumPy/SciPy and JAX CPU jobs. JAX requires a deployment
that installs the optional dependency and enables float64 before importing JAX;
select it explicitly with `options=ExecutionOptions(backend="jax", device="cpu")`.
GPU jobs are rejected before submission. Methods must have `progress=False`
and `step_observer=None`; this prevents local callbacks being silently moved
to another machine. Configured method instances are accepted, while initialized
or completed run instances are rejected. Numerical algorithms and the default
local/JAX execution paths are unchanged.

## JAX CPU resource comparison

[`examples/modal_jax_cpu_app.py`](../../examples/modal_jax_cpu_app.py) deploys
`gc2d-jax-cpu-comparison` with `integrate_cpu_1`, `integrate_cpu_2` and
`integrate_cpu_4`. The image pins dependencies and sets `JAX_ENABLE_X64=true` and
`JAX_PLATFORMS=cpu`. Each function sets its CPU request **and limit** to 1, 2 or
4 cores, with the same 2048 MiB memory request / 4096 MiB limit, 600-second
execution timeout, zero configured retries and one maximum container.
These limits are allocations, not exclusive physical processor reservations;
record actual hardware when interpreting scaling. See
[Modal resource limits](https://modal.com/docs/guide/resources).

```bash
modal deploy examples/modal_jax_cpu_app.py
```

The benchmark worker decodes the job once and runs one complete first integration
followed by three measured complete integrations. Reusing physical objects
preserves JAX field bindings and compiled kernels. All returned arrays are NumPy,
so timings include device synchronization. First-call setup/compilation, warm
wall times, process times, versions, CPU model and thread settings remain separate
in the diagnostics. Provisioning and network transfers are outside worker timing;
the study additionally measures the whole client call. This follows the
[JAX benchmarking guidance](https://docs.jax.dev/en/latest/benchmarking.html).

The reusable `studies.modal_cpu_comparison` module assembles a deterministic
radial-star problem, checks agreement across CPU allocations and saves all three
solutions through ordinary persistence. The local matched notebooks live under
`notebooks/developements/poincare_section/modal_jax_bm4midpoint_star_48_10_cycles_50_steps/`:
`calculation.ipynb` computes and uploads, while `visualisation.ipynb` downloads
and plots without integration. Both use the same explicit run ID and bucket
prefix derived with `solution_destination`. CPU order is 1, 2, 4 in independent
containers; warm samples characterize that particular short run, not variation
across container placements. The study compares resource scaling, not integrator
accuracy, and does not generate a DOP853 reference.

## Concurrent Poincare method/rho batches

[`examples/modal_poincare_methods_app.py`](../../examples/modal_poincare_methods_app.py)
deploys `gc2d-poincare-methods`, with at most **44 simultaneous containers** for
four methods and eleven rho values. Every container requests and
limits two CPUs, requests 2048 MiB with a 4096 MiB limit, enables float64 JAX CPU,
and has a 7200-second timeout with zero configured retries. The existing
ten-container `gc2d-poincare-rho-sweep` endpoint retains its resource settings;
both delegate to the same timed cycle-result worker implementation.

```bash
modal deploy examples/modal_poincare_methods_app.py
```

`studies.poincare_rho_batch.RhoBatchJob` describes one independent calculation
with `job_id`, `experiment_path`, `run_id` and an explicit `RhoStarConfig`.
Import both it and `run_modal_rho_batch` from their defining module. Scientific
parameters and the rho grid belong in the calculation notebook. Use canonical
method names `BM4Midpoint`, `BM4Implicit`, `RK4` and `GaussLegendre4` in `RhoStarConfig.method`.

```python
from studies.poincare_rho_batch import RhoBatchJob, run_modal_rho_batch

report = run_modal_rho_batch(
    source, jobs, record_directory="logs/poincare_methods/batch_v1",
    max_workers=33, expected_source_sha256=SOURCE_SHA256,
)
if report["failed_count"]:
    raise RuntimeError("The batch contains failed calculations; inspect progress.json.")
```

The batch fingerprints the source before submitting and loads each distinct
physical HDF5 configuration once. It derives every archive prefix with
`solution_destination`, using bucket storage by default. Each caller has its own
executor and receipt directory below `record_directory/<job_id>`. Completed
matching archives are reused; confirmed calls are recovered without spawning a
replacement. A failed job or bucket upload does not cancel sibling jobs and is
reported as failed. Publication failures retain the existing recovery archive;
its path is included when available. Numerical work never falls back to local
execution. The returned report requires JAX in the saved result diagnostics.

An atomically replaced `progress.json` records statuses, destinations, call IDs,
receipt locations and timing metadata, without credentials or raw exception
messages. `completed_count` counts validated and published archives;
`failed_count` must be zero for complete success. The configured parallelism is
a ceiling: `measured_peak_worker_overlap` is computed from returned absolute
worker start/end times intersecting the current batch invocation. Historical
archives do not count as running in the current batch. Failed workers without
returned timestamps are absent from this measured lower bound, and container
provisioning and transfer durations are outside the worker intervals.

### Additional particles inside saved Poincare gaps

`studies.poincare_gap_probes.GapProbeSeeds` specifies initial cell fractions,
persistent particle IDs, gap names and the rho used for selecting the seeds.
Pass it as `RhoBatchJob.probe_seeds` to integrate explicit positions with the
same physical field and method controls. The eight-particle extension uses
`RhoStarConfig(particles=8, arms=8, ...)` for the execution configuration;
its geometry comes exclusively from the explicit seeds, not the star constructor.
The archive records `study="poincare_gap_probes"` and the complete `probe_config`.
Reuse requires matching seed metadata as well as the original scientific controls.

The four radial-star experiments each contain a `gap_probes/` calculation and
visualisation notebook pair. Their eight positions were chosen from common empty
interiors of all four saved rho=0.30 sections and remain fixed across rho=0.00
through 0.50. Forty-four independent JAX CPU calls integrate particles 41–48 for
5,000 cycles at 50 steps per cycle. Use `max_workers=44` to submit this entire
batch concurrently; account limits and provisioning still determine actual overlap.
The original forty-particle archives remain separate. The viewers validate and
append the saved probe coordinates, preserving original particle IDs and colors.
They do not integrate trajectories while rendering or publishing.

## Receipts and recovery

Each run uses `Function.spawn`, saves the returned call ID and then waits with
`FunctionCall.get`. Modal's asynchronous calls can outlive the local process;
see [invocation durability](https://modal.com/docs/guide/function-invocation-methods).
The executor itself still presents the synchronous `run` interface.

Before submission, the executor writes a JSON receipt under `record_directory`
(relative to the current working directory by default). It records the endpoint,
creation time, protocol version and SHA-256 of the serialized input. Once Modal
confirms submission, the receipt is atomically updated with its call ID before
waiting. These files contain metadata only, not scientific results. They remain
local and ignored by Git under the default `logs/` destination.

`last_call_id` and `last_record_path` identify the most recent invocation.
Use a separate executor instance for each concurrent caller. Receipts describe
submission confirmation, not completion or the live state of a remote process.

To recover after a disconnect or restart, recreate the original problem, method
and request, and load the confirmed receipt:

```python
executor = Execution_Modal.from_record("logs/execution_modal/SAVED_RECEIPT.json")
solution = simulate(problem, method, request, execution=executor)
```

Alternatively, construct `Execution_Modal(resume_call_id="fc-SAVED_ID")` if only
the call ID is available. In either case `run` fetches that call without spawning
another integration, and `simulate` validates the returned physical result.
The worker echoes the input digest so recovery rejects a result belonging to a
different job, even when its output times and initial state happen to match.
This is an exact serialized-input comparison, not a semantic equivalence test:
equivalent objects with different serialization, or changed library versions,
can be rejected. Preserve the original construction and versions when resuming.

Modal currently retains asynchronous outputs for seven days; a receipt is not
permanent result storage. Fetch and persist completed results within Modal's
retention period. A resumed executor continues to refer to the same call on
every `run`; construct a fresh executor to start a new integration.

## Failure behavior

- Local validation, serialization and initial receipt-write errors happen before
  submission. The integration is not started by this adapter.
- If submission fails without a confirmed ID, the receipt remains unconfirmed.
  The server may already have accepted the job. Inspect Modal before resubmitting;
  `from_record` refuses to automatically restart an unconfirmed submission.
- If writing the confirmed receipt fails after submission, the exception note
  and `last_call_id` still expose the ID. Recover by that ID without resubmitting.
- Numerical and SDK exceptions keep their original type, cause and traceback.
  Notes add the call ID and receipt path. The adapter does not retry, return a
  partial `Solution`, or fall back to local execution.
- `wait_timeout` limits local waiting in seconds. `None` waits indefinitely and
  zero polls once. A wait expiry or local interrupt does not cancel the remote
  job. A remote execution-duration failure is a separate Modal exception.
- Malformed result envelopes or a mismatched input digest are rejected before
  `simulation` sees the result. `Solution` and `simulate` retain their existing
  structural, sampling and initial-state validation responsibilities.

See the official [FunctionCall API](https://modal.com/docs/sdk/py/latest/FunctionCall)
and [exception reference](https://modal.com/docs/sdk/py/latest/exception).

## Validation scope

Offline tests cover separate-process payload execution, local/remote-equivalent
RK4, BM4Implicit, DOP853 and Radau results, persistence of recovered arrays,
subclass delegation, submission uncertainty, lost connections, interrupts,
wait timeouts, receipt-write failures, malformed output and wrong-job recovery.
The deployment definition is registered and its worker is executed locally when
the optional SDK is installed. These checks do not deploy an App, invoke Modal
compute or upload scientific results to Backblaze.
