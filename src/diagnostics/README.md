# Numerical diagnostics

## Complete solutions on disk or in a bucket

`diagnostics.persistence.save_solution` and `load_solution` serialize the
canonical `Solution` independently of the integration lifecycle. They are also
explicitly exported by `diagnostics`. Numerical methods and `Solution` do not
perform network operations. Both GC and FC component-major layouts are supported;
other layout types are rejected rather than inferred from the state size.

```python
from diagnostics.persistence import load_solution, save_solution
from diagnostics.paths import solution_destination

# Bucket storage is the default. Remote credentials stay outside the repository.
location = save_solution(solution, experiment_path="developements/my_experiment",
                         run_id="run_001", metadata=experiment_metadata,
                         potential=base_potential)

# For a deliberately local run, explicitly select its destination instead:
# destination = solution_destination("developements/my_experiment", "run_001", storage="local")
# location = save_solution(solution, destination, metadata=experiment_metadata,
#                          potential=base_potential)

record = load_solution(location)
solution = record.solution
base_potential = record.potential
experiment_metadata = record.metadata
```

`solution_destination(experiment_path, run_id)` defaults to
`gc2d_data:gc2d-notebooks-data/<experiment_path>/<run_id>`. Pass `storage="local"`
to select `outputs/`, or `bucket_root="other_remote:other_bucket"` to select
another configured bucket. Explicit destinations to `save_solution` remain
supported. When omitting the destination, the experiment path and run ID are
required; missing identifiers are never guessed. Failed bucket uploads raise an
error even when a local recovery archive is retained.

The remote syntax is rclone's `remote:bucket/prefix`, not an S3 URL. The transport
uses the configured rclone executable on `PATH`, falling back to
`~/.local/bin/rclone`. Credentials stay in rclone's configuration (normally
`~/.config/rclone/rclone.conf`); they are not copied into artifacts. A local load
does not require rclone or bucket credentials. See the official
[rclone remote syntax](https://rclone.org/docs/) and
[copyto documentation](https://rclone.org/commands/rclone_copyto/).

Choose the experiment prefix from its directory relative to `notebooks/`.
For `notebooks/developements/my_experiment/`, use
`developements/my_experiment/<run_id>/` below the bucket or local `outputs/`.
Calculation and visualisation notebooks share that prefix and run identifier.

Each result contains:

- `solution.npz`: saved times, physical states, optional initial state and all
  diagnostic arrays, preserving their dtypes and shapes.
- `metadata.json`: schema, layout, scalar diagnostics, software versions,
  potential grid/interpolation/provenance and caller-supplied experiment metadata.
- `potential.npz`, when a potential is supplied: the actual processed mean,
  complex modes and frequencies, plus typed HDF5 attributes when applicable.
- `manifest.json`: the allowed file inventory, byte sizes and SHA-256 hashes.

The potential snapshot reconstructs the actual sampled field even when its
original HDF5 file is unavailable. Pass the **base** field and record `rho` and
the dynamics parameters to reconstruct the effective gyroaveraged field. Record
the complete potential recipe, physical settings, method settings, initial
geometry and code version in `experiment_metadata` as well: a `Solution` alone
does not own them. Reconstruction does not run the integrator. The archive records
NumPy and SciPy versions; matching samples do not guarantee identical interpolation
across different library implementations. Generic potential provenance must be
JSON-serializable; HDF5 attributes and numerical diagnostics must be pickle-free.

Completed results are never overwritten. Use a fresh run directory/prefix, and
distinct prefixes for concurrent writers. The manifest is transferred last;
loaders verify every file before constructing a `Solution`. A remote load always
downloads to a fresh temporary directory and removes it afterwards. If publication
fails, the exception identifies a complete local staging directory that can be
loaded directly or used for a manual retry, transferring the manifest last.
Incomplete destinations are not valid results. Local transport uses exclusive
file creation, and remote transfers use `copyto --immutable --checksum`.

Development examples live in `notebooks/developements/persistence_local/` and
`notebooks/developements/persistence_bucket/`, each with `calculation.ipynb` and
`visualisation.ipynb`. Select the `Python (GC2D)` kernel. Their common composition
is `studies.persistence_demo`; presentation is
`visualization.persistence.plot_stored_gc_solution`. These short synthetic-field
examples demonstrate storage and independent loading, not numerical-method
accuracy. Their local `.gitignore` files keep them and their output data untracked.

## Observers and study-specific persistence

This package contains opt-in numerical diagnostics used by reproducible studies.
Production methods emit neutral integration-stage or complete-step
observations; Jacobian calculations, symplecticity metrics, and output
persistence live here so they cannot affect simulations unless an observer is
explicitly passed. Generic complete-step maps use centered differences.

`five_method_comparison_csv` stores the complete five-method long-time study in
one versioned CSV. Each data row represents one saved time, with explicit
columns for all reference and numerical trajectories, particle errors, energy
errors, and step-aligned nonlinear or projection diagnostics. The first row
also carries the reproducibility configuration, timing samples, summaries, and
execution log as JSON metadata, allowing visualization without reintegration.
`parallel_bm4_recurrence_npz` stores dense parallel recurrence trajectories,
worker summaries, configuration, and experiment metadata in one compressed,
schema-versioned NPZ archive. Its loader reconstructs the validated recurrence
result so calculation and visualization notebooks can run independently.
Implicit ABBA step observations additionally expose their converged stages, so
diagnostics can select either the implicit-function factorization or the
equivalent stage-increment factorization of the ideal-root tangent. These step
observations also expose the selected nonlinear solver, correction and residual-
evaluation counts, and accepted residual so solver-work observers do not repeat
the nonlinear solve.

`diagnostics.ImplicitABBAIterationObserver` records nonlinear iterations,
explicit residual evaluations, final infinity-norm residual, effective stopping
tolerance, residual-to-tolerance ratio, and projection-multiplier norm for
selected accepted steps. It supports both implicit ABBA formulations, including
Newton and Broyden, and defaults to sampling every complete step. Compatibility
fields retain the former `newton_*` spelling while solver-neutral properties
and arrays use `nonlinear_*` names.

`diagnostics.ImplicitBM4IterationObserver` uses the same persisted record
schema for `BM4Implicit`. Each observation represents the
single Hairer projection solve surrounding one complete twelve-stage BM4
cycle. The observer does not perform additional BM4 maps or finite-difference
Jacobians.

`diagnostics.abba_jacobian.ImplicitABBAJacobianObserver` is independent of the
symplecticity studies. It records the local physical Jacobian of selected
complete implicit ABBA steps, extracts one `2 x 2` block per independent GC
particle, classifies the characteristic discriminant, and persists eigenvalue,
eigenvector, and singular-value decompositions. Complex eigenvectors are kept
in the NPZ arrays but are not assigned real line angles. The observer does not
accumulate tangent maps and does not calculate area or symplecticity metrics.

`diagnostics.symplecticity.SymplecticityObserver` studies the two-copy GC
state with `track_energy=False`. It writes indexed blocks below:

```text
outputs/<notebook folder>/<notebook stem>/<YYYY-MM-DD>/
```

Each block consists of a scalar CSV summary, compressed arrays in NPZ format,
and versioned JSON metadata. `diagnostics.output.write_diagnostic_block`
provides the shared synchronized writer. Existing blocks are never overwritten.

`diagnostics.projection.ProjectedSymplecticityAreaObserver` propagates the
tangent of the complete physical map ``P Phi E`` from the initial GC boundary.
At every selected complete BM4 step it records the projected Jacobian, its
symplectic defect, the separation between internal copies, and the transported
polygon area.

`diagnostics.symplecticity.GCAreaSymplecticityObserver` differentiates
complete physical GC steps. It records both the local symplecticity defect of
each numerical step and the accumulated defect, determinant drift, and
transported area of the discrete flow. Its `jacobian_method` is one of
`finite_difference`, `implicit_function`, or `stage_increment`; the two
analytic choices require an implicit ABBA step observation.

Method comparisons can select a subset with `method_names` and skip adaptive
integration with `reused_reference` in `run_five_method_comparison`. The caller
checks physical provenance; the runner requires identical saved times and initial
states. When several integration steps fall between saved states, the CSV stores
each diagnostic in interleaved substep columns described by `diagnostic_substeps`.
The loader restores the full step histories; older CSV files default to one step
per saved interval. Newton plots use the complete integration step grid.
Passing `reused_audit_reference` instead recomputes DOP853 with the configured
tolerances and maximum step while retaining the aligned Radau audit history.
Set `parallel_models=True` to assign each selected model campaign to a separate
thread. Repetitions remain sequential within each model, and recorded runtimes
represent elapsed time under concurrent CPU load.

## Buffered observer lifecycle

`diagnostics.buffering.DiagnosticBuffer` owns pending samples, output block
indices, the common writer, and finalization for stage symplecticity, physical
area, projected area, local ABBA Jacobians, implicit iterations, and trajectory
symplecticity. Each observer supplies only its diagnostic calculation and block
schema through composition. Public records, samples, output block descriptors,
chunk sizes, and existing file schemas remain unchanged.

Closing an observer writes its final partial chunk once. Observers with a
complete-step cadence also retain the last completed step before closing.
Successful writes release pending samples; an unsuccessful write retains them
and its index for an explicit retry. If integration already raised, a cleanup
failure is attached as an exception note instead of replacing the original
error, and the observer stops accepting new events.
After such an exceptional exit, call `flush()` to retry the pending write at
the same index; `close()` remains idempotent, and new events remain rejected.

Interactive retention remains separate from pending-output buffering. In
particular, `ImplicitABBAJacobianObserver.samples` intentionally retains every
matrix-valued sample after writing; `chunk_size` controls pending writes and
does not limit that documented history.
