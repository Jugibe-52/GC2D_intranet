# BM4 Poincare study: 48 radial particles, 5,000 cycles

This independent study uses 48 freshly spaced initial points from the domain centre along one radius, with radius/L from 0 to 0.49. Each particle keeps its own fixed colour. The periodic HDF5 potential and original GC2D implementation are frozen in `assets/` and checked before import.

`calculo.ipynb` defines the scientific parameters, integrates and saves numerical results. `visualizacion.ipynb` loads those results without integrating. Output paths are relative to this folder. No reference trajectory is computed, as requested. Newton convergence does not certify the trajectory's integration accuracy.

The integrator is implicit BM4 with a reduced Hairer projection, analytic Newton Jacobian, float64, atol 1e-12, rtol 1e-11 and at most 40 Newton corrections. There are 20 complete steps per forcing cycle (h=0.05), 5,000 cycles, 100,000 steps and 100,001 saved states per particle. The physical units, HDF5 provenance and time/length scales are recorded in metadata. Sixteen independent spawned workers process three particles each, assigned round-robin. Linear algebra uses one thread per worker. Newton residuals and multiplier norms are measured per three-particle group.

## Run and resume

Use Python 3.11 or 3.12 and install `requirements.txt` in a virtual environment. From this directory:

```bash
python run.py calculate --run-id MY_RUN
python run.py calculate --resume --run-id MY_RUN
python run.py visualize --run-id MY_RUN
```

Use `--resume` with the same ID and unchanged parameters and source. Completed runs are verified without reintegration. Each worker commits an immutable checkpoint every 500 steps (25 cycles). It contains every accepted position, each step's convergence summary, and every Newton iteration's residual and multiplier infinity norm. Atomic data publication precedes the JSON commit marker. Checksums, initial states, contiguous step ranges and a source/parameter fingerprint are checked before resuming. Incomplete temporary files are ignored. The exact global time grid is retained across chunk boundaries.

## Numerical outputs

All numerical data are in `resultados/<run-id>/`:

- `positions_after_each_cycle.csv`: 240,000 return positions, particle IDs, fixed colours, wrapped/unwrapped coordinates and physical units.
- `initial_positions.csv`: all 48 initial conditions.
- `bm4_trajectory.npz`: all accepted positions at every step, cycle positions, colours and full Newton/multiplier arrays.
- `newton_steps.csv.gz`: 1,600,000 worker-step rows, including accepted multiplier infinity norm, correction count, residual, tolerance and convergence flag.
- `newton_iterations.csv.gz`: all residual evaluations, from the initial zero multiplier to the accepted iterate, with their multiplier infinity norms.
- `metadata.json`, `run_parameters.json`, `COMPLETE.json`: parameters, units, runtime, provenance and file integrity.
- `checkpoints/`: restart data, mirrored to private S3 during AWS calculation.

`figuras/<run-id>/` contains the Poincare section, initial positions, worker timing, Newton/multiplier diagnostic plots and a standalone HTML cycle viewer. Executed notebooks and logs are saved alongside the respective outputs. Plot downsampling affects display only; the numerical files retain all steps and iterations.

## Validation

A disposable two-cycle, 40-step run was deliberately interrupted after step 10 in every worker, then resumed for 30 steps. `validate_restart.py` compares all 48 particle trajectories, all saved times and all four per-step diagnostics bit for bit with the unmodified public BM4 integrator. Reports are in `aws/validation_restart.json`. `aws/validate_checkpoint_store.py` independently tests upload commit ordering, restoration after disk loss and rejection of corrupt remote data. This transport test uses a local S3 test double. The visualization notebook was executed against the resumed saved output.

## AWS

Target: c8a.4xlarge Spot, Frankfurt, Ubuntu 24.04, 16 workers, existing private bucket and instance role. `aws/cloud_job.py` synchronizes committed checkpoints every 30 seconds and restores them when needed. A Spot interruption can discard only progress since the last checkpoint available on the surviving disk or remote storage. An enabled system service resumes the same run after a reboot. The global six-hour deadline persists across boots. The instance shuts down after successful export or a calculation failure; failed runs retain available checkpoints and logs.

The final downloadable archive includes all numerical products and figures. The intermediate checkpoints remain in S3 separately, avoiding duplication in the archive. On completion, download the archive and its SHA256 sidecar, verify it and `COMPLETE.json`, then cancel the persistent Spot request. Do not terminate an active persistent Spot instance before cancelling its request, because that can launch a replacement. EBS and S3 storage can incur charges after the compute instance stops.
