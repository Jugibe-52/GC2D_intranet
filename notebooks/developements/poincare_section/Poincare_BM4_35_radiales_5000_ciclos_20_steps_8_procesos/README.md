# BM4 Poincare study: 5,000 cycles and 20 steps per cycle

This self-contained study preserves 35 non-interacting radial particles, the
frozen GC2D potential, implicit BM4, eight worker processes, float64 arithmetic,
Newton tolerances and fixed particle colours from the previous study. No
reference solution is computed, as explicitly requested in this conversation.

## Status

Prepared locally; the 5,000-cycle production simulation has not been launched.
A two-cycle validation with all 35 particles and eight processes passed. Both
notebooks executed successfully; CSV and NPZ diagnostics matched exactly. Five
steps per group matched the uninstrumented solver exactly (zero difference).
Mean Newton corrections in this short run: 1.996875; maximum: 2. This does not
certify convergence or trajectory accuracy across the full 5,000-cycle horizon.
Small validation runs are explicitly labelled and never become `latest_success`.
Previous studies and their downloaded results are unchanged.

## Calculation and visualization

`calculo.ipynb` defines every physical and numerical parameter and saves outputs
to `resultados/<run-id>/`. `visualizacion.ipynb` loads those files without calling
the integrator, and writes plots and an offline viewer to `figuras/<run-id>/`.
Paths are relative to this folder, which includes the potential and frozen code.

The production run has 100,000 complete steps, 100,001 saved trajectory nodes
per particle, and 175,000 Poincare return positions (5,000 per particle).
The normalized step is 0.05, compared with 0.02 in the previous 2,000-cycle run.
The step count is unchanged, but a larger step may require more Newton work;
Newton convergence does not establish trajectory accuracy.

Use Python 3.11 or 3.12 on Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run.py calculate --run-id production_5000cycles --timeout 18000
.venv/bin/python run.py visualize --run-id production_5000cycles --timeout 18000
```

For a short validation, use `run.py calculate --smoke --run-id validation_new`.
This changes only the cycle count to two in an in-memory notebook copy.

## Complete Newton and multiplier diagnostics

Each particle group solves one reduced projection equation per complete step.
The norm of mu is the infinity norm of that group's multiplier; it is not a
per-particle value. The following are saved with numerical checksums:

- `newton_steps.csv.gz`: worker, step, cycle, normalized/physical time, Newton
  corrections, residual evaluations, initial/final residual infinity norms,
  tolerance, residual/tolerance, final multiplier infinity norm and convergence.
- `newton_iterations.csv.gz`: worker, step, time, iteration, residual infinity
  norm, tolerance, residual/tolerance, multiplier infinity norm and accepted flag.
  Iteration zero is the initial zero multiplier, before any Newton correction.
- `bm4_trajectory.npz`: complete trajectories and the same diagnostics, including
  lossless ragged iteration arrays and absolute offsets per worker and step.
- `metadata.json`: per-worker mean, RMS, maximum and final multiplier norms,
  Newton-work summaries, version information, physical units and source hashes.

`newton_diagnostics.py` installs an observational callback in a private copy of
the verified frozen solver, restored after integration. The callback reads
already-computed values; it does not repeat field/map evaluations or change
Newton updates, tolerances or accepted states. The snapshot itself is unchanged.
CSV histories include every recorded iterate and every complete step; plotting
uses block maxima for long step histories while retaining all downloaded data.
Worker progress is emitted every 1,000 accepted steps.

## AWS recommendation

Recommended: **c8a.2xlarge**, eight physical cores and 16 GiB RAM, in Frankfurt
(`eu-central-1`), keeping eight worker processes and one BLAS thread per worker.
AWS advertises up to 30% higher performance than C7a; this is not a measured
speedup for this code. A short run on the chosen instance should precede any
runtime estimate. Availability in a particular availability zone and the
account's current quota must be checked at launch.

Sources checked on 2026-09-20:

- https://aws.amazon.com/ec2/instance-types/c8a/
- https://aws.amazon.com/about-aws/whats-new/2026/02/amazon-ec2-c8a-instances-europe-frankfurt-europe-ireland-regions/

`aws/cloud_job.py` runs both notebooks and includes both complete compressed
CSV diagnostics in the private result archive. It requires boto3 in its launcher
environment and the existing scoped project S3 role. The notebook interpreter
is `.venv/bin/python`. Deployment, download verification and instance shutdown
are recorded separately from the calculation. No cloud deployment is recorded
until it has actually occurred.
