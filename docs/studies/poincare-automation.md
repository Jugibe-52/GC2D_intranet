# Fixed-particle Poincare automation

`$simular-poincare` prepares, runs and views GC studies with the same 48 particles
as the historical BM4Midpoint radial viewer. Forty ranked star particles and
eight fixed gap probes are integrated together in one problem per case. Their
IDs, positions and original colors remain fixed across parameters and methods.
All requested cases share one `viewer.html`, using the existing viewer engine.

## Invocation

Use the local skill or the CLI from the repository root:

```bash
.venv/bin/python scripts/run_poincare_automation.py prepare --rho 0.3 --method BM4Midpoint --B 1.5
```

This prints the resolved case count, integration-step total and new study
folder under `notebooks/developements/poincare_section/`. It writes
`configuration.json` and `cases.json`, without loading or integrating a field.
Pass `--output /absolute/path` to choose a destination. Repeating prepare accepts
only an identical configuration; changed inputs need another directory.

```bash
.venv/bin/python scripts/run_poincare_automation.py run /absolute/study/directory
.venv/bin/python scripts/run_poincare_automation.py view /absolute/study/directory
.venv/bin/python scripts/run_poincare_automation.py status /absolute/study/directory
```

Run completes missing cases and exports the viewer. View validates saved cases
and samples their effective fields without integration. Status is read-only.
The skill opens the generated viewer and reports actual results. A failed case
remains visible in the manifest, makes run exit unsuccessfully, and is disabled
in the viewer. Successful cases survive and are validated/reused on retry.
Partial local archives are preserved under an `.incomplete-...` name. A local
OS lock prevents two simultaneous writers to the same study.

## Parameters

| CLI option | Default or meaning |
| --- | --- |
| `--rho` | Required list; reference-study dimensionless gyro-radius |
| `--method` | Required list: BM4Midpoint, BM4Implicit, RK4, GaussLegendre4 |
| `--B` | List, default `1.5`; finite signed nonzero magnetic field |
| `--cycles` | `5000` forcing cycles |
| `--steps-per-cycle` | `50` complete integration steps; 250000 steps per default case |
| `--source` | `data/potential/V1/PHI_2.h5`, resolved relative to the recorded project root |
| `--source-index` | `15`; exactly one original positive-frequency variable field, plus constant field 0 |
| `--no-spatial-rescaling` | Disable the default unit-cell spatial change of variables |
| `--no-temporal-rescaling` | Disable the default physical-cycle time change of variables |
| `--reference-length` | `0.06` meters; defines the requested rho convention |
| `--sigma` | Omitted: no smoothing; otherwise nonnegative grid-sample width |
| `--interpolation-order` | `3`; `5` is also supported |
| `--coupling-frequency` | `0`; per-cycle BM4 coupling rate, required zero for classical methods |
| `--newton-absolute-tolerance` | `1e-12` in runtime coordinates |
| `--newton-relative-tolerance` | `1e-11`; solver threshold, not trajectory-error guarantee |
| `--newton-max-iterations` | `40`; analytic Jacobian, relative step `cbrt(eps)` |
| `--backend` | `scipy`; optional `jax`, with float64 enabled within the local job |
| `--device`, `--device-index` | `cpu`, `0`; unsupported choices fail without fallback |
| `--cycles-per-frame` | `25`; playback only, every return remains stored |
| `--field-grid-size`, `--vector-grid-size` | `64`, `17`; viewer sampling only |

Lists of rho, B and method form a Cartesian product with one common cycle
schedule. A study with two rho values, two B values and two methods has eight
cases. The physical model is GC; full cyclotron and additional numerical
methods are not silently substituted. Multimode and constant-only forcing
are outside this single-period workflow.

The CLI uses local execution and local archives. Python callers may provide an
explicitly configured existing `Execution` subclass to `run_automation`.
Remote prerequisites/authorization belong to that executor; the automation
never provisions services, deploys applications or synchronizes buckets.
Optional backend/device validation occurs before case execution.

## Physical meaning

Raw HDF5 frequencies are cyclic Hz. Let `T_ref=1/f_ref`, `L=n*dx` and X0 be the
physical cell origin. With both rescalings enabled:

```text
q = (X-X0)/L
u = t/T_ref
P = (T_ref/L**2)*Phi
rho_m = rho*reference_length/(2*pi)
rho_runtime = rho_m/L
H = gyroaverage(P, rho_runtime)/B
```

The full potential, including its mean, uses the same scale. B is applied once
by GC dynamics. The underlying phase is `exp(i*2*pi*f*t)` in physical time.
There is no extra legacy amplitude correction. With either flag disabled,
the study derives coordinates, runtime rho and cycle duration directly from
the current loader metadata. Coupling is mapped as a time rate as well.

Each archive stores unwrapped component-major float64 states, cycle zero and
one sample per cycle. The default shapes are `(96,5001)` for states and
`(5001,48,2)` for display positions. The manifest, exact source hash, all input
controls, particle identity and physical time provenance prevent accidental
reuse across configurations. Historical radial fields agree at corresponding
cycle indices after mapping coordinates; their archived legacy seconds are
not rewritten or presented as the corrected physical clock.

The viewer folds only display copies into the unit cell. Its left panel shows
the effective gyroaveraged Hamiltonian and GC velocity in cell/cycle units,
with 51 phases and audited endpoint closure. The right panel contains all
48 particles, start-cycle rings and every saved return. Changing case updates
both panels. A source fingerprint mismatch prevents field reconstruction.

## Implementation and validation

The shared composition is in `studies.poincare_automation`, manifest/reuse in
`studies.poincare_automation_io`, and export in
`visualization.poincare_automation`. The viewer reuses
`export_poincare_panel_comparison`; the historical notebooks and archives remain
independent. Development outputs remain ignored by Git.

Use reduced 1-2 cycle studies for routine validation. Tests cover exact particle
identity, defaults without a full run, all four methods and rescaling choices,
DOP853/Radau reference agreement, archive reuse, failure recovery and viewer
payloads. Full-duration production runs require a simulation request; creating
or validating the skill does not launch them.
