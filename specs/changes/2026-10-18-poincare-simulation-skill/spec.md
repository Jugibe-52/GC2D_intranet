+++
schema_version = 2
kind = "change-spec"
id = "2026-10-18-poincare-simulation-skill"
revision = 1
status = "verified"
approval = "2026-10-18: User accepted specification revision 1 and plan revision 1 and explicitly requested implementation with the skill-creation authorization prefix."
sources = ["src/studies/poincare_rho_sweep.py", "src/studies/poincare_gap_probes.py", "src/initial_conditions/star.py", "src/potential/load.py", "src/dynamics/gc.py", "src/visualization/poincare_rho_sweep.py", "src/visualization/poincare_comparison.py", "src/visualization/poincare_gap_overlay.py", "src/diagnostics/persistence.py"]
documents = ["AGENTS.md", "MEMORY.md", "specs/README.md", "specs/current/potential-import.md", "specs/current/simulation.md", "specs/current/results.md", "docs/dynamics/gc2d-h5-import.md"]
requirements = ["PNC-001", "PNC-002", "PNC-003", "PNC-004", "POT-004", "POT-005", "SIM-001", "SIM-002", "SIM-003", "SIM-004", "SOL-001"]
proposed_requirements = ["PNC-001", "PNC-002", "PNC-003", "PNC-004"]
open_questions = []

[[criteria]]
id = "AC-001"
description = "Every case uses the same 48 ordered initial cell fractions, IDs and colors as the reference viewer, independently of rho, B and method."
requirements = ["PNC-001", "SIM-001"]

[[criteria]]
id = "AC-002"
description = "Resolve requested rho and method values, default B=1.5, both rescaling flags true, 5000 cycles and 50 steps per cycle; construct the explicit case matrix and reject invalid or unsupported inputs before integration."
requirements = ["PNC-002", "PNC-004", "SIM-002"]

[[criteria]]
id = "AC-003"
description = "Use cyclic HDF5 frequencies, equivalent potential/state/rho/time scaling, and B exactly once in GC dynamics; distinguish physical time from historical radial archive clocks."
requirements = ["PNC-002", "POT-004", "POT-005"]

[[criteria]]
id = "AC-004"
description = "Integrate all 48 particles together for each case, save cycle zero and every complete cycle as float64 unwrapped states, and reuse only validated complete matching archives."
requirements = ["PNC-002", "SIM-003", "SIM-004", "SOL-001"]

[[criteria]]
id = "AC-005"
description = "One HTML viewer includes all 48 particles and all requested case choices using the existing visualizer, stable colors, folded coordinates, cycle controls, selection groups and start-cycle rings."
requirements = ["PNC-001", "PNC-003"]

[[criteria]]
id = "AC-006"
description = "The viewer's effective Hamiltonian and GC velocity agree with the selected case in cell/cycle coordinates, update together, and close across 51 independently sampled phases without trajectory integration."
requirements = ["PNC-003", "POT-004"]

[[criteria]]
id = "AC-007"
description = "The discoverable simular-poincare skill uses the tested workflow, asks only for missing material inputs, separates preparation/run/view, and respects skill-creation and execution authorization boundaries."
requirements = ["PNC-004"]

[[criteria]]
id = "AC-008"
description = "Reduced scientific, persistence, viewer and workflow checks pass; historical studies remain readable and unchanged; current contracts and English/Spanish documentation match the implemented result."
requirements = ["PNC-001", "PNC-002", "PNC-003", "PNC-004", "POT-005"]
+++

# Poincare simulation skill: specification

## Problem

The reference study is composed manually from a 40-particle star and eight
separate probe integrations. Its viewer already displays all 48 particles,
but its configuration, archive paths and physical reconstruction are tied to
historical runs. Copying its notebooks would retain obsolete normalization
assumptions and hard-coded execution assertions.

The user requests a reusable skill with fixed particles and the same viewer,
selectable rho, B, model/method and numerical controls. The clarified duration
is **5000 cycles**, not 5000 integration steps. One viewer must contain all
48 particles together.

## Scope

Prepare a project-local `simular-poincare` skill and the shared study, execution
and export support needed to make it usable. The physical model is guiding
center (GC), as in the reference. Numerical method choices initially cover
`BM4Midpoint`, `BM4Implicit`, `RK4` and `GaussLegendre4`, which the existing
study already supports. Other physical models or methods require an explicit
scope extension; do not silently substitute them.

Reserve PNC-001 for the fixed particle set, PNC-002 for reproducible case
composition and execution, PNC-003 for the shared viewer, and PNC-004 for the
skill workflow. Existing POT/SIM/SOL requirements retain their meanings.

Exclude changes to integrator algorithms, a new visualization engine, changes
to the source HDF5 file, historical archive migration, production integrations
during implementation validation, cloud deployment, bucket synchronization,
personal installation, commits and pushes. Reference notebooks remain local
ignored inputs; their paths are provenance, not a future runtime dependency.

This `prepare` operation authorizes these draft records and their Spanish
copies only. Creating the skill later still requires an implementation command
starting with `&`, under the explicit personal rule for skill creation. The
SDD invocation alone does not waive that requirement.

## Proposed behavior

### Fixed particle identity: PNC-001

The baseline is
`notebooks/developements/poincare_section/bm4_midpoint_star_40_rho_0_30_folded_radial_potential/visualisation.ipynb`,
with geometry from its `calculation.ipynb` and
`gap_probes/calculation.ipynb`, inspected on 2026-10-18.

For particle `i+1`, `i=0,...,39`, use the cell fractions

```text
radius_i = 0.425 * i / 39
angle_i = 2*pi*(i modulo 8)/8
fraction_i = (0.5, 0.5) + radius_i*(cos(angle_i), sin(angle_i))
```

Append these exact fractions in the stated order:

| ID | R fraction | Z fraction | Original gap label |
| --- | --- | --- | --- |
| 41 | 0.4407 | 0.5817 | northwest |
| 42 | 0.4417 | 0.5896 | northwest |
| 43 | 0.6073 | 0.5675 | northeast |
| 44 | 0.6030 | 0.5753 | northeast |
| 45 | 0.4317 | 0.3856 | southwest |
| 46 | 0.4376 | 0.3814 | southwest |
| 47 | 0.5749 | 0.3911 | southeast |
| 48 | 0.5790 | 0.3918 | southeast |

Keep star IDs 1-40 with the 40-sample `turbo` gradient over `(0.025, 0.975)`.
Keep probe colors, in order, `#111827`, `#7c3aed`, `#e6007e`, `#92400e`,
`#0891b2`, `#0f766e`, `#d97706`, `#be123c`. Do not recolor the original star
with a 48-sample gradient. Preserve arm and probe groups. The probes were
selected at rho=0.30; their names make no claim about empty regions at other
parameters. Initial fractions, particle count and ordering are fixed for this
skill, not a configurable random seed or star size.

### Inputs and defaults: PNC-002 and PNC-004

| Setting | Resolution |
| --- | --- |
| rho | One or more explicit finite nonnegative values; ask if absent from the request/context |
| Method | One or more of the four supported methods; ask if absent; suggest the reference BM4Midpoint |
| B | One or more finite nonzero signed values; default `(1.5,)` in the source study's physical convention |
| Potential | `data/potential/V1/PHI_2.h5`, constant field 0 and original variable index `(15,)` |
| Spatial/temporal rescaling | Both true by default; explicit overrides retain physical equivalence |
| rho reference length | `lambda=0.06 m`, explicitly recorded |
| Interpolation/filtering | Cubic; `sigma=None` |
| Duration/step | 5000 forcing cycles; 50 complete integration steps per cycle |
| Saved samples | Cycle 0 and every complete cycle, including the final cycle |
| BM4 coupling | Zero by default; record its cycle-time and runtime conventions |
| Implicit solver | Analytic Jacobian; absolute tolerance `1e-12`, relative tolerance `1e-11`, at most 40 corrections; relative Jacobian step `cbrt(eps)` |
| Execution/storage | Local execution and local results by default; backend/device and explicit remote requests resolved before running |
| Viewer | Folded unit cell; playback 25 cycles/frame; field grid 64, vector grid 17, 50 phase intervals |

Solver tolerances above are in runtime coordinates and are not guaranteed
trajectory errors. Require zero coupling for RK4/GaussLegendre4. Start with one
selected positive-frequency source mode so one forcing period is well-defined;
reject unsupported multimode or constant-only cycle studies explicitly.

Use the Cartesian product of requested rho, B and method lists. Resolve common
duration and saved cycle schedule for the invocation so every case fits one
viewer. Show resolved defaults, case count, destination and total integration
steps before execution. Defaults need no repeated confirmation; request only
missing material choices. Do not silently fall back between backends/executors.

### Physical meaning and sampling: PNC-002

Let `X0` be the physical cell origin, `L=n*dx` its full width, `f_ref` the raw
cyclic HDF5 frequency in Hz, and `T_ref=1/f_ref` seconds. The public requested
rho retains the reference study's meaning:

```text
rho_m = rho * lambda / (2*pi)
s = L if spatial_rescaling else 1
T = T_ref if temporal_rescaling else 1
q = (X-X0)/s if spatial_rescaling else X
u = t/T
P(q,u) = (T/s**2) * Phi(X,t)
rho_runtime = rho_m/s
H_runtime = gyroaverage(P, rho_runtime)/B
D = T_ref/T                         # runtime duration of one forcing cycle
output_times = D * arange(cycles+1)
max_step = D/steps_per_cycle
```

Load with the current public API; all field components receive the same scale
and `Potential.frequencies` stores `f*T`. Phase remains `exp(i*2*pi*f*t)` in
physical time. B belongs exclusively to GC dynamics. There is no new standalone
`1/(2*pi)` amplitude correction. Do not reinterpret requested rho as a fraction
of the cell width: it maps to `rho_m/L` in the default unit-cell coordinates.

Integrate the 48 particles in one vectorized GC problem per case, in packed
component order `[R_1,...,R_48,Z_1,...,Z_48]`. Default saved states have shape
`(96, 5001)` and folded positions `(5001, 48, 2)`; there are exactly 250000
complete integration steps. Persist unwrapped float64 states and complete
scientific/execution provenance. Reuse only a complete, validated match;
different inputs cannot overwrite or be mistaken for an existing case.

New archives use the loader's physical time convention. At the same cycle
index, the default normalized field reproduces the historical radial
Hamiltonian after its coordinate change. That equivalence does not relabel
the historical recorded clock `2*pi/f_ref` as physical `1/f_ref`, or imply
bitwise agreement of chaotic trajectories after 5000 cycles.

### Single viewer: PNC-003

Produce one HTML entry point for the invocation, with all 48 particles in the
same return panel and the existing potential/velocity panel beside it. Reuse
the current rendering engine and controls. All requested cases appear in the
same selector; labels show rho, B and method when they vary. Incomplete cases
are unavailable rather than fabricated or silently omitted.

Keep all particles initially selected, persistent colors, arm/probe selection,
cycle zero, playback and hollow start-cycle rings. Changing case updates both
particles and field. Stored coordinates stay unwrapped; only display copies
are folded. Save every return regardless of playback stride.

Reconstruct the effective gyroaveraged Hamiltonian from the verified source
and case metadata, without reintegration. In cell/cycle display coordinates,
with runtime cell width `L_runtime=L/s`, use

```text
H_display = D * H_runtime / L_runtime**2
velocity_display = D * velocity_runtime / L_runtime
phase_j = j/50; runtime_time_j = phase_j*D, j=0,...,50
```

Sample both endpoint fields independently and verify periodic closure before
encoding them. Keep potential/arrow scales fixed across phases within each
case and label them in cell/cycle units. Never display raw Phi as the effective
Hamiltonian or divide by B twice.

### Skill operation: PNC-004

The skill resolves inputs and delegates to shared code; it does not copy large
notebooks or implement numerical physics in prompts. Provide prepare, run and
view operations. Preparing a run records its configuration without integrating;
running an explicitly requested simulation completes and validates its cases;
viewing existing results rebuilds/opens their one viewer without integration.
An execution request through the skill authorizes its necessary outputs, but
consultation and status remain read-only. Installing/creating the skill does
not itself launch the 5000-cycle default experiment.
