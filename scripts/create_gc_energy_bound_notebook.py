"""Build the executable development notebook for one HDF5 guiding-center orbit."""

from pathlib import Path
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(value: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(value).strip()))


def code(value: str) -> None:
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(value).strip()))


md(r"""
# Does the energy error reach a bounded envelope?
## One guiding-center trajectory from PHI_2.h5: BM4Implicit versus classical RK4

This study uses the **actual project HDF5 potential** and the project solvers.
The initial condition is one point at radius **0.2 cell periods** from the cell center,
at angle zero. Its velocity is determined by the guiding-center field.

**Run:** open this notebook inside the GC2D project, select the project's `.venv`
Python kernel, and run the cells in order. The project must be installed in that
environment (the repository setup is `python -m pip install -r requirements.txt`).
No numerical method or potential is reimplemented in this notebook.

The canonical parameters request **200 normalized cycles**, three fixed step sizes,
an independently audited DOP853 reference, three alternating timing repetitions,
and energy observations at every accepted step. This can take substantial time.
Change the explicit parameters before running. Development validation uses a short
disposable copy; such reduced results are never stored in this canonical notebook.

**Scope:** one trajectory and two methods are deliberate user-requested departures
from the standard three-particle/five-method comparison. All applicable shared
physical, reference, nonlinear, timing, periodic-distance and sampling controls
are retained. Notebook prose and plot labels follow the project's English-language convention.
""")

md(r"""
## 1. Physical energy is not an invariant of this time-dependent field

The project convention is

$$H(t,x,y)=\langle\Phi\rangle_\rho(t,x,y),\qquad
\dot x=-\partial_y H,\quad \dot y=\partial_x H.$$

With `indx=(0,1)`, the loaded potential includes the mean and the dominant
positive-frequency mode. Consequently,

$$\frac{dH(t,z(t))}{dt}=\partial_tH(t,z(t)),$$

so **$H(t,z_n)-H(0,z_0)$ is not a conservation error**. We distinguish:

| Quantity | Definition | Interpretation |
|---|---|---|
| Physical variation | $H(t,z_n)-H(0,z_0)$ | Includes true time dependence |
| Physical-energy error | $H(t,z_n)-H(t,z_{\rm DOP853})$ | Accuracy against the same interpolated ODE |
| Extended balance | $\kappa'=-\partial_tH$, $K=H+\kappa$, $\kappa(0)=0$ | $K$ is constant in the continuous augmented system |
| Envelope | $E_K(T,h)=\max_{0\le t_n\le T}|K_n-K_0|$ | Does the largest error continue to grow? |
| Trajectory error | minimum-image distance to DOP853 | Energy accuracy does not establish orbit accuracy |

BM4's $\kappa$ is reconstructed by `GCGeneralizedEnergyObserver` from its twelve
accepted base stages, with the project's $\kappa=k/2$ normalization. No sparse-output
quadrature is used. RK4 advances $\kappa$ with the same four RK stages as its state.

`BM4Implicit` uses one reduced Hairer projection around the complete BM4 composition.
It accepts the **physical state only**; the observer does not turn it into a fully
extended solver. A small physical symplectic defect at fixed time does not prove
that the reconstructed $(x,y,t,\kappa)$ map is symplectic.

The familiar $O(h^4)$ long-time energy estimate requires a suitably regular
autonomous Hamiltonian, a symplectic map, a small constant step, and a bounded
regularity domain. Those hypotheses must not be assumed from a finite plot.
In particular, cubic spatial splines have limited regularity, and Newton error,
roundoff, reference resolution and chaotic separation can influence the measured
envelopes. The $h^4$ lines below are comparison guides, not guaranteed bounds.
""")

code('''
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
from uuid import uuid4
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

from diagnostics.paths import find_project_root, notebook_output_directory
from diagnostics.reference_trajectory import load_reference_trajectory
from diagnostics.reference_progress import reference_progress_log
from diagnostics.gc_energy_bound import save_energy_bound_result, load_energy_bound_result
from potential import load_gc2d_h5_potential
from dynamics import GuidingCenterDynamics
from studies.initial_conditions import radial_gc_configuration, domain_center
from studies.reference_trajectory import HighPrecisionReferenceConfig, run_high_precision_reference_trajectory
from studies.gc_energy_bound import (
    GCEnergyBoundConfig, run_gc_energy_bound_study, audit_initial_step_geometry,
    energy_bound_conclusions,
)
from visualization.radial_reference import plot_radial_reference_audit
from visualization.notebooks import display_animation
from visualization.gc_energy_bound import (
    show_energy_table, plot_energy_histories, plot_energy_refinement, plot_energy_blocks,
    plot_gc_accuracy_and_cost, plot_bm4_nonlinear_work, animate_energy_orbit,
)

ROOT = find_project_root(Path.cwd())
NOTEBOOK_PATH = ROOT / "notebooks/developements/BM4_RK4_ejecutable.ipynb"
RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
OUTPUT_DIRECTORY = notebook_output_directory(NOTEBOOK_PATH, project_root=ROOT) / RUN_TAG
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.2})
print("Python:", sys.executable)
print("Project:", ROOT)
print("Run output:", OUTPUT_DIRECTORY)
''')

md(r"""
## 2. Explicit and editable experiment parameters

One unit of normalized time is the selected characteristic period $T_0=2\pi/\omega_0$.
The physical coordinate scale is $\lambda=0.06$ and
$\hat x=2\pi(R-R_0)/\lambda$. The potential normalization is
$\hat\Phi=(2\pi)^2\Phi/(\omega_0\lambda^2B)$.
Actual dimensional conversion factors are printed from HDF5 metadata below.
The initial spatial radius fraction is distinct from the normalized gyro-radius `RHO`.
There is no random sampling and therefore no seed.

Keep the duration and every horizon divisible by every step, and each step an integer
multiple of the finest step. The reference must save that finest grid. Reducing the
step never changes it during one trajectory and never introduces a shortened final step.

For a first short run, use `T_END=2.0` and `HORIZONS=(0.5, 1.0, 2.0)`.
For a stationary HDF5 control, explicitly change `FIELD_INDICES` to `(0,)`:
that studies the mean field and must be labelled as a different experiment.
""")

code('''
# Measured field: preserve the native spatial resolution and use cubic splines.
H5_PATH = ROOT / "data/potential/V1/PHI_2.h5"
MAGNETIC_FIELD = 1.5
CHARACTERISTIC_LENGTH = 0.06
FIELD_INDICES = (0, 1)
INTERPOLATION_ORDER = 3
RHO = 0.3
INITIAL_RADIUS_FRACTION = 0.2  # Fraction of the whole cell period, one point only.
INITIAL_ANGLE = 0.0            # Radians from +x.

T_START, T_END = 0.0, 200.0
STEPS = (0.1, 0.05, 0.025)
HORIZONS = (1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0)
COUPLING_FREQUENCY = float(np.pi / 8)
NEWTON_ATOL, NEWTON_RTOL = 1e-12, 1e-11
NEWTON_MAX_ITERATIONS = 40
JACOBIAN_RELATIVE_STEP = float(np.cbrt(np.finfo(float).eps))
TIMING_REPEATS = 3
BLOCK_COUNT = 10
PLATEAU_RELATIVE_GROWTH = 0.10  # Descriptive last-window criterion; never a theorem.

DOP853_RTOL, DOP853_ATOL, DOP853_MAX_STEP = 1e-10, 1e-12, 0.025
RADAU_RTOL, RADAU_ATOL, RADAU_MAX_STEP = 1e-11, 1e-13, 0.0125
REFERENCE_DIRECTORY = None  # Optional existing reference directory printed by a previous run.
DISPLAY_LEVEL = 0           # 0 uses the coarsest step; change to inspect another refinement.
ANIMATION_FRAMES, ANIMATION_FPS = 201, 10

config = GCEnergyBoundConfig(
    t_span=(T_START, T_END), steps=STEPS, horizons=HORIZONS, rho=RHO,
    coupling_frequency=COUPLING_FREQUENCY, newton_atol=NEWTON_ATOL,
    newton_rtol=NEWTON_RTOL, newton_max_iterations=NEWTON_MAX_ITERATIONS,
    jacobian_relative_step=JACOBIAN_RELATIVE_STEP, timing_repeats=TIMING_REPEATS,
    block_count=BLOCK_COUNT, plateau_relative_growth=PLATEAU_RELATIVE_GROWTH,
)
reference_config = HighPrecisionReferenceConfig(
    t_span=(T_START, T_END), save_interval=STEPS[-1], rho=RHO,
    relative_tolerance=DOP853_RTOL, absolute_tolerance=DOP853_ATOL,
    maximum_step=DOP853_MAX_STEP, audit_relative_tolerance=RADAU_RTOL,
    audit_absolute_tolerance=RADAU_ATOL, audit_maximum_step=RADAU_MAX_STEP,
    distance_convention="periodic",
)
assert H5_PATH.is_file(), f"Missing measured HDF5 potential: {H5_PATH}"
assert 0 <= DISPLAY_LEVEL < len(STEPS)
print("Fixed steps per method:", [round((T_END-T_START)/h) for h in STEPS])
print("Reference saved states:", reference_config.output_sample_count)
''')

code('''
potential = load_gc2d_h5_potential(
    H5_PATH, B=MAGNETIC_FIELD, characteristic_length=CHARACTERISTIC_LENGTH,
    indx=FIELD_INDICES, interpolation_order=INTERPOLATION_ORDER,
    nx=None, ny=None, denoising=False, spatial_normalization="characteristic_length",
)
configuration = radial_gc_configuration(
    potential, radial_fractions=(INITIAL_RADIUS_FRACTION,), angle=INITIAL_ANGLE,
)
assert configuration.initial_state.shape == (2,)
dynamics = GuidingCenterDynamics(potential, rho=RHO)
initial_state = configuration.initial_state
initial_rows = [{"particle": 1, "x0": float(initial_state[0]), "y0": float(initial_state[1]),
                 "radial_fraction": INITIAL_RADIUS_FRACTION, "angle": INITIAL_ANGLE}]
show_energy_table(initial_rows, ("particle", "x0", "y0", "radial_fraction", "angle"))
source = potential.metadata
potential_metadata = {
    "source_path": str(H5_PATH), "B": MAGNETIC_FIELD,
    "characteristic_length": CHARACTERISTIC_LENGTH, "indx": list(FIELD_INDICES),
    "interpolation_order": INTERPOLATION_ORDER, "denoising": False,
    "spatial_normalization": "characteristic_length", "resampling": None,
    "source_field_indices": source.source_field_indices.tolist(),
    "source_frequencies": source.source_frequencies.tolist(),
    "characteristic_period": source.characteristic_period,
    "characteristic_frequency": source.characteristic_frequency,
    "normalization_factor": source.normalization_factor,
    "runtime_frequencies": potential.frequencies.tolist(),
    "grid_shape": list(potential.grid.shape), "cell_period": potential.grid.period,
}
initial_metadata = {"geometry": "one point on one radius from cell center",
                    "radial_fraction": INITIAL_RADIUS_FRACTION, "angle": INITIAL_ANGLE,
                    "initial_state": initial_state.tolist(), "random_seed": None}
print(json.dumps(potential_metadata, indent=2))
print("Initial Hamiltonian:", dynamics.hamiltonian(T_START, initial_state))
print("Initial partial_t H:", -dynamics.extended_momentum_derivative(T_START, initial_state))
print("Initial GC velocity:", dynamics.vector_field(T_START, initial_state))
''')

md(r"""
## 3. Local structural check and reference resolution

The first-step Jacobian is measured with two centered finite-difference increments.
BM4 is identified through its twelve-stage accepted record and its reduced projection.
The physical defect is $\|D\Phi^T\Omega D\Phi-\Omega\|_F$; its measured value includes
finite-difference and nonlinear-solve errors. It is not an audit of the reconstructed
extended state and it is not extrapolated to the entire orbit.

The reference uses the existing high-precision DOP853/Radau pipeline. Their discrepancy
is a **measured resolution indicator**, not a rigorous error bound. The pipeline
persists both trajectories, solver settings and the fingerprint of the actual
gyroaveraged interpolated field. A reused reference must match the field, radius,
initial state, solver controls and finest time grid.
""")

code('''
geometry_audit = audit_initial_step_geometry(potential, configuration, config=config)
show_energy_table(geometry_audit,
    ("method", "step", "fd_relative_step", "determinant", "symplectic_defect"))
''')

code('''
if REFERENCE_DIRECTORY is None:
    with reference_progress_log(OUTPUT_DIRECTORY / "reference_progress.log",
                               t_span=config.t_span, interval_seconds=10.0) as report:
        computed_reference = run_high_precision_reference_trajectory(
            potential, configuration, notebook_path=NOTEBOOK_PATH, config=reference_config,
            potential_metadata=potential_metadata, initial_condition_metadata=initial_metadata,
            reference_name="gc_energy_1p_" + RUN_TAG, version="v1", project_root=ROOT,
            overwrite=False, progress_callback=report,
        )
    reference = computed_reference.trajectory
else:
    reference = load_reference_trajectory(REFERENCE_DIRECTORY)
    # Compare serialized controls, allowing tuple-to-list conversion in JSON.
    expected_config = json.loads(json.dumps(asdict(reference_config)))
    stored_config = json.loads(json.dumps(dict(reference.metadata["config"])))
    if stored_config != expected_config:
        raise ValueError("Reused reference solver settings or saved grid do not match this notebook.")
print("Reusable reference directory:", reference.paths.directory)
print("Reference field SHA-256:", reference.metadata["dynamics_fingerprint_sha256"])
print("Maximum per-particle DOP853/Radau distance:", reference.audit_distances.max(axis=1))
figure, _ = plot_radial_reference_audit(reference, potential, np.asarray(domain_center(potential)))
figure.savefig(OUTPUT_DIRECTORY / "initial_position_and_reference_audit.png", dpi=160)
plt.show()
''')

md(r"""
## 4. Fixed-step comparison and persistence

The same physical orbit is integrated independently by BM4 and RK4 on each grid.
Every accepted step is saved, so envelopes and projection maxima include all nodes.
Reference samples are selected from the finest common grid without interpolating
the numerical method or restarting it at the intermediate horizons.

The three timing repetitions alternate method order and use one BLAS thread.
Reference construction and energy observers are excluded. An untimed replay obtains
the energy histories and must reproduce each timed physical trajectory. BM4's one
nonlinear solve per complete step and its residual acceptance are checked explicitly.
RK4 is excluded from nonlinear-work statistics.

The output HDF5 contains the reference, Radau audit, all numerical states and all
energy, work and projection histories. CSV tables are exported alongside it.
""")

code('''
result = run_gc_energy_bound_study(potential, configuration, config=config, reference=reference)
result.metadata.update({"source": potential_metadata, "initial": initial_metadata,
                        "geometry_audit": geometry_audit, "source_notebook": str(NOTEBOOK_PATH)})
RESULT_PATH = save_energy_bound_result(OUTPUT_DIRECTORY / "gc_h5_energy_bound.h5", result)
print("Full study and audited reference saved to:", RESULT_PATH)

show_energy_table(result.summary, ("method", "step", "trajectory_rms", "trajectory_final",
    "H_error_rms", "H_error_max", "K_error_max", "reference_H_floor", "reference_distance_floor"))
''')

md(r"""
## 5. Energy histories: distinguish physical change, error and balance

The top-left plot can vary even when the reference is accurate. The top-right plot
measures physical-energy accuracy. The lower plots test the extended-balance
envelope. A constant-looking curve on a logarithmic scale must be checked against
the horizon table and the signed block ranges below.
""")

code('''
figure = plot_energy_histories(result, level=DISPLAY_LEVEL)
figure.savefig(OUTPUT_DIRECTORY / "energy_histories.png", dpi=160)
plt.show()
show_energy_table([row for row in result.envelopes if row["step"] == STEPS[DISPLAY_LEVEL]],
                 ("method", "metric", "horizon", "maximum", "growth_ratio"))
''')

code('''
figure = plot_energy_blocks(result, level=DISPLAY_LEVEL)
figure.savefig(OUTPUT_DIRECTORY / "energy_blocks.png", dpi=160)
plt.show()
''')

md(r"""
## 6. Refining h at fixed T

The empirical slope is fitted to the three finest steps separately for every horizon.
It need not equal four for a particular diagnostic, a long chaotic trajectory, or a
piecewise polynomial field. Check whether physical-energy and trajectory errors lie
well above their DOP853/Radau discrepancy before interpreting them. Small errors near
that floor cannot establish a convergence rate. There is no reference-derived floor
for the BM4 observer's momentum quadrature; its balance is evaluated against the
constant continuous $K_0$ and needs its own step/tolerance sensitivity check.
""")

code('''
figure = plot_energy_refinement(result)
figure.savefig(OUTPUT_DIRECTORY / "energy_refinement.png", dpi=160)
plt.show()
show_energy_table(result.orders, ("method", "metric", "horizon", "slope"))
''')

md(r"""
## 7. Trajectory accuracy, runtime and nonlinear work

Coordinates are unwrapped in the orbit plots. Error distances use the minimum-image
periodic convention, so crossing a cell edge does not create an artificial jump in
the reported separation. Runtime ratios below compare the same step and saved grid.
Projection multipliers are numerical corrections, not physical forces.
""")

code('''
figure = plot_gc_accuracy_and_cost(result, level=DISPLAY_LEVEL)
figure.savefig(OUTPUT_DIRECTORY / "trajectory_and_cost.png", dpi=160)
plt.show()
show_energy_table(result.summary, ("method", "step", "runtime_median", "runtime_q25", "runtime_q75"))
for step in STEPS:
    samples = {row["method"]: row["runtime_median"] for row in result.summary if row["step"] == step}
    print(f"h={step:g}: median BM4/RK4 runtime ratio = {samples['BM4Implicit']/samples['RK4']:.5g}")

figure = plot_bm4_nonlinear_work(result, level=DISPLAY_LEVEL)
figure.savefig(OUTPUT_DIRECTORY / "bm4_nonlinear_work.png", dpi=160)
plt.show()
implicit_rows = [row for row in result.summary if row["method"] == "BM4Implicit"]
show_energy_table(implicit_rows, ("step", "solves_per_step", "newton_mean", "newton_max",
    "newton_total", "residual_evaluations_mean", "residual_evaluations_total", "residual_tolerance_ratio_max"))
show_energy_table(implicit_rows, ("step", "mu_mean", "mu_rms", "mu_max", "mu_final"))
''')

code('''
animation = animate_energy_orbit(result, level=DISPLAY_LEVEL,
                                 frame_count=ANIMATION_FRAMES, fps=ANIMATION_FPS)
display_animation(animation, interactive=True)
''')

md(r"""
## 8. What the completed run supports

The conclusion below is derived from computed values, including the last-interval
envelope growth. The descriptive threshold is explicit in the configuration.
Even an apparent plateau over 200 cycles does not prove an infinite-time bound.
Extend the horizon at fixed h, compare the block means/ranges, and repeat with
tighter Newton tolerances before attributing slow growth to the method itself.
For autonomous conservation from the same data source, repeat the experiment with
the mean-only `(0,)` field and label that change explicitly.
""")

code('''
conclusions = energy_bound_conclusions(result)
display(Markdown(conclusions))
(OUTPUT_DIRECTORY / "conclusions.md").write_text(conclusions + "\\n", encoding="utf-8")
print("To reopen a completed run without solving again:")
print(f"result = load_energy_bound_result({str(RESULT_PATH)!r})")
''')

md(r"""
## Project sources and interpretation

- [HDF5 import and units](../../docs/dynamics/gc2d-h5-import.md).
- [Physical guiding-center dynamics](../../src/dynamics/gc.py).
- [BM4Implicit method and projection](../../docs/models/bm4-implicit/simulation/bm4-simulation-architecture.md).
- [BM4 accepted-stage energy observer](../../src/diagnostics/energy/observer.py).
- [Classical RK4 and optional momentum evolution](../../src/methods/classical/rk4.py).
- [DOP853/Radau reference pipeline](../../src/studies/reference_trajectory.py).
- Hairer, Lubich and Wanner, *Geometric Numerical Integration*, IX.8: the long-time
  near-conservation result is conditional; it does not assert conservation of a
  time-dependent physical Hamiltonian.

This notebook studies the same interpolated guiding-center model as the project.
It does not validate the underlying measured field or the guiding-center approximation.
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3 (GC2D project .venv)", "language": "python"},
    "language_info": {"name": "python", "version": "3.11"},
})
nbf.validate(nb)
path = ROOT / "notebooks/developements/BM4_RK4_ejecutable.ipynb"
nbf.write(nb, path)
print(f"Created {path}: {len(cells)} cells, ready for a fresh run.")
