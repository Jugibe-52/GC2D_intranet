# Implemented and checked

On 15 November 2026 the user accepted preparation and execution of the agreed
GC2D code change, with reflected as the default and periodic selectable, and
explicitly excluded TFM_redaccion. The revised four documents were presented
before implementation; preparation validation passed (198 records, 22 requirements).

## Delivered behavior

Potential, its factories, JAX and new study configurations default to
spatial_extension="reflected"; "periodic" retains the old construction.
Constrained cubic splines give a C2 reflected Hamiltonian and C1 canonical field
with period 2L. Normal derivatives vanish along complete edges; trajectories
and intermediate states remain unwrapped and may cross them.

Gyroaveraging uses the doubled reflected samples and preserves independent upper
edges through scaling, snapshots and persistence. Display copies use the same
reflection and transform actual velocities. Physical distances use the full period.

Historical untagged data remain periodic, retaining original study identities.
Reflected solution archives use schema 3 so older readers reject them; periodic
and field-free archives retain schema 2. Readers check extension, period and
endpoint samples. Undefined error ratios at stationary corners return NaN.

The [permanent guide](../../../docs/dynamics/spatial-extension.md), import,
persistence, JAX and study documentation and their Spanish copies are updated.
Eleven existing quintic notebooks explicitly select periodic.

## Numerical evidence

- Thirteen spline/field tests cover analytic polynomial derivatives, signs,
  shifted and negative coordinates, edges, corners, CPU/JAX and automatic
  Hessians, with float64 absolute tolerances up to 2e-12.
- Closed cosine samples with L=2 and rho=0.37 differ from analytic J0
  multipliers by at most 3.33e-16. Tests also cover repeated averaging and
  unequal opposite-edge values.
- PHI_2.h5 with normalized space/time and rho=0.05 gives a maximum normal-slope
  residual of 1.32e-14 over 127 points per edge at times 0, 0.23 and 0.7.
- A polynomial-field RK4 run starts at (-0.49,1.25), uses times 0.31 to 1.31
  and step 0.002 (500 steps), and crosses x=-0.5. Its final physical point is
  (-0.6596515510,1.1627096374), displayed at (-0.3403484490,1.1627096374).
  The [comparison figure](reflection-check.png) and animation were rendered
  and inspected; saved states remain unchanged.
- The reduced HDF5 automation test uses two cycles and 50 fixed steps per cycle.
  Measured DOP853/Radau disagreement is 9.37e-14, maximum method discrepancy
  1.402e-10, and coordinate-rescaling discrepancy 3.886e-15.
  Requested reference controls are rtol=1e-11, atol=1e-13, max_step=0.005
  for DOP853; rtol=1e-12, atol=1e-14, max_step=0.0025 for Radau.
  Those controls are not measured errors.

## Validation and corrections

The headless full run completed 765 tests in 640.880 seconds, with three GPU
skips, initially reporting 13 failed assertions/subtests and one error.
Corrections addressed a misplaced structural type annotation, periodic
assumptions in historical fixtures, zero-error ratios and obsolete bytecode
directories appearing as packages. No sources were removed; caches were preserved
under /tmp/gc2d-obsolete-cache-rnz1erpq and /tmp/gc2d-retired-package-caches-83rr3nnp.

After corrections all 108 tests in the following affected/new modules pass.
The full 765-test suite was not repeated.

    MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=tests \
      .venv/bin/python -m unittest -v \
      test_adaptive_integration test_contract_layout test_gauss_legendre4_studies \
      test_package_layout test_poincare_probe test_poincare_rho_sweep \
      test_runge_kutta test_states test_solution_persistence \
      test_reflected_potential test_reflected_spline \
      test_spatial_extension_study_contracts test_potential_snapshot test_visualization

A prior standard-input runner could not launch spawned workers; invoking unittest
as a module resolved it. Other passed checks: 17 automation/viewer tests,
19 historical HDF5 tests, reduced examples, mypy across 315 sources,
git diff --check and python -m build (source and wheel). A build without isolation
initially lacked setuptools; the ordinary isolated build succeeded.

The eleven notebook changes insert one configuration line each. JSON, cell
syntax, outputs, IDs and metadata were checked. Construction, gyroaveraging and
derivative evaluation pass with temporary nx=ny=8 and max_wave_number=3 overrides.
No complete notebook was executed and old outputs were retained.

Final SDD validation: 199 records and 22 requirements pass. This validates
document structure, not scientific correctness.

## Limits and scope

Reflected currently requires degree 3; periodic supports degrees 2 through 5.
The gyroaverage is spectral averaging of samples followed by spline reconstruction,
not exact circle integration of the spline. No GPU was available. No long
scientific experiment or new TFM result is claimed.

No TFM file was edited or compiled. Unrelated changes were preserved.
No commits, pushes, cloud publication or personal-configuration changes were made.
Spanish reading copies remain local and ignored by Git.
