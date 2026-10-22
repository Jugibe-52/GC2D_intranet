+++
schema_version = 2
kind = "change-plan"
id = "2026-10-18-hdf5-independent-rescaling"
revision = 3
spec_revision = 2
sources = ["src/potential/potential.py", "src/potential/load.py", "src/potential/_evaluation.py", "src/potential/jax_potential.py", "src/dynamics/gc.py", "src/dynamics/fc.py", "src/studies/dimensional_h5_midpoint.py", "src/studies/poincare_rho_sweep.py", "tests/test_gc2d_h5_potential.py", "tests/test_potential_snapshot.py", "src/potential/grid.py", "src/studies/h5_provenance.py", "src/studies/_potential_snapshot.py", "src/diagnostics/io/persistence.py", "src/studies/poincare_radial_cycle_time.py", "src/studies/poincare_probe.py", "src/studies/reference_trajectory.py", "src/visualization/poincare_method_comparison.py", "tests/test_h5_study_selection.py", "tests/test_solution_persistence.py", "tests/test_jax_potential.py", "tests/test_poincare_rho_sweep.py"]
documents = ["specs/changes/2026-10-18-hdf5-independent-rescaling/spec.md", "specs/current/potential-import.md", "docs/dynamics/gc2d-h5-import.md"]
open_questions = []
+++

# Independent HDF5 rescaling: implementation plan

## Design

This plan reviews specification revision 2 and the accepted radial decision.
Retain one Potential representation. Validate input, read original arrays,
select modes, filter optionally and apply explicit length, time and amplitude
scales in load.py. Build the runtime splines from the resulting samples. Keep
no-transpose orientation and square-grid restrictions. Expose Boolean flags;
remove magnetic/characteristic scales and resampling from the public loader.

Use angular rates directly in shared NumPy/JAX reconstruction and derivatives.
Random omega=1 retains its phase. Version persisted rates and decode known old
cyclic values only at read boundaries. Preserve arrays, class and provenance.
GC dynamics applies B once to its effective gyroaveraged Hamiltonian; all
physical equations and adapters consume that same Hamiltonian.

Dimensional study composition starts from source samples and constructs its
cycle-time stream with T and B. The radial viewer retains its independent
1/(2*pi) factor. Historical length/custom-period/B conventions are reconstructed
explicitly in studies/historical_h5.py using the new reader, not a second HDF5
parser or a compatibility import tree. Keep historical floating-point arithmetic
for hashes. Optional historical sampling uses original source axes and its own
separate spline fit, retaining the old interpolation boundary and roundoff.
This exception exists only to reproduce archived numerical fields.

## Compatibility

Migrate active Python and notebook loaders, inherited classes, aliases, forwarded
keyword dictionaries, constructors and phase displays. Keep historical rho and
source-index recovery. Frozen bundles with their own verified old source keep
that evaluator's convention. Backup/conflict and superseded study artifacts are
not live consumers; document exact exclusions. Saved production outputs remain
untouched. Source-based historical reconstructions retain custom periods and
fingerprints; reject unknown archive conventions rather than guessing.

Update current POT-001 through POT-005, public and BM4 model docs, and Spanish
reading copies. Record file/cell migrations and checks in tasks.md. Audit FC
consumers separately: their equations and random-field phases are unchanged;
GC variable scaling is not applied to FC parameters.

## Verification plan

Use asymmetric square fixtures with nonzero origins and angular frequencies
3 and 3*sqrt(2), mean-only selections and invalid rectangular samples. Verify
raw samples exactly, phase/coordinates at 1e-12, amplitude-aware derivative maps
and float64 NumPy/JAX at 1e-10. Test all four flags, first/second/mixed derivatives,
filtering, subclass preservation, gyro-radius maps, B and persistence boundaries.

Compare periodic shear to an analytical time integral and smooth 2D trajectories
with three particles over two reference cycles, rho=0 and nonzero rho. Use
DOP853 rtol/atol=1e-11/1e-13, max step T/100, and independent Radau
1e-12/1e-14, max step T/200. Require position discrepancy/L <=1e-8 and reference
floor/L <=1e-9; report observed errors, not just requested tolerances.

Run historical regressions and compare the actual PHI_2 field and fingerprint
with the pre-change implementation, including explicit old sampling. Validate
notebook load/composition expressions in memory with a small source, two cycles,
three particles, two rho values and eleven frames. Check the measured viewer
preparation/background path separately. Never overwrite outputs or launch remote
jobs. Finish strict mypy, headless full suite, smoke examples, build and SDD check.

## Outcome and limitations

The approved scope is implemented and evidence is maintained in [tasks.md](tasks.md).
Reduced checks do not certify long-time trajectory accuracy or GPU/cloud runs.
