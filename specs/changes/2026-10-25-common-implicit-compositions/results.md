# Execution results

On 2026-10-25 the user accepted preparation and execution of the clarified
structure: direct common inheritance for ABBA2/ABBA4/BM4 implicit methods,
ABBA4 as its own operation tuple with one projection, and removal of ABBA6.
The four preparation documents were presented before source edits.

ABBA2Implicit, ABBA4Implicit and BM4Implicit now directly inherit
ProjectedPalindromicMethod. It owns solver validation, preparation and advancement.
ABBA2 and ABBA4 declare their explicit tuples; BM4Implicit reuses BM4Midpoint's
explicit tuple. Fixed recipes have no dataclass field. ABBA4 retains its complete
triple jump and one exterior solve. The two ABBA intermediate bases are removed.

ABBA6's method, exports, event, derivative adapter, dedicated accuracy study and
active consumers are removed without aliases. Current guides and model theory
reflect the common base; retained ABBA6 documents are marked historical. The ABBA
family diagram was regenerated. Seven affected theory PDFs were compiled and
the changed pages inspected. Spanish reading copies were updated. Memory records
the decision within its 300-line limit. No notebook consumers were found.

## Numerical comparison

Before editing, captured the five remaining presets on the deterministic
three-particle fixture from tests/test_extended_family.py: Potential.random with
A=0.08, M=3, nx=ny=16, seed=27, interpolation_order=5 and rho=0.05; packed initial
x values 1.0 through 1.2 and y values 1.2 through 1.4, t=[0.3,0.5], h=0.02,
eleven saved times and EnergyObserver plus step observation. Accepted states
have six physical entries; temporary copies have twelve. Solver defaults were
unchanged. Compared 4,250 named arrays/scalars, including physical states, passive
momentum and nested event fields, excluding callbacks and dynamics objects.

Every numeric maximum absolute difference was **0**, with matching shapes and
string metadata. This is an observed finite-run comparison, not a proof for all
inputs. Convergence, signed maps, derivatives and projection controls remain
covered by the repository tests.

## Checks

- The 39 focused family, runtime, generic-function, BM4 and import tests passed.
- The eight ABBA configuration tests passed after updating the expected count
  from 26 to 18: sixteen implicit combinations and two midpoint variants.
- Source typing passed for 303 files.
- gc_orbit.py, projected_abba.py and palindromic_time.py completed successfully.
- Source and wheel distributions built successfully; neither contains the retired
  ABBA6 runtime module.
- SDD validation passed for 129 records and 22 requirements.
- `MPLBACKEND=Agg .venv/bin/python -m unittest discover -s tests -v` ran all
  712 tests in 778.568 seconds. It reported one failure from the obsolete
  26-variant assertion, already corrected to 18 while that run was in progress.
  The subsequent configuration and package-layout run passed all 21 tests,
  including the eight corrected configuration checks. Combined final coverage
  is **709 passing tests and three GPU skips**, with no unresolved failures.
  JAX CPU, shortened final steps, generic recipes, convergence, derivatives,
  energy, observers and persistence were exercised.

No long notebook, remote study or GPU execution was performed. GPU tests depend
on hardware availability. Historical trajectories and prior change records were
preserved. No commit or push was made.
