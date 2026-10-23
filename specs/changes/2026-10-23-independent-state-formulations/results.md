# Execution results

Accepted on 2026-10-23 through `$sdd ejecuta`: implement the presented
independent representations, method-owned initialization and step preparation,
preserved projection/solver behavior and CPU/JAX verification.

## Implemented behavior

`SingleFormulation` and `DoubleFormulation` retain structural layout, particle
count and tracking only. Their `lift`, `join_copies`, `split_copies` and `pack`
operations support GC, FC and external layouts on NumPy/JAX. FC duplication
includes velocities. The former state class names are removed without aliases.

Methods initialize copies and zero passive momentum, prepare each step's equal
or multiplier-shifted candidates, select projection/solver, and integrate passive
energy from accepted stages. Methods supply evaluated Hamiltonian arrays for
diagnostics. Accepted dimensions, projection equations, solver defaults, energy
normalization and saved labels/schemas are unchanged. Specialized GC/FC maps
reuse the copy operations and retain their public routes.

Current specifications, affected model guides, migration instructions and eight
model diagrams are updated with Spanish Markdown copies. Representative shared
and midpoint diagrams were visually checked. Memory records the confirmed
ownership rules in 298 lines; its Spanish copy has 300. Historical records and
the unchanged mathematical derivations were retained.

## Numerical comparison

The conditions in [steps.md](steps.md) were captured before edits at commit
`7ed3fe0` and repeated afterwards: 100 cases and 1282 arrays, covering built-in
GC/FC method combinations, energy on/off, off-grid samples, reduced/simultaneous
projection and Newton/Broyden on NumPy/SciPy and JAX CPU. Python 3.12.3, NumPy
1.26.4, SciPy 1.17.1 and JAX/jaxlib 0.7.1 were used with `JAX_ENABLE_X64=true`.

Every before/after array is exactly equal: maximum difference **0**, including
physical states, passive energy, accepted grids and recorded nonlinear work.
Recorded solver settings also agree exactly. Absolute/relative tolerances and
iteration limits were 1e-14/1e-13/20 for Gauss and SDIRK;
1e-13/1e-12/12 for HBVM and default implicit ABBA/BM4; and 1e-12/1e-11/40 for explicit
Newton/Broyden projection configurations. DOP853/Radau used absolute 1e-12 and
relative 1e-10 error controls. Explicit methods perform no nonlinear solve.

| NumPy/SciPy versus JAX CPU | Maximum absolute difference |
| --- | ---: |
| Physical trajectories | 5.551115123125783e-16 |
| Passive momentum | 2.6020852139652106e-18 |
| Physical Hamiltonian | 6.938893903907228e-17 |
| Generalized energy and its error | 6.938893903907228e-17 |

These satisfy the planned trajectory rtol=3e-10, atol=3e-11 and energy rtol=3e-9,
atol=3e-11. They measure backend/regression agreement, not physical trajectory
error. Capture/comparison scripts, settings, arrays and logs are in
`/tmp/gc2d-independent-formulations/` for this session.

## Checks

- Seven new tests cover dimensions/sample axes, full FC copies, array ownership,
  invalid input, independent imports, supplied energy, JAX tracing/derivatives,
  consecutive steps, shifted candidates and shadow steps.
- All 54 focused tests passed. An initial invocation had one module import
  error from an existing test-search-path assumption; its eight tests passed
  when rerun with `PYTHONPATH=tests:src`.
- `MPLBACKEND=Agg JAX_ENABLE_X64=true .venv/bin/python -m unittest
  discover -s tests -v`: 661 tests in 475.914 s, 658 passed and three GPU checks
  skipped. Existing convergence, reversal, CPU/JAX and archive tests passed.
- `.venv/bin/python -m mypy src` passed for 305 source files.
- `examples/gc_orbit.py` and `examples/projected_abba.py` passed.
  `.venv/bin/python -m build` produced source and wheel distributions.
- All notebooks, including ignored development notebooks, were audited for
  retired names; none required migration. In-memory runs of cells 2/4/6 of
  `single_particle_orbit.ipynb` and `single_gc_bm4_orbit.ipynb` passed with M=2,
  16x16 fields, interval [0, 0.02], step 0.002 and seven samples. Canonical
  notebooks and saved scientific results were not rewritten.
- The new documentation example passed. Spanish copies are present and ignored.
  Final diff review and `git diff --check` passed.
  `.venv/bin/python scripts/check_sdd.py` passed: 84 records, 22 requirements;
  this validates document structure only.

JAX GPU checks are unavailable on this machine. No long notebook campaign or
new FC integrator was introduced; double FC support is a representation
capability. No accepted implementation work remains pending.
