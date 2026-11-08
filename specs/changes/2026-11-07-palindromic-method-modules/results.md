# Results

On 2026-11-07 the user accepted the presented proposal and authorized its
execution: rename the extended family to palindromic and give each method its
own module, preserving numerical behavior and the existing Runge–Kutta work.

## Implemented

`methods.palindromic` contains seven concrete method modules and three shared
base modules, as proposed. Contracts and diagnostics use the same family name;
active consumers import the new defining modules. Public method names and
exports from `simulation` remain unchanged. Retired packages have no aliases.
The ten class definitions and all ABBA4/BM4 coefficient declarations retain
their original contents apart from their module location and imports.

The shared model documents now live in `docs/models/palindromic/`. Current
specifications, repository guidance, migration guides, seven model diagrams, nine theory PDFs, and
Spanish reading copies were updated. Historical diagrams and completed change
records retain their original evidence. Four local notebooks received import
or documentation-link updates and remain ignored by Git. `MEMORY.md` records
the canonical module ownership and remains at 299 lines.

## Verification

The pre-move baseline and migrated implementation were compared on 17 NumPy
cases covering all seven concrete methods, Newton/Broyden, and both projection
formulations where supported. The generic methods included explicit C with
`HarmonicCoupling(0.4)`. Inputs were `Potential.random(A=.08, M=3, nx=16, ny=16,
seed=27, interpolation_order=5)`, GC `rho=.05`, three initial particles at
`x=(1,1.1,1.2)`, `y=(1.2,1.3,1.4)`, interval `(0.3,0.415)`, step `0.03`, and
saved times `(0.3,0.307,0.32,0.349,0.36,0.399,0.415)`. The last accepted step
was `0.025` in runtime time units. Implicit solver tolerances were absolute
`1e-12`, relative `1e-11`, with at most 40 corrections and analytic Jacobians.
These are solver controls, not claimed trajectory-error bounds.

All 3,734 numeric arrays and 600 other fields matched exactly, including
states, energy, multipliers, stage observations, and nonlinear work: maximum
measured discrepancy **0**. Configured methods also passed pickle round trips.
The capture/compare script and baseline are retained temporarily at
`/tmp/gc2d_palindromic_baseline.py` and `/tmp/gc2d_palindromic_baseline.npz`;
run its `save` mode on the old layout and `compare` mode on the new layout.

- The eight focused test modules passed: **55 tests**, including generic
  CPU/JAX agreement, import ownership, serialization, and local worker checks.
  Stale imports and mock targets found during migration were corrected.
- `python -m mypy src`: passed for **317 source files**.
- Both headless smoke examples passed: `examples/gc_orbit.py` and
  `examples/projected_abba.py`.
- Source and wheel distributions built successfully from a clean temporary
  copy of the current source. Wheel contents contain the three palindromic
  packages and all ten defining modules, with no retired extended/classical
  packages. No dependency installation into the project was needed.
- Nine PDFs compiled with two LaTeX passes; rendered pages and the updated
  family diagram were visually inspected. Seventy updated family links resolve.
- `MPLBACKEND=Agg python -m unittest discover -s tests -v`: **756 tests**
  completed in **788.384 seconds**, with **no failures and 3 GPU skips** because
  no JAX GPU is available on this host.
- The final SDD document check passes: **184 records, 22 requirements**.

## Notebook limitation

`bm4_simple_inspection.ipynb` passed a three-sample run on `(0,0.02)` with
step `0.01`. The HDF5 inspection notebook's changed imports also work, but its
existing `coupling_frequency=pi/8` constructor option predates this change and
is unsupported by the already uncoupled BM4 implementation. A disposable
in-memory adaptation omitting that argument passed the same reduced run.
The canonical scientific setting and saved outputs were preserved; this does
not claim that its unmodified calculation cell runs. The BM4/RK4 notebook pair
changed only Markdown links, whose targets were checked without running its
studies. No notebook result was saved or uploaded.

No remote jobs, commits, pushes, or personal-configuration changes were made.
