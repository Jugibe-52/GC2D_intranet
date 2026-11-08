# Proposed change

Rename `src/methods/extended/` to `src/methods/palindromic/`. Give every concrete
method its own defining module:

| Module inside `methods/palindromic/` | Method |
| --- | --- |
| `abba2_implicit.py` | `ABBA2Implicit` |
| `abba4_implicit.py` | `ABBA4Implicit` |
| `abba2_midpoint.py` | `ABBA2Midpoint` |
| `bm4_implicit.py` | `BM4Implicit` |
| `bm4_midpoint.py` | `BM4Midpoint` |
| `implicit.py` | `PalindromicImplicit` |
| `midpoint.py` | `PalindromicMidpoint` |

Place the three shared bases in `base.py` (`PalindromicMethod`),
`projected_base.py` (`ProjectedPalindromicMethod`), and `midpoint_base.py`
(`MidpointPalindromicMethod`). Preserve their current inheritance: implicit
methods derive from the projected base, midpoint methods from the midpoint
base, and both bases derive from `PalindromicMethod`. Keep operations, maps,
configuration, observations, and algebraic helpers shared rather than copying
them into each method.

Keep recipes as class tuples with their current coefficients and operation
order. `ABBA2Midpoint` continues to reuse `ABBA2Implicit.recipe`, and
`BM4Implicit` continues to reuse `BM4Midpoint.recipe` through direct imports
from their defining modules. The ABBA4 coefficient array moves with
`ABBA4Implicit`, and its consumers import it there.

Preserve the existing class names and their exports from `methods` and
`simulation`. For example, `from simulation import ABBA4Implicit` remains
valid; its defining import becomes
`from methods.palindromic.abba4_implicit import ABBA4Implicit`. Retire
`methods.extended` and the combined method modules without compatibility
aliases. Apply the family rename to `contracts.methods.extended`,
`diagnostics.methods.extended`, and `docs/models/extended` as well. Contract
records and diagnostic helpers may continue sharing modules; they are not
integration methods.

Preserve the mathematics and controls: physical states have shape `(2N,)`,
temporary doubled states `(4N,)`, and existing coordinates and units remain
unchanged. Keep signed time stages, final arithmetic averaging for midpoint
methods, one outer projection for implicit compositions including ABBA4,
solver tolerances, energy observations, and NumPy/JAX behavior. The family
rename does not rename physical concepts such as extended phase space,
`extended_momentum`, or the doubled formulation.
