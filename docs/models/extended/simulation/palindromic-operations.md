# Palindromic methods with explicit time operations

All ABBA and BM4 presets inherit `PalindromicMethod` in `methods.extended.palindromic`. Their numerical execution is shared. A recipe defines the chronological operations; an exterior projection defines how the accepted physical state is recovered. The BM4 coupling removal remains a separate completed change.

## Operations and coefficients

For two planar copies `u` and `v` at clock `t`, with signed `d = coefficient*h`:

| Operation | Exact update | Clock |
|---|---|---|
| `A(coefficient)` | `u <- u + d*f(t,v)` | unchanged |
| `B(coefficient)` | `v <- v + d*f(t,u)` | unchanged |
| `T(coefficient)` | spatial copies unchanged | `t <- t + d` |
| `C(coefficient)` | independent harmonic rotation of the copy difference | unchanged |

Thus `A(0.5)` means A for `h/2`. There is no additional half inside A or B. The factor `1/2` used later for physical passive momentum has a separate purpose: it converts the sum from the doubled Hamiltonian to physical kappa.

States are component-major `(u_x, u_y, v_x, v_y)`, length `4N`. The clock is a scalar in runtime time units. A/B retain their physical source and signed duration for exact Jacobians and accepted energy quadrature. T has an identity spatial tangent. C requires an explicitly supplied `HarmonicCoupling`, preserves the copy mean and rotates the difference by `2*frequency*d`.

## Constructing and using a palindrome

```python
from simulation import A, B, T, Composition, PalindromicImplicit, PalindromicMidpoint

center: Composition = (A(0.5), B(0.5), T(1.0), B(0.5), A(0.5))
edges: Composition = (T(0.5), A(0.5), B(0.5), B(0.5), A(0.5), T(0.5))

implicit = PalindromicImplicit(edges, composition_name="ABBA-edges")
midpoint = PalindromicMidpoint(edges)
```

`Composition` is the public type alias `tuple[Operation, ...]`, not a recipe
object. The tuples are written in execution order. The first evaluates spatial
flows at the step endpoints; the second evaluates them at the midpoint time.
The consecutive translations `T(0.5), T(0.5)` are combined into `T(1.0)`.
Spatial flows remain separate to retain field evaluations and energy quadrature.

Operation construction checks finite coefficients. Method preparation checks a
nonempty tuple of operations, reflected operation/coefficient pairs and unit
weights separately for A, B and T. Signed and zero coefficients are allowed.
The total C weight scales its independent rate. These checks run once per run,
not on every residual evaluation. Symmetry does not prove fourth or sixth order;
arithmetic projection need not preserve reversibility or exact symplecticity.

To migrate, replace `Composition(name, operations)` with the operations tuple and
pass `composition_name=name` to the generic method if a diagnostic label is needed
(the default is `"custom"`). `Composition.from_half` and
`Composition.from_direct_adjoint` are removed. Write the full tuple explicitly;
a half can also be reflected with ordinary tuple concatenation. Recreate pickled
method configurations using the new representation. Saved scientific results
are not rewritten.

A new method can select a recipe without implementing another numerical loop:

```python
from dataclasses import dataclass
from simulation import ABBA2Midpoint

@dataclass
class EndTimeMidpoint(ABBA2Midpoint):
    composition_name = "ABBA-edges"
    recipe = (T(0.5), A(0.5), B(0.5), B(0.5), A(0.5), T(0.5))
```

NumPy supports this subclass. JAX supports the exact built-in generic configurations `PalindromicImplicit(recipe)` and `PalindromicMidpoint(recipe)` and the named ABBA/BM4 presets. Arbitrary subclass overrides are still rejected by the existing JAX dispatch contract.

For explicit C use:

```python
from simulation import C
from methods.extended.maps import HarmonicCoupling

coupled: Composition = (
    T(0.5), C(0.5), A(0.5), B(0.5),
    B(0.5), A(0.5), C(0.5), T(0.5),
)
method = PalindromicMidpoint(coupled, coupling=HarmonicCoupling(0.7))
```

Both C in the recipe and the coupling object are required. BM4 presets have neither and retain no coupling constructor control. Generic methods do not infer an order from the recipe name.

## Presets, projection and observations

The standard ABBA2, ABBA4 and BM4 recipes are tuple literals declared
directly as `recipe = (...)` on their method classes in `extended/abba.py` and
`extended/bm4.py`. They are shared class attributes, with no dataclass field.
ABBA2Midpoint reuses ABBA2Implicit.recipe; BM4Implicit reuses BM4Midpoint.recipe.
The generic core validates and executes the tuples. Consecutive T operations are merged by adding their
weights, giving 5, 15 and 30 elementary operations respectively. The same
spatial operations remain, with no automatic fusion or implicit time placement.
Only T changes time. Merging translations preserves the mathematical map;
floating-point clock rounding can differ from two separate additions.

`ProjectedPalindromicMethod` owns the common implicit advance and one exterior spatial Hairer solve. `MidpointPalindromicMethod` owns the final arithmetic mean. Both inherit common run preparation and quadrature description from `PalindromicMethod`. Preset-specific code retains public controls, metadata and established observation adapters; no preset defines its own advancement loop.

The implicit generic method accepts reduced or simultaneous equations, Newton/Broyden, and analytic or centered-difference Jacobians. Time and passive momentum remain outside the nonlinear unknowns. The stopping scale remains `atol + rtol*max(1, ||z||_inf)`.

Elementary operation records drive NumPy/JAX tangents and energy. The numerical
trace contains no grouped stages or grouping metadata. Observation adapters
assemble ABBA pairs from four retained shears and BM4 stages from two, without
replaying the map. Public event counts and shapes remain unchanged: BM4 has
12 observed groups, 30 elementary operations and 24 spatial shears.

Recipe names belong to the method configuration. Metadata distinguishes
`composition_name`, `operation_count`, `operation_sequence`,
`operation_coefficients` and the independent coupling rate. Coefficients are
stored as a JSON string so configuration metadata cannot be confused with arrays
of per-step history.

ABBA2Implicit, ABBA4Implicit and BM4Implicit directly inherit `ProjectedPalindromicMethod`, which also owns their preparation and solver validation. ABBA4 declares its complete tuple, with one exterior projection; it does not invoke three projected ABBA2 steps. ABBA6 and the intermediate ABBA bases have been removed.

## Verification and example

`tests/test_palindromic_operations.py` covers tuple validation, independent
end-time shears, signed reversal, C tangents, shared inheritance, recipe-only
subclasses, center/end convergence, passive energy and CPU/JAX agreement.
`tests/test_extended_family.py` compares merged and unmerged clock schedules
for both step signs, including states, derivatives and energy sources. Existing
ABBA/BM4 tests retain their orders, observers, particle packing, solvers and
output sampling. Measured pre-change comparisons are recorded in the
[tuple simplification results](../../../../specs/changes/2026-10-25-tuple-compositions/results.md).

Run `python examples/palindromic_time.py` for a small comparison with the same potential, initial state and step. Changing the placement of T creates a different finite-step method; it does not alter the standard presets.
