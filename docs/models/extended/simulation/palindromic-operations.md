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

center = Composition.from_half("ABBA-center", (A(0.5), B(0.5), T(0.5)))
edges = Composition.from_half("ABBA-edges", (T(0.5), A(0.5), B(0.5)))

implicit = PalindromicImplicit(edges)
midpoint = PalindromicMidpoint(edges)
```

The constructor reflects the supplied half. The chronological sequences are `A/2 B/2 T/2 T/2 B/2 A/2` and `T/2 A/2 B/2 B/2 A/2 T/2`. In the first, spatial evaluations use the step endpoints; in the second, they all use the midpoint time. Adjacent equal flows are deliberately retained to preserve evaluation and observation conventions. An optional `center=` operation allows an odd-length palindrome; the complete sequence can also be passed to `Composition(name, operations)`.

Validation checks finite coefficients, the reflected operation/coefficient pairs and unit total weights separately for A, B and T. Signed and zero coefficients are allowed. The total C weight scales its independent rate. Symmetry does not prove fourth or sixth order. The tests establish second-order convergence for both time placements on a nonautonomous rotation; the two finite-step maps are different. Arithmetic projection need not preserve reversibility or exact symplecticity.

A new method can select a recipe without implementing another numerical loop:

```python
from dataclasses import dataclass, field

@dataclass
class EndTimeMidpoint(PalindromicMidpoint):
    recipe: Composition = field(default=edges, init=False)
```

NumPy supports this subclass. JAX supports the exact built-in generic configurations `PalindromicImplicit(recipe)` and `PalindromicMidpoint(recipe)` and the named ABBA/BM4 presets. Arbitrary subclass overrides are still rejected by the existing JAX dispatch contract.

For explicit C use:

```python
from simulation import C
from methods.extended.maps import HarmonicCoupling

coupled = Composition.from_half("coupled-edges", (T(0.5), C(0.5), A(0.5), B(0.5)))
method = PalindromicMidpoint(coupled, coupling=HarmonicCoupling(0.7))
```

Both C in the recipe and the coupling object are required. BM4 presets have neither and retain no coupling constructor control. Generic methods do not infer an order from the recipe name.

## Presets, projection and observations

The standard ABBA2, ABBA4, ABBA6 and BM4 recipes are unchanged numerically. `Composition.from_direct_adjoint` expands each former adjoint stage into A,B,T and each direct stage into T,B,A with the original signed stage weight. Explicit coupled generic recipes can additionally expand C,A,B,T and its reverse. Execution never decides operations from a stage's parity; it traverses the expanded sequence. Only T changes time.

`ProjectedPalindromicMethod` owns the common implicit advance and one exterior spatial Hairer solve. `MidpointPalindromicMethod` owns the final arithmetic mean. Both inherit common run preparation and quadrature description from `PalindromicMethod`. Preset-specific code retains public controls, metadata and established observation adapters; no preset defines its own advancement loop.

The implicit generic method accepts reduced or simultaneous equations, Newton/Broyden, and analytic or centered-difference Jacobians. Time and passive momentum remain outside the nonlinear unknowns. The stopping scale remains `atol + rtol*max(1, ||z||_inf)`.

Elementary operation records drive NumPy/JAX tangents and energy. Existing grouped ABBA pair and BM4 stage records are reconstructed from this trace without replaying the map. Their public counts and shapes remain unchanged: BM4 has 12 historical groups and 36 elementary operations, of which 24 are spatial shears. New metadata distinguishes `operation_count`, `operation_sequence`, `operation_coefficients` and the independent coupling rate. Coefficients are stored as a JSON string so configuration metadata cannot be confused with arrays of per-step history.

## Verification and example

`tests/test_palindromic_operations.py` covers half construction, consistency, independent end-time shears, signed reversal, C tangents, shared inheritance, recipe-only subclasses, center/end convergence, passive energy and CPU/JAX agreement. Existing ABBA/BM4 tests retain their orders, observers, particle packing, solvers and output sampling. The six pre-change trajectory and momentum baselines matched exactly in the tested three-particle nonautonomous case.

Run `python examples/palindromic_time.py` for a small comparison with the same potential, initial state and step. Changing the placement of T creates a different finite-step method; it does not alter the standard presets.
