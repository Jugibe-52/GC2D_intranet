# Proposed behavior

Composition becomes a public type alias for tuple[Operation, ...]. Define the
ABBA2, ABBA4, ABBA6 and BM4 recipes explicitly, retaining A/B/T/C and their signed
coefficients in full-step units. Declare each fixed recipe directly on its method
class as a tuple literal, with no dataclass field; variants share the same tuple.
ABBA2 becomes
`(A(.5), B(.5), T(1.), B(.5), A(.5))`.

Validate the tuple's operation types, palindrome and separate unit totals for
A/B/T when preparing a method. Keep names on method configurations, with an
optional composition_name for generic recipes. Remove the old Composition
constructor and its from_half/from_direct_adjoint factories; document migration.

Merge consecutive T operations in the presets by adding their coefficients.
Keep spatial operations separate, preserving field evaluations and passive
energy quadrature. Floating-point clock rounding may differ; compare with the
previous implementation rather than require bitwise identity.

Keep only elementary operations in the numerical trace. Existing ABBA and BM4
observation adapters reconstruct their public groups from retained spatial
states, without repeating integration. Preserve event counts and shapes,
projection placement, signed steps, dense/particle derivatives and NumPy/JAX
support. The elementary operation counts become 5, 15, 35 and 30 respectively.

The physical layout remains two temporary planar copies (4N components), one
scalar clock, and one accepted physical state (2N). Energy remains observer-owned.
See [the operation guide](../../../docs/models/extended/simulation/palindromic-operations.md).
