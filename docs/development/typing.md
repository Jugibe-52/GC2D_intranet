# Visible types at result assignments

Throughout `src/`, show the type of a function result where it first receives a
variable name. Keep the function's return annotation too: the definition describes
the contract, while the receiving variable makes its caller readable on its own.

For one receiving name, put the annotation on the assignment:

```python
saved: States = States.prepare(times, initial, layout=layout)
```

For unpacking, declare the receiving names immediately before the assignment:

```python
physical: np.ndarray
auxiliary: dict[str, np.ndarray]
physical, auxiliary = formulation.extract_sample(time, sample)
```

Declare each name only once in a function or module. Reassignments, already typed
parameters and names declared earlier retain that type. If the first assignment
is `None`, annotate it with the supported optional type there. Branches share a
function scope; they do not need duplicate declarations. The discarded `_` target
does not need an annotation. Class attributes keep their existing field contracts;
adding a dataclass field merely to annotate a call would change construction.

Use the existing result classes and aliases instead of spelling out internal
library implementation types. Keep imports needed only for annotations under
`TYPE_CHECKING` when they would introduce optional dependencies or cycles. Quote
module-level annotations that refer to those names unless the module already uses
postponed annotations. Local variable annotations are not evaluated at runtime.

Use `np.ndarray` for host arrays, `jax.Array` for device arrays, and
`contracts.arrays.BackendArray` for numerical operations that preserve either
backend. These names do not promise a shape, physical unit or dtype. Keep those
contracts in the nearby documentation and runtime validation. Use concrete types
for devices, figures, paths and project records whenever the operation establishes
them. Truly dynamic metadata, serialization and library callbacks may still need
`Any`; a local annotation does not validate external input.

The source sweep covers assignments receiving calls and unpacked results, including
earlier initializations of reused names. It does not require annotations on every
literal, arithmetic expression, loop variable or comprehension. Files under
`tests/`, `examples/`, `scripts/` and `notebooks/` were not mechanically rewritten.

Check changes with `python -m mypy src`. For broad annotation work, also compare
executable statements before and after, exercise imports without optional backends,
and run the relevant existing tests. Type checking alone does not establish
numerical correctness or ensure that an annotation accurately describes a value
whose upstream type is `Any`.
