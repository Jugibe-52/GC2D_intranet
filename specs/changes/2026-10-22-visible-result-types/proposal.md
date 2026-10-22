# Proposed change

Apply the agreed convention throughout `src/`: annotate a single receiving
variable on its assignment; declare unpacked variables immediately before their
first assignment. Keep return annotations on function definitions as well.

```python
device: Device = resolve_device(options)

prepared: HamiltonianSystem
device: Device
prepared, device = prepare_environment(dynamics, method, options)
```

These are independent examples. Declare each name only once per scope; parameters
and already annotated variables retain their declarations. When an earlier value
such as `None` initializes the same variable, annotate that initialization with
the full supported type. Do not add dataclass fields or change attribute ownership.

Use existing project contracts and readable types compatible with current values.
Preserve NumPy/JAX distinctions and existing array precision policies; this change
does not introduce shape typing or replace numerical arrays with wrapper classes.
Keep genuinely dynamic library boundaries explicit rather than inventing guarantees.
Replace the known JAX device `Any` with `jax.Device` along its preparation path,
without importing optional dependencies on the normal CPU path.

The accepted scope is all Python source under `src/`, including batch runtime
helpers. The `tests/`, `examples/`, `scripts/` and `notebooks/` directories are not
targets for the annotation sweep. Record the convention in permanent
documentation and repository memory. No public renames or numerical changes are
part of this work.
