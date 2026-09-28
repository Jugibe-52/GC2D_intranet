# Optional JAX potential evaluation

`Potential.evaluate`, `electric_field`, and `evaluate_grid` accept a keyword-only
`execution` argument. Import its immutable configuration from
`contracts.execution.Execution`. Omitting it always selects SciPy on CPU and
returns NumPy arrays. An explicit JAX choice returns native JAX arrays.

```python
from contracts.execution import Execution

scipy_cpu = Execution()
jax_cpu = Execution(backend="jax", device="cpu")
jax_gpu = Execution(backend="jax", device="gpu", device_index=0)
```

`Execution` validates backend/device combinations without importing JAX or
initializing hardware. SciPy supports CPU index 0 only. Device availability is
checked when the selected evaluator is first used. This contract currently
configures potential evaluation; `simulation.runner.simulate(problem, method,
request)` remains the execution facade for the unchanged NumPy integrators and
does not yet accept `execution`. CPU process counts are not part of this phase.

## Installation and use

JAX is optional and is imported only when constructing the evaluator. For the
tested CPU environment:

```bash
python -m pip install -c constraints.txt -e '.[jax]'
```

GPU use additionally requires a compatible device, driver, and JAX accelerator
installation. Follow the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
for that machine; the CPU extra alone does not configure GPU support.

```python
import jax
import numpy as np
from contracts.execution import Execution
from potential import load_gc2d_h5_potential

# Configure precision once, before preparing or compiling calculations.
jax.config.update("jax_enable_x64", True)
potential = load_gc2d_h5_potential("data/potential/V1/PHI_2.h5")
effective = potential.gyroaverage(0.3)
execution = Execution(backend="jax", device="cpu")
# For a configured accelerator: device="gpu", device_index=0.

x = np.array([0.1, 0.2, 0.3])
y = np.array([0.4, 0.5, 0.6])
phi = effective.evaluate(0.25, x, y, execution=execution)
ex, ey = effective.electric_field(0.25, x, y, execution=execution)
phi_xy = effective.evaluate(0.25, x, y, dx=1, dy=1, execution=execution)
phi_t = effective.evaluate(0.25, x, y, dt=1, execution=execution)

# Copy only when host-side analysis or persistence actually needs the values.
host_phi = np.asarray(phi)
```

The JAX evaluator does not alter the global precision flag. It rejects disabled
float64 explicitly, rather than silently truncating arrays to float32. Requests
for unavailable devices also fail explicitly, without a CPU fallback.

`evaluate` requires matching `x` and `y` shapes; time broadcasts against their
common shape. `evaluate_grid` returns `(nx, ny, *time.shape)` from the stored
samples. `electric_field` returns `(-phi_x, -phi_y)` and evaluates the full grid
when both coordinates are omitted. All outputs remain on the selected device.
NumPy inputs are accepted, but repeatedly supplying host arrays on a GPU incurs
transfers. Reuse device arrays for repeated calls.

## Shared core and lifetime

`Potential` owns one immutable `PreparedPotential` record, exposed as
`potential.prepared`. It holds the validated samples, grid, frequencies, and
canonical SciPy spline preparation. Read-only spline knots and coefficients
are exported lazily through `prepared.spline_data`.

Two concrete evaluators consume that same record:

- `potential.scipy_evaluator.ScipyPotentialEvaluator`: calls the canonical
  SciPy spline routines and returns NumPy arrays.
- `potential.jax_evaluator.JaxPotentialEvaluator`: copies the exported
  coefficients to the selected device and runs the JIT-compiled spline kernel.

Both implement the shared `PotentialEvaluator` base in `_evaluation.py`, which
owns shape and derivative validation, grid field evaluation, and the electric
field definition. Periodic wrapping and harmonic reconstruction are shared
functions, used by the NumPy code and inside JAX compilation. Spatial spline
evaluation and device placement remain backend-specific.

Normal callers use `Potential` methods directly. The explicit evaluator classes
remain available for advanced use and accept either a `Potential` or its
`PreparedPotential` record. Their package exports are explicit; no import
aliases or retired compatibility modules are introduced.

Evaluators are cached per potential and equal `Execution` values. Reusing the
configuration across value, grid, and electric-field calls reuses the device
buffers. A JAX call never changes the default of a subsequent call without
`execution`. Initialization inside an outer `jax.jit` materializes constant
buffers eagerly, so the persistent cache cannot retain transient tracers.

Public physical attributes (`grid`, `mean`, `modes`, `frequencies`, and
`interpolation_order`) are now read-only properties backed by the prepared
record. Construct a new potential when changing physical data; gyroaveraging
already returns its own potential for nonzero radii. This prevents cached
evaluators from silently using obsolete arrays. Pickling a `Potential` stores
physical inputs and metadata only: receiving processes rebuild CPU splines and
start with an empty evaluator cache. JAX modules and device buffers are excluded.

## Mathematical equivalence

Spline construction, HDF5 preprocessing, and gyroaveraging remain on CPU. Both
paths use the **same actual knots and coefficients** fitted by SciPy's
`RectBivariateSpline`; JAX does not fit an alternative cubic interpolator.
Its differentiated Cox--de Boor recurrence supplies spatial derivatives
directly, with no finite differences. Time dependence is shared:

```text
Phi(t, x, y) = Phi_0(x, y) + 2 Re sum_j[C_j(x, y) exp(i 2*pi*f_j*t)].
```

Spline degrees 2 through 5 and derivative restrictions are unchanged:
`0 <= dx, dy < interpolation_order` and `dt` in `(0, 1, 2)`. Temporal derivatives
multiply each mode by `(i*2*pi*f_j)**dt`; the mean contributes only for `dt=0`.
The shared reconstruction retains sequential addition in mode order.
Derivative orders are static integers under JIT. Calls can be enclosed in
`jax.jit`, and spatial automatic differentiation is supported away from knots
and periodic seams. CPU and GPU need not be bitwise identical.

## Scope and performance

The current integrators continue calling `Potential` without an execution
choice and therefore use SciPy/CPU. Selecting JAX for a standalone potential
call does not enable JAX for the complete simulation. Numerical-method APIs,
the `simulation` package name, and the `Solution` contract are unchanged.

The first call prepares the selected evaluator and compiles for the input
shapes and derivative orders. Measure subsequent calls separately and
synchronize them with `block_until_ready()`. Conversions to NumPy are explicit;
transferring data at every CPU integration stage can outweigh faster GPU work.

The example compares SciPy, resident JAX inputs/output, and the complete NumPy
input/output boundary through the same public methods for 256 particles:

```bash
python examples/jax_potential.py --device cpu --particles 256
python examples/jax_potential.py --device gpu --particles 256
python examples/jax_potential.py --device cpu --particles 256 --h5 data/potential/V1/PHI_2.h5
```

Without `--h5`, the example uses a clearly identified synthetic field. A Git LFS
pointer is not a usable HDF5 input. These evaluator timings do not predict
complete BM4/ABBA integration performance.

## Validation

`tests/test_jax_potential.py` compares values, gradients, Hessians, mixed time
derivatives, all supported spline degrees, shifted periodic boundaries,
scalar/broadcast/grid shapes, mean-only and zero fields, gyroaveraging, a
synthetic HDF5 fixture, JIT composition, and automatic spatial derivatives.
GPU equivalence runs when hardware is available and otherwise reports a skip.

`tests/test_potential_execution.py` checks the common public API, an independent
harmonic reconstruction, immutable configuration and physical data, evaluator
reuse, unchanged defaults, first use inside JIT, unavailable hardware, float64
enforcement, and serialization without device resources. A separate CI job
installs the optional JAX dependency and runs both suites on CPU.

```bash
python -m unittest discover -s tests -p 'test_jax_potential.py' -q
python -m unittest discover -s tests -p 'test_potential_execution.py' -q
```
