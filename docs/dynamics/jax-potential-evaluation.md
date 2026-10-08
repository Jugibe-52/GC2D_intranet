# NumPy and optional JAX potential evaluation

One `Potential` owns the physical samples and fitted splines. Its pointwise
`evaluate(t, x, y)` method selects the calculation from its arguments:

| Arguments | Calculation and result |
|---|---|
| NumPy arrays, lists or Python scalars | SciPy and a NumPy array |
| Any JAX array or tracer among `t`, `x`, `y` | JAX and a JAX array |
| Mixed NumPy and JAX arguments | JAX, converting the remaining inputs |

Coordinate-based `electric_field` follows the same rule. `evaluate_grid` and
`electric_field` with both coordinates omitted always use NumPy/SciPy, and reject
JAX time inputs. For host-side grid analysis, convert time explicitly with
`np.asarray(time)` outside compiled code. No potential method takes execution
options, and the last call does not change subsequent backend selection.

Simulation backend and device choices remain in
`contracts.execution_options.ExecutionOptions`. Fixed methods run their stages
and time loop on the selected device; DOP853/Radau use JAX fields with a SciPy
adaptive controller. See the [execution guide](../simulation/jax-execution.md).

## Installation and use

JAX is optional and the NumPy path does not import it. For the tested CPU
environment:

```bash
python -m pip install -c constraints.txt -e '.[jax]'
```

GPU use additionally requires a compatible device, driver, and JAX accelerator
installation. Follow the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
for that machine; the CPU extra alone does not configure GPU support.

```python
import jax
import numpy as np
from potential import Potential

# Configure precision before preparing or compiling calculations.
jax.config.update("jax_enable_x64", True)
potential = Potential.random(A=0.7, M=8, nx=64, ny=64, seed=27).gyroaverage(0.3)
x = np.array([0.1, 0.2, 0.3])
y = np.array([0.4, 0.5, 0.6])

host_phi = potential.evaluate(0.25, x, y)
device = jax.devices("cpu")[0]  # Use "gpu" on a configured accelerator.
xd, yd = jax.device_put((x, y), device)
phi = potential.evaluate(0.25, xd, yd)
ex, ey = potential.electric_field(0.25, xd, yd)
phi_xy = potential.evaluate(0.25, xd, yd, dx=1, dy=1)
phi_t = potential.evaluate(0.25, xd, yd, dt=1)

# Transfer results only when host-side analysis or persistence needs them.
host_phi_from_jax = np.asarray(phi)
grid_phi = potential.evaluate_grid(0.25)  # Always a NumPy array.
```

`Potential` does not alter the global precision flag. JAX calculations reject
disabled float64 explicitly rather than silently truncating to float32. Select
a device by placing the input arrays there; inputs explicitly committed to
incompatible devices are rejected. A request for unavailable hardware fails
without a CPU fallback.

`evaluate` requires matching `x` and `y` shapes; time broadcasts against their
common shape. `evaluate_grid` returns `(nx, ny, *time.shape)` from the stored
samples. `electric_field` returns `(-phi_x, -phi_y)`. Reuse device arrays in
repeated JAX calls to avoid recurring host-to-device transfers.

## Preparation and lifetime

`Potential` directly owns validated samples, grid, frequencies and the canonical
SciPy spline preparation. Public physical attributes (`grid`, `mean`, `modes`,
`frequencies`, and `interpolation_order`) remain read-only. Construct a new
potential to change physical data; `gyroaverage(0)` returns the same object and
a nonzero radius produces its own potential.

Private functions implement backend-specific spline evaluation. Validation,
periodic wrapping and harmonic reconstruction retain one shared contract.
Preparing JAX coefficients exports the existing SciPy knots and coefficients;
it never refits the splines or transfers the full grid samples for evaluation.
Concrete device coefficients are prepared lazily and reused. Under `jax.jit`,
coefficients become constants placed by the compiled computation; temporary
tracers are never stored in persistent caches, including on first use.

Pickling stores physical inputs and metadata only. Receiving processes rebuild
CPU splines and start with empty device caches. JAX modules, compiled functions
and device buffers are excluded.

`PreparedPotential`, `PotentialEvaluator`, `ScipyPotentialEvaluator`, and
`JaxPotentialEvaluator`, including their former modules and exports, have been
removed. Replace `potential.prepared` access with the corresponding read-only
properties on `potential`. Replace JAX evaluator construction and calls with
`jax.device_put` of the inputs followed by `potential.evaluate` or coordinate-based
`potential.electric_field`. Former `execution=` keywords on potential methods
remain unsupported. There is no separate `JaxPotential` class.

## Mathematical equivalence

Spline construction, HDF5 preprocessing, and gyroaveraging remain on CPU. Both
paths use the same actual knots and coefficients fitted by SciPy's
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
Derivative orders are static integers under JIT. Pointwise calls support
`jax.jit`, `jax.grad` and `jax.vmap`; derivatives are interpreted away from knots
and periodic seams. A traced time argument selects JAX even when both coordinates
are NumPy arrays. CPU and GPU need not be bitwise identical.

## Shared dynamics and simulation lifecycle

`GuidingCenterDynamics` and `FullCyclotronDynamics` support NumPy and JAX using
the same methods and physical equations. State layouts preserve the backend
while separating or packing component-major states. Vector fields,
Hamiltonians, passive momentum derivatives and analytic particle Jacobians do
not force JAX states back to NumPy.

Integration preparation selects the device and compiles these shared methods.
It reuses compiled function identities through bounded caches; a parallel
`JaxDynamics` physical implementation is no longer needed. The compiled driver
still rejects arbitrary custom dynamics and subclass overrides explicitly.
DOP853/Radau retain an explicit host/device adapter for their SciPy controller.

A standalone JAX evaluation does not change simulation defaults. Pass
`options=ExecutionOptions(backend="jax", device="cpu")` to `simulate` to select
JAX execution. Every route exports the usual NumPy `Solution`; see the
[execution guide](../simulation/jax-execution.md).

## Timing and validation

The first JAX call includes coefficient preparation and compilation for its
input shapes and derivative orders. Measure later calls separately and
synchronize them with `block_until_ready()`. The example compares SciPy,
resident JAX inputs/output, and explicit input/output transfers for 256 particles:

```bash
python examples/jax_potential.py --device cpu --particles 256
python examples/jax_potential.py --device gpu --particles 256
python examples/jax_potential.py --device cpu --particles 256 --h5 data/potential/V1/PHI_2.h5 --characteristic-frequency "$OMEGA0"
```

Set `OMEGA0` to the desired finite positive source angular-frequency scale
before running the HDF5 example. Without `--h5`, the example uses a clearly
identified synthetic field. A Git LFS pointer is not a usable HDF5 input.
Potential-evaluation timings do not predict complete BM4/ABBA integration
performance.

The focused potential suites cover numerical equivalence, periodicity,
derivatives, supported spline degrees, array dispatch, CPU-only grid behavior,
immutability, device reuse, serialization, JIT composition and automatic
differentiation. Integration tests compare the shared GC/FC physics through
fixed and hybrid methods. GPU checks are skipped when hardware is unavailable.

```bash
python -m unittest discover -s tests -p 'test_jax_potential.py' -q
python -m unittest discover -s tests -p 'test_potential_execution.py' -q
python -m unittest discover -s tests -p 'test_jax_methods.py' -q
```
