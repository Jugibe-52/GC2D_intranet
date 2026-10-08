# GC2D HDF5 potential and guiding-center dynamics

This document describes the shared potential-to-dynamics boundary used by the
GC2D numerical integrators. It is independent of any particular time-integration
model; model documentation states only which capabilities it consumes.

The [execution configuration](jax-potential-evaluation.md) selects SciPy/CPU or
JAX/CPU/GPU at simulation preparation. `Potential.evaluate` uses NumPy/SciPy;
`JaxPotential` inherits the data representation and overrides only `evaluate`
with JAX. Simulation preparation shares existing splines with JAX potentials.
`evaluate_grid` remains NumPy-only; inherited electric fields follow the
potential class. The HDF5 import pipeline remains on CPU.

The HDF5 import path loads the primary GC2D field format into the potential and
simulation APIs. Its implementation lives in
[`src/potential/load.py`](../../src/potential/load.py), and the package
exports `Potential`, `JaxPotential` and `GC2DH5Metadata` from
[`src/potential/__init__.py`](../../src/potential/__init__.py).

The corresponding component and data-flow diagram is
[`gc2d-h5-potential-architecture.puml`](gc2d-h5-potential-architecture.puml).

## Responsibilities

The import path separates the following responsibilities:

- The HDF5 adapter reads, validates, selects, nondimensionalizes, and optionally
  preprocesses the fields stored in an HDF5 file. Its private data loader returns
  normalized samples and provenance without constructing a runtime potential.
- `Potential.load(...)` passes those data to `cls(...)`, preserving
  subclass construction and preparing the runtime splines once.
- `Potential` stores the resulting mean and modes using the same interpolation,
  time-reconstruction, derivative, and gyroaveraging implementation used by
  artificially generated potentials.
- `GC2DH5Metadata` stores the immutable dimensional source coordinates,
  frequencies, selection indices, scales, attributes, and source path without
  mixing those values into the runtime-field constructor signature.

The loader defines the primary GC2D HDF5 schema and runtime contract. It is not
a general-purpose HDF5 potential reader.

Private preparation functions keep these contracts separate from the import
algorithm. `_validated_import_controls` checks dimensional scales and filtering
options before opening the file; `_validated_field_selection` validates original
HDF5 indices and their strictly positive frequencies without reordering them. `GC2DH5Metadata` delegates
source-index/frequency agreement and attribute copying to its own helpers,
while preserving the order in which provenance values are validated and frozen.
Periodic construction and resampling share the sample-count check in
`potential.grid`; direct `Grid` construction retains its distinct type errors.
These helpers preserve the field conventions.

## Public entry points

```python
from potential import Potential

# Choose the time unit explicitly from the desired source mode.
import h5py

source_path = "data/potential/V1/PHI_2.h5"
with h5py.File(source_path, "r") as source:
    omega0 = float(source["freqs"][15])
potential = Potential.load(
    source_path,
    characteristic_frequency=omega0,
    characteristic_length=0.06,
    interpolation_order=3,
)
```

Use `Potential.random(...)` for generated fields and `Potential.load(...)`
for measured fields. Both produce the same runtime representation. The former
`Potential.from_gc2d_h5(...)` method and `load_gc2d_h5_potential(...)` function
have been removed; call `Potential.load(...)` with the same options. The old
`potential.gc2d_h5` module path has moved to `potential.load`. Source reading
and preprocessing remain separate from runtime evaluation.

The HDF5 class method accepts the following options:

| Argument | Default | Meaning |
| --- | --- | --- |
| `filename` | Required | Path to the GC2D HDF5 file. |
| `B` | `1.5` | Non-zero magnetic-field normalization parameter. |
| `characteristic_length` | `0.06` | Physical mode length `lambda` mapped to `2*pi`. |
| `characteristic_frequency` | Required | Finite positive source angular frequency `omega0=2*pi/T0`; `None` is rejected. |
| `indx` | `(15,)` | Original HDF5 indices of variable fields; source field 0 is always the constant term. |
| `nx`, `ny` | `None` | Optional target sizes for periodic resampling. They must be supplied together. |
| `sigma` | `None` | Gaussian standard deviation in grid samples on each axis; `None` disables filtering, and finite non-negative values enable it before resampling. |
| `interpolation_order` | `3` | Spatial spline degree, restricted to values from 2 to 5. |

`indx` directly indexes the original `fields` and `freqs` datasets. The default
`(15,)` loads `fields[15]` with `freqs[15]`, in addition to the constant field
`fields[0].real`. Passing `(15, 20)` keeps those two variable fields in that order.
There is no amplitude ranking or one-based index conversion.

Every explicitly selected frequency must be finite and strictly positive;
selecting zero or a negative frequency raises `ValueError`. Indices must be
integers within the source bounds. The first source frequency must be exactly
zero, even if another zero-frequency field exists later in the file.

`indx=()` loads only the constant field. `indx=None` selects every finite,
positive-frequency field in source order. Negative and zero-frequency channels
may exist elsewhere in the file, but cannot be explicitly selected as variables.

### Migrating studies and saved results

For the original `PHI_2.h5`, the former default `(0, 1)` and the new `(15,)`
produce identical constant and variable fields, normalized frequencies, and
normalization factors: source field 15 is its only positive-frequency channel.
Use `(15,)` in existing study settings; replace constant-only `(0,)` with `()`.
For other files, use recorded `source_field_indices` rather than guessing how
an old amplitude rank maps to a source index. A custom old frequency scale can
be retained explicitly through `characteristic_frequency`.

Study defaults share `potential.load.DEFAULT_FIELD_INDICES`. `source_selection`
in `LocalJAXStarConfig` and `ModalCPUStarConfig`, and `selectors` in
`PoincareStarConfig`, now default to that tuple and are keyword-only. Archive readers recover old rank selectors from recorded
source indices; notebook archives lacking them can recover the former `(0, 1)`
only when their source contains exactly one positive-frequency channel. Saved
trajectories and historical provenance are not rewritten.

## Expected HDF5 schema

The loader reads four datasets from the root of the file:

| Dataset | Required representation | Meaning |
| --- | --- | --- |
| `Rcells` | One-dimensional real array | Sampled x coordinates. |
| `Zcells` | One-dimensional real array | Sampled y coordinates. |
| `freqs` | One-dimensional real array | One angular frequency per stored field. |
| `fields` | Complex array with shape `(len(freqs), len(Zcells), len(Rcells))` | Mean and oscillatory spatial fields. |

Root HDF5 attributes are copied into the resulting `GC2DH5Metadata` as read-only
source metadata. Missing datasets currently produce the corresponding `h5py`
key error rather than a custom schema error.

The loader preserves the stored two-dimensional field orientation and does not
transpose the arrays. The runtime adapter treats the first spatial array axis as
x and the second as y. The current path is therefore verified for the square
GC2D grids used by the project; non-square files should not be assumed safe
until the stored `(y, x)` schema and runtime `(x, y)` convention are reconciled.

## Import pipeline

### 1. Validate loader options

Before opening the file, the loader requires:

- finite, non-zero `B`;
- finite, positive `characteristic_length`;
- required finite, positive `characteristic_frequency`; omission or `None` raises `TypeError`;
- `sigma=None` or a finite, non-negative Gaussian width.

The interpolation order is validated later, when resampling or constructing
the runtime potential, and must be an integer from 2 to 5.

After reading the file, it validates the field shape against the number of
frequencies and sampled coordinates.

### 2. Read the constant field

Require `freqs[0] == 0` exactly and use `fields[0].real` as `Phi0`.
A missing first frequency or a nonzero first frequency raises `ValueError`.
No search for another stationary channel or near-zero tolerance is used.

### 3. Select variable fields

Validate the original indices in `indx` and their frequencies, then read only
those field slices. Keep each complex coefficient paired with its frequency
and original index in the caller's order. No `ptp` calculation or sorting is
needed. An empty selection leaves the constant field alone.

### 4. Nondimensionalize space, time, and the fields

Let `lambda` be `characteristic_length`. Let `omega0` be
`characteristic_frequency`, which must be supplied explicitly even for a
constant-only selection. The characteristic period is
`T0 = 2*pi/omega0`. Runtime time counts complete characteristic periods:

```text
x_hat = 2*pi*(x - x[0])/lambda
y_hat = 2*pi*(y - y[0])/lambda
t_hat = t/T0 = omega0*t/(2*pi)
f_hat_j = omega_j/omega0
Phi_hat = (2*pi)**2*Phi/(omega0*lambda**2*B)
```

Characteristic-length scaling is the only HDF5 coordinate convention. Both
axes start at zero and use the same factor `2*pi/lambda`; a complete source
period `P` becomes `2*pi*P/lambda`, which equals `2*pi` only when `P=lambda`.
The loader does not infer `lambda` from the source cell size. Spatial derivatives
and gyroaverage radii use these dimensionless runtime coordinates.

The `spatial_normalization` argument and metadata field, the `SpatialNormalization`
export, and the `unit_box` loading mode have been removed. Existing calls that
explicitly selected `"characteristic_length"` should omit that keyword. Consumers
of the former `unit_box` mode must adapt coordinates, radii and any Hamiltonian
scaling to the characteristic-length convention. A unit-cell display can still
be constructed as postprocessing, without changing integration coordinates.

The implementation retains a divisor-style provenance value,

```text
normalization_factor = omega0*lambda**2*B/(2*pi)**2,
```

and divides the mean field and every retained mode by that value. The
first selected mode completes one cycle per normalized time unit and has
temporal period `1` only when its source frequency equals `omega0`. The complete `PHI_2.h5` spatial box has length `0.18`, so
the default `lambda=0.06` maps it to a dimensionless box of length `6*pi`.

With the default single-mode selection and `omega0` equal to that source
frequency, the complete potential is periodic in runtime time with period `1`. If several modes are
selected, a finite common temporal period exists only when all normalized
frequency ratios are commensurate; `1` need not then be a period of the
combined field.

If no variable field is selected, the required characteristic frequency still
sets the normalization factor and characteristic period; the constant field remains.
A missing default source index 15 is an error, not a fallback to another mode.

Existing callers that relied on automatic frequency selection must now pass
the desired source frequency explicitly. Selection order no longer determines
the normalization scale.

### 5. Preserve selected-field provenance

The constant field is always present. Variable-field selection order is preserved. Runtime `frequencies`, dimensional `source_frequencies`,
`modes`, and `source_field_indices` remain aligned.

### 6. Apply optional denoising

The adapter delegates filtering to `_denoise_fields` in
[`src/potential/potential.py`](../../src/potential/potential.py), preserving
the source samples and SciPy's default reflected boundary handling.

When `sigma` is not `None`, `scipy.ndimage.gaussian_filter` is applied to:

- the real mean field, when present;
- the real and imaginary parts of each mode separately.

Denoising takes place after normalization and selection but before resampling.
A width of zero leaves the fields unchanged. The `denoising` argument has been
removed: migrate `denoising=False` to `sigma=None`, and `denoising=True` to
`sigma=1.0` (or the previously supplied width).

### 7. Apply optional resampling

When `nx` and `ny` are provided, the selected fields are evaluated on new
half-open uniform axes with no duplicated periodic endpoint. The temporary
interpolators use the same wrapped spline recipe as runtime evaluation.
Both paths call the private `potential._periodic_spline._build_periodic_spline`
helper, which owns axis padding, sample wrapping, and separate real/imaginary
fits. Each caller supplies its validated axis spacing, preserving the grid's
stored precision and the existing treatment of periodic boundaries.

Resampling therefore introduces two interpolation stages:

```text
HDF5 samples
    -> article nondimensionalization
    -> temporary periodic splines
    -> resampled field arrays
    -> persistent periodic runtime splines
```

### 8. Construct metadata and `Potential`

The final step first constructs one `GC2DH5Metadata` value with the following
provenance information:

- dimensional source frequencies for the selected runtime modes;
- dimensional source axes;
- characteristic length, frequency, and period;
- original HDF5 field indices;
- normalization factor;
- root attributes;
- source path.

New saved-solution archives omit the redundant spatial-mode field from HDF5
provenance. The archive reader accepts older provenance explicitly marked
`"characteristic_length"` and removes that redundant marker when reconstructing
metadata. It rejects older `"unit_box"` provenance with a migration error; those
results must be regenerated with the supported convention rather than relabeled.
Old pickled metadata containing the removed constructor argument must likewise
be regenerated. Other experiment metadata, including a study's own coordinate
representation, is retained.

It then constructs the common `Potential` from the processed runtime arrays,
the single metadata value, and the interpolation order:

```python
Potential(
    grid,
    mean=mean,
    modes=modes,
    frequencies=frequencies,
    metadata=metadata,
    interpolation_order=3,
)
```

The loader owns the extraction and calculation of provenance. The potential
retains the resulting immutable metadata after the HDF5 file is closed. HDF5
provenance is accessed explicitly through attributes such as
`potential.metadata.source_x` and `potential.metadata.normalization_factor`.

## Runtime representation

[`Potential`](../../src/potential/potential.py) owns the runtime representation;
[`JaxPotential`](../../src/potential/jax_potential.py) inherits it and overrides
only pointwise evaluation. Artificial construction and HDF5 loading produce the same mean-plus-modes
representation; their origin differs only through optional provenance metadata.

The common representation provides the following behavior:

- all stored arrays and metadata are exposed as immutable values;
- HDF5 dimensional provenance is grouped in `potential.metadata`, while an
  artificial potential uses `metadata=None`;
- every complex spatial field uses one real and one imaginary
  `RectBivariateSpline`;
- the axes are extended beyond both sides of the omitted periodic endpoint;
- extended field values wrap samples from the opposite edge;
- query coordinates are reduced modulo the dimensionless box period;
- values and spatial derivatives obey the same periodic wrapping;
- runtime interpolation is periodic for every potential origin.

The associated `Grid` period is the complete runtime source-box length. In the
characteristic-length convention it is not necessarily `2*pi`; for the primary
file and default characteristic length, the period is `6*pi` because the `0.18`
source box contains three characteristic lengths of `0.06`.

## Time reconstruction and derivatives

The physical potential is reconstructed as

```text
Phi(t, x, y) = Phi0(x, y)
             + 2 Re sum_j[C_j(x, y) exp(+i 2*pi*f_hat_j*t_hat)].
```

The main runtime methods are:

- `evaluate(t, x, y, ...)`: reconstructs the mean and positive-frequency modes
  at paired coordinates with the same shape;
- `evaluate_grid(t, ...)`: reconstructs them on the complete stored grid;
- `electric_field(...)`: evaluates `(-Phi_x, -Phi_y)` through the same
  frequency-aware derivative machinery;
- `gyroaverage(rho)`: applies the Larmor-circle average to every stored field.

Spatial derivative orders are delegated to the persistent splines. The time
derivative interface supports `dt=0`, `dt=1`, and `dt=2`. Every oscillatory
mode is multiplied by `(i*2*pi*f_hat_j)**dt`; therefore `dt=1` reconstructs
`Phi_t_hat` and `dt=2` reconstructs `Phi_t_hat_t_hat` with the dimensionless
frequency of each selected mode.
The stationary mean contributes only for `dt=0` and contributes zero to both
time derivatives. Spatial and time orders can be combined, so calls such as
`evaluate(..., dx=1, dt=1)` and `evaluate(..., dy=1, dt=1)` provide `Phi_xt`
and `Phi_yt`.

## Gyroaveraging

`GuidingCenterDynamics` constructs its effective potential once:

```python
self.effective_potential = potential.gyroaverage(rho)
```

For `rho=0`, the potential returns itself. For positive `rho`, each mean or
mode field is transformed with a two-dimensional FFT and multiplied by

```text
J0(2*pi*rho*sqrt(kx**2 + ky**2)).
```

The inverse FFT produces a new `Potential`. Its immutable `GC2DH5Metadata`
value is reused, preserving dimensional source axes and
frequencies, characteristic scales, source indices, normalization, attributes,
and source path. The runtime frequencies and interpolation order are also
preserved.

## Guiding-center consumption

The effective HDF5 potential supplies all field operations required by
[`GuidingCenterDynamics`](../../src/dynamics/gc.py):

```text
vector_field = (-Phi_y, Phi_x)
hamiltonian = Phi
extended_momentum_derivative = -Phi_t
```

Derivative requirements distinguish the physical solve from energy monitoring:

- Spatial Newton uses `Phi_xx`, `Phi_xy`, and `Phi_yy`, with or without tracking.
- Spatial Broyden needs field values and no analytic residual Jacobian.
- `track_energy=True` additionally uses `Phi_t` for the passive momentum. It
  never projects the clock or momentum, so it requires no mixed or second time
  derivatives for the physical solve.

Spatial analytic second derivatives require `interpolation_order >= 3`.
The potential's mixed derivatives and frequency-aware `dt=2` remain available
for independent diagnostics, but the retired full time/momentum projection is
not an active method configuration. All energy histories use normalized
physical `kappa`, whose derivative is `-Phi_t`.

A minimal simulation setup is:

```python
import numpy as np

from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import ABBA4Implicit, InitialValueProblem, SimulationRequest, simulate

# Choose the time unit explicitly from the desired source mode.
import h5py

source_path = "data/potential/V1/PHI_2.h5"
with h5py.File(source_path, "r") as source:
    omega0 = float(source["freqs"][15])
potential = Potential.load(
    source_path,
    characteristic_frequency=omega0,
    characteristic_length=0.06,
    interpolation_order=3,
)
dynamics = GuidingCenterDynamics(potential, rho=0.0)
configuration = GCInitialConfiguration.from_components(
    x=np.asarray([potential.grid.period / 2]),
    y=np.asarray([potential.grid.period / 2]),
)
problem = InitialValueProblem(dynamics, configuration)
request = SimulationRequest.uniform(
    t_span=(0.0, 1.0),
    max_step=1e-3,
    sample_count=101,
)
solution = simulate(
    problem,
    ABBA4Implicit(
        projection_formulation="reduced_multiplier",
        nonlinear_solver="newton",
        state_extension="physical",
    ),
    request,
)
```

## Invariants and limitations

- Coordinate axes must be one-dimensional, finite, strictly increasing, and
  uniformly spaced.
- The current `Grid` contract requires equal sampled spans along x and y.
- Mean and mode arrays must be finite and match the coordinate shape.
- Every selected mode must have one finite, positive frequency.
- At least one source mean or mode field must remain after HDF5 selection;
  the generic `Potential` can also represent the identically zero field.
- `nx` and `ny` must be supplied together and must each be at least 2.
- Spatial coordinates must be supplied as an x-y pair and are periodically
  wrapped into the dimensionless source box.
- Fully extended Newton requires a potential implementation whose
  `evaluate(..., dt=2)` contract returns the true second time derivative. The
  common implementation satisfies this contract mode by mode, including a zero
  stationary-mean contribution.

## Verification

[`tests/test_gc2d_h5_potential.py`](../../tests/test_gc2d_h5_potential.py)
verifies:

- positive-frequency filtering, amplitude ordering, article
  nondimensionalization, selection, and positive phase reconstruction;
- denoising and both periodic interpolation stages;
- spatial first and second derivatives, first and second time derivatives,
  mixed space-time derivatives, and coordinate wrapping;
- gyroaveraging and compatibility with `GuidingCenterDynamics`;
- a complete HDF5 -> dynamics -> implicit ABBA4 -> `Solution` integration;
- rejection of invalid selection, resampling, scale, and magnetic-field inputs.
