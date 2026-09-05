# GC2D HDF5 potential and guiding-center dynamics

This document describes the shared potential-to-dynamics boundary used by the
GC2D numerical integrators. It is independent of any particular time-integration
model; model documentation states only which capabilities it consumes.

The HDF5 import path loads the primary GC2D field format into the potential and
simulation APIs. Its implementation lives in
[`src/potential/gc2d_h5.py`](../../src/potential/gc2d_h5.py), and the package
exports both `load_gc2d_h5_potential` and `GC2DH5Potential` from
[`src/potential/__init__.py`](../../src/potential/__init__.py).

The corresponding component and data-flow diagram is
[`gc2d-h5-potential-architecture.puml`](gc2d-h5-potential-architecture.puml).

## Responsibilities

The import path has two distinct responsibilities:

- `load_gc2d_h5_potential(...)` reads, validates, selects, nondimensionalizes,
  and optionally preprocesses the fields stored in an HDF5 file.
- `GC2DH5Potential` stores the resulting fields and preserves their HDF5
  interpolation, time reconstruction, derivative, and gyroaveraging semantics
  behind the common `Potential` interface.

The loader defines the primary GC2D HDF5 schema and runtime contract. It is not
a general-purpose HDF5 potential reader.

## Public entry point

```python
from potential import load_gc2d_h5_potential

potential = load_gc2d_h5_potential(
    "data/potential/V1/PHI_2.h5",
    characteristic_length=0.06,
    interpolation_order=3,
)
```

The loader accepts the following options:

| Argument | Default | Meaning |
| --- | --- | --- |
| `filename` | Required | Path to the GC2D HDF5 file. |
| `B` | `1.5` | Non-zero magnetic-field normalization parameter. |
| `characteristic_length` | `0.06` | Physical fluctuation length `lambda` mapped to `2*pi`. |
| `characteristic_frequency` | `None` | Positive source angular frequency `omega0=2*pi/T0`; the dominant sorted mode is used when omitted. |
| `indx` | `(0, 1)` | Selected mean and fluctuation indices after mode ordering. |
| `nx`, `ny` | `None` | Optional target sizes for periodic resampling. They must be supplied together. |
| `denoising` | `False` | Enables Gaussian filtering before optional resampling. |
| `sigma` | `1.0` | Non-negative standard deviation used by the Gaussian filter. |
| `interpolation_order` | `3` | Spatial spline degree, restricted to values from 2 to 5. |

`indx` uses the GC2D HDF5 selection semantics:

- index `0` selects the mean field;
- index `1` selects the first positive-frequency mode after sorting;
- index `2` selects the second sorted mode, and so on.

The default `indx=(0, 1)` selects the mean and the dominant retained
positive-frequency mode. Passing `indx=None` explicitly selects the mean
position and every retained positive-frequency mode. A missing mean field
simply leaves that component absent.

## Expected HDF5 schema

The loader reads four datasets from the root of the file:

| Dataset | Required representation | Meaning |
| --- | --- | --- |
| `Rcells` | One-dimensional real array | Sampled x coordinates. |
| `Zcells` | One-dimensional real array | Sampled y coordinates. |
| `freqs` | One-dimensional real array | One angular frequency per stored field. |
| `fields` | Complex array with shape `(len(freqs), len(Zcells), len(Rcells))` | Mean and oscillatory spatial fields. |

Root HDF5 attributes are copied into the resulting potential as read-only source
metadata. Missing datasets currently produce the corresponding `h5py` key
error rather than a custom schema error.

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
- finite, positive `characteristic_frequency` when explicitly supplied;
- a Boolean `denoising` value;
- finite, non-negative `sigma`.

The interpolation order is validated later, when resampling or constructing
the runtime potential, and must be an integer from 2 to 5.

After reading the file, it validates the field shape against the number of
frequencies and sampled coordinates.

### 2. Identify the mean field

Frequencies numerically close to zero are detected with a tolerance of
`1e-5`. If at least one is present, the real part of the first such field becomes
the mean potential `Phi0`.

Additional zero-frequency entries do not enter the oscillatory reconstruction.

### 3. Retain and order fluctuation modes

Negative-frequency fields are discarded. The remaining strictly positive
frequencies and their complex spatial fields are sorted by descending
peak-to-peak field amplitude.

This ordering is important because both the normalization frequency and the
format's `indx` values refer to the sorted list, not the original HDF5 order.
The original source indices are retained in `source_field_indices` for
traceability.

### 4. Nondimensionalize space, time, and the fields

Let `lambda` be `characteristic_length`. Let `omega0` be
`characteristic_frequency`, or the frequency of the first mode after amplitude
sorting when the argument is omitted. The characteristic period is
`T0 = 2*pi/omega0`. Runtime time counts complete characteristic periods:

```text
x_hat = 2*pi*(x - x[0])/lambda
y_hat = 2*pi*(y - y[0])/lambda
t_hat = t/T0 = omega0*t/(2*pi)
f_hat_j = omega_j/omega0
Phi_hat = (2*pi)**2*Phi/(omega0*lambda**2*B)
```

The implementation retains a divisor-style provenance value,

```text
normalization_factor = omega0*lambda**2*B/(2*pi)**2,
```

and divides the mean field and every retained fluctuation by that value. The
dominant mode therefore completes one cycle per normalized time unit and has
temporal period `1`. The complete `PHI_2.h5` spatial box has length `0.18`, so
the default `lambda=0.06` maps it to a dimensionless box of length `6*pi`.

With the default single-mode selection, the complete potential is therefore
exactly periodic in runtime time with period `1`. If several modes are
selected, a finite common temporal period exists only when all normalized
frequency ratios are commensurate; `1` need not then be a period of the
combined field.

If the file contains no retained positive-frequency mode, the normalization
factor defaults to `1.0` unless a characteristic frequency is supplied; only a
selected mean can then be constructed.

### 5. Apply `indx`

The loader selects the requested mean and sorted fluctuation modes. Selection
order is preserved. Runtime `frequencies`, dimensional `source_frequencies`,
`fluctuations`, and `source_field_indices` remain aligned.

### 6. Apply optional denoising

When `denoising=True`, `scipy.ndimage.gaussian_filter` is applied to:

- the real mean field, when present;
- the real and imaginary parts of each fluctuation separately.

Denoising takes place after normalization and selection but before resampling.

### 7. Apply optional resampling

When `nx` and `ny` are provided, the selected fields are evaluated on new
half-open uniform axes with no duplicated periodic endpoint. The temporary
interpolators use the same wrapped spline recipe as runtime evaluation.

Resampling therefore introduces two interpolation stages:

```text
HDF5 samples
    -> article nondimensionalization
    -> temporary periodic splines
    -> resampled field arrays
    -> persistent periodic runtime splines
```

### 8. Construct `GC2DH5Potential`

The final step constructs a `GC2DH5Potential` with the processed arrays and the
following provenance information:

- dimensionless selected frequencies and dimensional source frequencies;
- dimensional source axes;
- characteristic length, frequency, and period;
- original HDF5 field indices;
- normalization factor;
- root attributes;
- source path;
- interpolation order.

## Runtime representation

`GC2DH5Potential` subclasses the generic
[`Potential`](../../src/potential/potential.py). This inheritance is important
because `GuidingCenterDynamics` accepts objects through the common `Potential`
API and performs a strict runtime type check.

The HDF5 subclass defines its own format-specific evaluation behavior:

- all stored arrays and metadata are exposed as immutable values;
- every complex spatial field uses one real and one imaginary
  `RectBivariateSpline`;
- the axes are extended beyond both sides of the omitted periodic endpoint;
- extended field values wrap samples from the opposite edge;
- query coordinates are reduced modulo the dimensionless box period;
- values and spatial derivatives obey the same periodic wrapping;
- runtime HDF5 interpolation is periodic.

The associated `Grid` period is the complete dimensionless source-box length,
not necessarily `2*pi`. For the primary file and default characteristic length,
the period is `6*pi` because the `0.18` source box contains three characteristic
lengths of `0.06`.

## Time reconstruction and derivatives

The physical potential is reconstructed as

```text
Phi(t, x, y) = Phi0(x, y)
             + 2 Re sum_j[C_j(x, y) exp(+i 2*pi*f_hat_j*t_hat)].
```

The main runtime methods are:

- `dynamic_part(...)`: evaluates only the selected positive-frequency modes;
- `evaluate(...)`: adds the mean field when appropriate;
- `electric_field(...)`: inherited from `Potential` and evaluated as
  `(-Phi_x, -Phi_y)` through the overridden HDF5 `evaluate(...)` method;
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

For `rho=0`, the HDF5 potential returns itself. For positive `rho`, each mean or
fluctuation field is transformed with a two-dimensional FFT and multiplied by

```text
J0(2*pi*rho*sqrt(kx**2 + ky**2)).
```

The inverse FFT produces a new `GC2DH5Potential`. Frequencies, dimensional
source axes and frequencies, characteristic scales, source indices,
normalization, attributes, source path, and interpolation order are preserved.

## Guiding-center consumption

The effective HDF5 potential supplies all field operations required by
[`GuidingCenterDynamics`](../../src/dynamics/gc.py):

```text
vector_field = (-Phi_y, Phi_x)
hamiltonian = Phi
extended_momentum_derivative = -Phi_t
```

Derivative requirements depend on the selected ABBA configuration:

- Newton with `state_extension="physical"` assembles exact particle Jacobians
  from `Phi_xx`, `Phi_xy`, and `Phi_yy`, independently of `track_energy`.
- Newton with `state_extension="fully_extended"` uses that spatial Hessian and
  additionally evaluates `Phi_xt`, `Phi_yt`, and `Phi_tt` to build the analytic
  `4 x 4` extended-vector-field Jacobian.
- Broyden evaluates the selected residual without analytic residual Jacobians.
  Physical Broyden with `track_energy=False` therefore needs field values only.
  Enabling energy tracking additionally uses `Phi_t` for the auxiliary
  conjugate-momentum update. Fully extended Broyden also uses `Phi_t`, but
  neither case requires the Hessian, mixed derivatives, or `Phi_tt`.

The spatial second derivatives used by either Newton branch require
`interpolation_order >= 3`. The HDF5 implementation's frequency-aware `dt=2`
contract makes fully extended Newton valid for stationary means and arbitrary
positive multifrequency selections; it does not use the unit-frequency
shortcut `Phi_tt=-Phi`.

A minimal simulation setup is:

```python
import numpy as np

from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import load_gc2d_h5_potential
from simulation import ABBA4Implicit, InitialValueProblem, SimulationRequest, simulate

potential = load_gc2d_h5_potential(
    "data/potential/V1/PHI_2.h5",
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
- Mean and fluctuation arrays must be finite and match the coordinate shape.
- Every selected fluctuation must have one finite, positive frequency.
- At least one mean or fluctuation field must remain after selection.
- `nx` and `ny` must be supplied together and must each be at least 2.
- Spatial coordinates must be supplied as an x-y pair and are periodically
  wrapped into the dimensionless source box.
- Fully extended Newton requires a potential implementation whose
  `evaluate(..., dt=2)` contract returns the true second time derivative. The
  HDF5 implementation satisfies this contract mode by mode, including a zero
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
