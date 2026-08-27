# Importing GC2D potentials from HDF5

The HDF5 import path loads the primary GC2D field format into the potential and
simulation APIs. Its implementation lives in
[`src/potential/gc2d_h5.py`](../../src/potential/gc2d_h5.py), and the package
exports both `load_gc2d_h5_potential` and `GC2DH5Potential` from
[`src/potential/__init__.py`](../../src/potential/__init__.py).

The corresponding component and data-flow diagram is
[`gc2d-h5-potential-architecture.puml`](gc2d-h5-potential-architecture.puml).

## Responsibilities

The import path has two distinct responsibilities:

- `load_gc2d_h5_potential(...)` reads, validates, selects, normalizes, and
  optionally preprocesses the fields stored in an HDF5 file.
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
    interpolation_order=3,
)
```

The loader accepts the following options:

| Argument | Default | Meaning |
| --- | --- | --- |
| `filename` | Required | Path to the GC2D HDF5 file. |
| `B` | `1.5` | Non-zero magnetic-field normalization parameter. |
| `indx` | `(0, 1)` | Selected mean and fluctuation indices after mode ordering. |
| `nx`, `ny` | `None` | Optional target sizes for inclusive-grid resampling. They must be supplied together. |
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
| `freqs` | One-dimensional real array | One temporal frequency per stored field. |
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

### 4. Normalize the fields

Let `f0` be the frequency of the first mode after amplitude sorting. The loader
uses

```text
normalization_factor = 2*pi*f0*B
```

and divides the mean field and every retained fluctuation by this same factor.
Consequently, `f0` is the frequency of the largest-amplitude retained mode; it
is not necessarily the smallest positive frequency.

If the file contains no retained positive-frequency mode, the normalization
factor defaults to `1.0` and only the selected mean can be constructed.

### 5. Apply `indx`

The loader selects the requested mean and sorted fluctuation modes. Selection
order is preserved, so the returned `frequencies`, `fluctuations`, and
`source_field_indices` remain aligned.

### 6. Apply optional denoising

When `denoising=True`, `scipy.ndimage.gaussian_filter` is applied to:

- the real mean field, when present;
- the real and imaginary parts of each fluctuation separately.

Denoising takes place after normalization and selection but before resampling.

### 7. Apply optional resampling

When `nx` and `ny` are provided, the selected fields are evaluated on new
inclusive `numpy.linspace` axes. The temporary interpolators use the
zero-padded spline recipe defined by the GC2D HDF5 format.

Resampling therefore introduces two interpolation stages:

```text
HDF5 samples
    -> temporary zero-padded splines
    -> resampled field arrays
    -> persistent runtime splines
```

### 8. Construct `GC2DH5Potential`

The final step constructs a `GC2DH5Potential` with the processed arrays and the
following provenance information:

- selected frequencies;
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
- the axes are extended by `interpolation_order + 1` samples on both sides;
- extended field values are filled with zeros;
- query coordinates are clipped to the sampled bounds;
- runtime HDF5 interpolation is non-periodic.

The associated `Grid` records regular periodic coordinates because the shared
simulation API and the FFT-based gyroaverage require that metadata. It does not
change the clipping and zero-padding behavior of the overridden HDF5
evaluation.

## Time reconstruction and derivatives

The physical potential is reconstructed as

```text
Phi(t, x, y) = Phi0(x, y)
             + 2 Re sum_j[C_j(x, y) exp(+i f_j t)].
```

The main runtime methods are:

- `dynamic_part(...)`: evaluates only the selected positive-frequency modes;
- `evaluate(...)`: adds the mean field when appropriate;
- `electric_field(...)`: inherited from `Potential` and evaluated as
  `(-Phi_x, -Phi_y)` through the overridden HDF5 `evaluate(...)` method;
- `gyroaverage(rho)`: applies the Larmor-circle average to every stored field.

Spatial derivative orders are delegated to the persistent splines. The time
derivative interface currently supports `dt=0` and `dt=1`; for `dt=1`, every
oscillatory phase is multiplied by `i*f_j`, and the stationary mean contributes
zero.

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

The inverse FFT produces a new `GC2DH5Potential`. Frequencies, source indices,
normalization, attributes, source path, and interpolation order are preserved.

## Guiding-center consumption

The effective HDF5 potential supplies all field operations required by
[`GuidingCenterDynamics`](../../src/dynamics/gc.py):

```text
vector_field = (-Phi_y, Phi_x)
hamiltonian = Phi
extended_momentum_derivative = -Phi_t
```

Exact particle Jacobians are assembled from `Phi_xx`, `Phi_xy`, and `Phi_yy`.
These second derivatives require `interpolation_order >= 3`.

A minimal simulation setup is:

```python
import numpy as np

from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import load_gc2d_h5_potential
from simulation import ABBA4Implicit1, InitialValueProblem, SimulationRequest, simulate

potential = load_gc2d_h5_potential(
    "data/potential/V1/PHI_2.h5",
    interpolation_order=3,
)
dynamics = GuidingCenterDynamics(potential, rho=0.0)
configuration = GCInitialConfiguration.from_components(
    x=np.asarray([0.22]),
    y=np.asarray([0.15]),
)
problem = InitialValueProblem(dynamics, configuration)
request = SimulationRequest.uniform(
    t_span=(0.0, 1.0),
    max_step=1e-3,
    sample_count=101,
)
solution = simulate(problem, ABBA4Implicit1(), request)
```

## Invariants and limitations

- Coordinate axes must be one-dimensional, finite, strictly increasing, and
  uniformly spaced.
- The current `Grid` contract requires equal sampled spans along x and y.
- Mean and fluctuation arrays must be finite and match the coordinate shape.
- Every selected fluctuation must have one finite, positive frequency.
- At least one mean or fluctuation field must remain after selection.
- `nx` and `ny` must be supplied together and must each be at least 2.
- Spatial coordinates must be supplied as an x-y pair and are clipped rather
  than periodically wrapped.
- General mean-plus-multifrequency HDF5 potentials should not yet be presented
  as unconditionally compatible with the fully extended analytic R4/R8
  methods. Their current second-time-derivative shortcut assumes the generic
  unit-frequency harmonic relation `Phi_tt = -Phi`; arbitrary HDF5 frequencies
  require a general `Phi_tt` implementation.

## Verification

[`tests/test_gc2d_h5_potential.py`](../../tests/test_gc2d_h5_potential.py)
verifies:

- positive-frequency filtering, amplitude ordering, normalization, selection,
  and positive phase reconstruction;
- denoising and both interpolation stages;
- spatial first and second derivatives, the first time derivative, and
  coordinate clipping;
- gyroaveraging and compatibility with `GuidingCenterDynamics`;
- a complete HDF5 -> dynamics -> implicit ABBA4 -> `Solution` integration;
- rejection of invalid selection, resampling, and magnetic-field inputs.
