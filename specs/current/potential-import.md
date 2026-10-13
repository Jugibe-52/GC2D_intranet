+++
schema_version = 1
kind = "spec"
id = "potential-import"
sources = ["src/potential/potential.py", "src/potential/load.py"]
documents = ["docs/dynamics/gc2d-h5-import.md"]

[[requirements]]
id = "POT-001"
tests = ["tests/test_gc2d_h5_potential.py::GC2DH5ImportTests.test_defaults_select_mean_and_source_field_15_with_B_1_5", "tests/test_gc2d_h5_potential.py::GC2DH5ImportTests.test_first_frequency_must_be_exactly_zero"]

[[requirements]]
id = "POT-002"
tests = ["tests/test_gc2d_h5_potential.py::GC2DH5ImportTests.test_direct_order_keeps_pairs_with_explicit_frequency", "tests/test_gc2d_h5_potential.py::GC2DH5ImportTests.test_all_positive_fields_and_constant_only_selection", "tests/test_gc2d_h5_potential.py::GC2DH5ImportTests.test_selected_frequencies_must_be_finite_and_strictly_positive", "tests/test_gc2d_h5_potential.py::GC2DH5ImportTests.test_invalid_source_indices_are_not_coerced"]

[[requirements]]
id = "POT-003"
tests = ["tests/test_input_validation_contracts.py::InputValidationContractsTests.test_import_requires_explicit_positive_frequency_before_file_access", "tests/test_input_validation_contracts.py::InputValidationContractsTests.test_import_controls_are_checked_before_opening_the_file"]

[[requirements]]
id = "POT-004"
tests = ["tests/test_gc2d_h5_potential.py::GC2DH5ImportTests.test_characteristic_length_sets_scale_independently_of_cell_period", "tests/test_gc2d_h5_potential.py::GC2DH5ImportTests.test_explicit_characteristic_frequency_controls_time_and_amplitude", "tests/test_gc2d_h5_potential.py::GC2DH5ImportTests.test_explicit_frequency_normalizes_a_mean_only_file"]

[[requirements]]
id = "POT-005"
tests = ["tests/test_gc2d_h5_potential.py::GC2DH5ImportTests.test_optional_sigma_preserves_unsmoothed_fields", "tests/test_input_validation_contracts.py::InputValidationContractsTests.test_field_selection_requires_integers_and_keeps_frozen_provenance", "tests/test_gc2d_h5_potential.py::GC2DH5ImportTests.test_load_preserves_subclass"]
+++

# Measured potential import

This baseline describes `Potential.load`, including inherited loading for
`JaxPotential`. It does not specify interpolation algorithms or every potential
constructor. See the [HDF5 contract](../../docs/dynamics/gc2d-h5-import.md) for
the source schema and mathematical derivation.

## POT-001: Constant field and default selection

Source field 0 contributes `fields[0].real` as the constant term, and its
frequency must equal zero exactly. `indx` defaults to `(15,)`, the original
source index of the selected variable field. The constant field is included
independently of `indx`. A source without the requested index fails; the loader
does not silently choose a replacement. Defaults include `B=1.5`,
`characteristic_length=0.06`, and cubic interpolation.

## POT-002: Direct source indices and frequency pairing

Explicit indices are a one-dimensional integer selection into the original
HDF5 arrays, in caller order. Every selected frequency is finite and strictly
positive; out-of-range or non-integer indices are rejected. No ranking by
amplitude or one-based conversion occurs. `indx=()` selects the constant field
alone; `indx=None` selects all finite positive-frequency entries in source order.
The source `freqs` array is one-dimensional, and `fields` has shape
`(number_of_frequencies, number_of_y_samples, number_of_x_samples)`.

## POT-003: Explicit physical scales and early validation

`characteristic_frequency` is a required, finite positive angular frequency in
source units, including constant-only loading. Omission or `None` raises
`TypeError`; non-positive or non-finite numeric values raise `ValueError`.
`characteristic_length` is finite and positive in the source coordinate units.
`B` is finite and nonzero and retains its sign. These scales and `sigma` are
validated before the source file is opened.

## POT-004: Spatial, temporal, and potential normalization

For length `lambda` and angular frequency `omega0`, the characteristic period
is `T0=2*pi/omega0`. Runtime coordinates and time are dimensionless:

```text
x_hat = 2*pi*(x-x[0])/lambda
y_hat = 2*pi*(y-y[0])/lambda
t_hat = t/T0
f_hat_j = omega_j/omega0
Phi_hat = Phi / (omega0*lambda**2*B/(2*pi)**2)
```

The same amplitude scaling applies to the mean and selected modes, even with
no variable field. Spatial period is derived from the source extent and length
scale. A variable mode has period one only if its source frequency equals
`omega0`; a multimode field need not have period one. These are analytical
contracts; numerical comparisons use explicit floating-point tolerances in the
linked regression tests, not a universal error guarantee.

## POT-005: Optional filtering and source provenance

`sigma=None` disables Gaussian smoothing; a finite nonnegative width enables it
in grid-sample units, with zero leaving the fields unchanged. Selection and
normalization precede filtering. The returned metadata retains original source
indices, dimensional frequencies, scales, and source attributes in owned,
read-only arrays. Loading through a subclass preserves that class.
