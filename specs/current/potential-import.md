+++
schema_version = 1
kind = "spec"
id = "potential-import"
sources = ["src/potential/potential.py", "src/potential/load.py", "src/dynamics/gc.py", "src/diagnostics/io/persistence.py"]
documents = ["docs/dynamics/gc2d-h5-import.md"]

[[requirements]]
id = "POT-001"
tests = ["tests/test_h5_rescaling.py::H5RescalingTests.test_default_preserves_source_and_physical_phase", "tests/test_gc2d_h5_potential.py::GC2DH5ImportTests.test_first_frequency_must_be_exactly_zero"]

[[requirements]]
id = "POT-002"
tests = ["tests/test_gc2d_h5_potential.py::GC2DH5ImportTests.test_direct_order_keeps_pairs_with_explicit_frequency", "tests/test_gc2d_h5_potential.py::GC2DH5ImportTests.test_all_positive_fields_and_constant_only_selection", "tests/test_gc2d_h5_potential.py::GC2DH5ImportTests.test_selected_frequencies_must_be_finite_and_strictly_positive", "tests/test_gc2d_h5_potential.py::GC2DH5ImportTests.test_invalid_source_indices_are_not_coerced"]

[[requirements]]
id = "POT-003"
tests = ["tests/test_h5_rescaling.py::H5RescalingTests.test_reference_cycle_and_constant_only_policy", "tests/test_input_validation_contracts.py::InputValidationContractsTests.test_import_controls_are_checked_before_opening_the_file"]

[[requirements]]
id = "POT-004"
tests = ["tests/test_h5_rescaling.py", "tests/test_h5_frequency_units.py"]

[[requirements]]
id = "POT-005"
tests = ["tests/test_h5_rescaling.py::H5RescalingTests.test_orientation_selection_filtering_and_owned_metadata", "tests/test_solution_persistence.py::SolutionPersistenceTests.test_legacy_cyclic_rates_decode_once_and_unknown_conventions_fail", "tests/test_potential_snapshot.py"]
+++

# Measured potential import

This contract describes `Potential.load`, inherited `JaxPotential.load`, and
cyclic runtime semantics. See the [HDF5 contract](../../docs/dynamics/gc2d-h5-import.md)
for units, orientation, historical study reconstruction and archive boundaries.

## POT-001: Constant field and unscaled defaults

Field 0 contributes `fields[0].real`; its frequency is exactly zero. `indx`
defaults to original variable index `(15,)`; missing indices fail. Both
`spatial_rescaling` and `temporal_rescaling` default to false. With `sigma=None`,
source samples, origins, spacing, resolution and cyclic frequencies are
preserved exactly apart from floating-point coordinate representation.
Interpolation defaults to cubic; import does not resample or transpose.

## POT-002: Original selection and orientation

Selections are one-dimensional integers in caller order, paired with finite
positive source cyclic frequencies. No amplitude ranking or index conversion
occurs. `indx=()` means constant only; `None` selects all finite positive modes.
`freqs` is one-dimensional; `fields` has shape `(len(freqs), len(Zcells), len(Rcells))`.
The existing square-grid/common-period representation is retained without
transposition; incompatible sampled shapes fail explicitly.

## POT-003: Independent controls and validation

The rescaling flags accept Boolean scalars. `sigma` is None or finite nonnegative.
These controls are checked before file access. The loader rejects removed B,
characteristic length/frequency and nx/ny keywords. Temporal rescaling requires
at least one selected positive mode and uses the first selected cyclic rate
as reference. Constant-only loading remains valid without temporal rescaling.

## POT-004: Cyclic reconstruction and equivalent variables

All runtime potentials store cyclic rates and reconstruct
`mean + 2 Re sum modes*exp(i*2*pi*f*t)`; time derivatives use `(i*2*pi*f)**dt`.
Random potentials store f=1/(2*pi), preserving omega=1. The full cell period is `L=n*dx`.
With spatial rescaling, `s=L` and `q=(x-origin)/s`; otherwise s=1 and coordinates
are unchanged. With temporal rescaling, `T=1/f_ref` and `u=t/T`;
otherwise T=1. Multiply every amplitude by `T/s**2` and cyclic rates by T.
The derivative map includes the same amplitude factor, times `s**(dx+dy)*T**dt`.

GC dynamics owns finite signed nonzero B and consistently uses H=P/B, default
B=1 for prepared Hamiltonians. Gyro-radius maps as rho/s; states, steps and
sample times use the same maps. Historical radial/cycle-time conventions stay
explicitly distinct and preserve old dynamics. Historical radial equals the corrected physical-cycle Hamiltonian as a
function of cycle index; archived clocks remain legacy and are not relabeled.

## POT-005: Filtering, immutable provenance and persistence

Optional reflected Gaussian filtering applies to source mean and real/imaginary
mode components before rescaling; zero leaves values unchanged. Arrays and
source provenance are owned and read-only. Metadata records original axes,
indices, cyclic source frequencies, flags, scales, amplitude divisor and sigma.
Factories and gyroaveraging preserve subclasses. New stored fields identify
cyclic rates; known unmarked cyclic arrays are unchanged, while explicitly
angular processed arrays divide by 2*pi once on read. Source frequency units
and physical/legacy time conventions have separate metadata; missing old
provenance remains legacy_unspecified. Validate original hashes before conversion.
Unknown conventions fail, while grids, amplitudes and old trajectories remain
unchanged. Frozen archives running their own original evaluator retain its
original convention. Historical normalization and optional sampling are explicit
study composition through the common raw loader, outside the new loader API.
