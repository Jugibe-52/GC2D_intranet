"""Reconstruct a measured HDF5 field against a prior experiment's provenance."""

from collections.abc import Mapping
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import h5py

from potential.load import GC2DH5Metadata
from potential.potential import Potential
from studies.dimensional_h5_midpoint import resolve_h5_source
from studies.reference_trajectory import potential_fingerprint


def restore_h5_selection(source: str | Path, specification: Mapping[str, Any]) -> tuple[int, ...]:
    """Recover direct field indices from a saved notebook's potential settings.

    Historical single-mode PHI_2 studies sometimes saved only rank selectors.
    Their mapping is unambiguous when the source has exactly one positive mode;
    other old selections require recorded original source indices.
    """
    if "source_field_indices" in specification:
        return tuple(int(index) for index in specification["source_field_indices"])
    selection = tuple(specification["mode_selection"])
    if 0 not in selection:
        return selection
    if selection == (0,):
        return ()
    with h5py.File(resolve_h5_source(source), "r") as h5:
        frequencies = np.asarray(h5["freqs"][()], dtype=float)
    positive = np.flatnonzero(np.isfinite(frequencies) & (frequencies > 0))
    if selection == (0, 1) and positive.size == 1:
        return (int(positive[0]),)
    raise ValueError("Legacy rank selection requires recorded source_field_indices.")


def prepare_verified_h5_field(
    source: str | Path, *, magnetic_field: float, characteristic_length: float,
    source_selection: tuple[int, ...], interpolation_order: int,
    expected_source_sha256: str, original_provenance: str | Path,
) -> tuple[Potential, dict[str, Any]]:
    """Reconstruct the original physical field from its verified HDF5 source.

    Frozen numerical code is not imported. Compare dimensional scales, selected
    fields and grid with the original experiment's provenance.
    """
    resolved = resolve_h5_source(source)
    with resolved.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != expected_source_sha256:
        raise ValueError('Source HDF5 checksum differs from the original experiment.')
    original = json.loads(Path(original_provenance).read_text())
    for key, expected in {
        'B_tesla': magnetic_field, 'characteristic_length_m': characteristic_length,
        'source_field_indices': list(source_selection), 'interpolation_order': interpolation_order,
        'source_hdf5_sha256': digest, 'denoising': False, 'resampling': False,
    }.items():
        if original[key] != expected:
            raise ValueError(f'Physical setting differs from the original experiment: {key}.')
    # Archived selectors referred to amplitude ranks; original source indices
    # above identify the same physical fields under the direct-index API.
    potential = Potential.load(resolved, B=magnetic_field,
        characteristic_length=characteristic_length, indx=source_selection,
        characteristic_frequency=2 * np.pi / original["characteristic_period_s"],
        interpolation_order=interpolation_order,
        sigma=None)
    provenance = potential.metadata
    assert isinstance(provenance, GC2DH5Metadata)
    assert provenance.characteristic_period is not None
    np.testing.assert_array_equal(provenance.source_field_indices, original['source_field_indices'])
    np.testing.assert_allclose(provenance.characteristic_period, original['characteristic_period_s'], rtol=1e-13)
    np.testing.assert_allclose(provenance.normalization_factor, original['normalization_factor'], rtol=1e-13)
    for key, value in asdict(potential.grid).items():
        np.testing.assert_allclose(value, original['grid'][key], rtol=1e-13, atol=1e-13)
    return potential, {
        'source_hdf5_sha256': digest, 'source_field_indices': original['source_field_indices'],
        'field_fingerprint': potential_fingerprint(potential),
        'length_scale_m': characteristic_length / (2 * np.pi),
        'time_scale_s': provenance.characteristic_period,
        'source_origin_m': original['source_origin_m'],
        'field_construction': 'Original HDF5, normalized mean and first selected positive-frequency mode; no resampling or denoising.',
    }



__all__ = ["prepare_verified_h5_field", "restore_h5_selection"]
