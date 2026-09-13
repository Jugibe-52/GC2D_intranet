"""Pickle-free, identity-checked storage for a single-particle recurrence study."""

from __future__ import annotations

import hashlib
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy


def file_sha256(path: str | Path) -> str:
    """Hash an input or source file without loading the complete file at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recurrence_identity(project_root: Path, *, source_archive: Path, potential_path: Path,
                        initial: np.ndarray, config: dict[str, Any],
                        potential_specification: dict[str, Any]) -> dict[str, Any]:
    """Fingerprint the source particle, interpolated ODE implementation and controls."""
    source_files = ("src/studies/single_particle_recurrence.py", "src/dynamics/gc.py",
                    "src/dynamics/_layout.py", "src/potential/potential.py",
                    "src/potential/gc2d_h5.py", "src/potential/grid.py")
    return dict(particle_number=2, initial_state=np.asarray(initial).tolist(),
                config=config, potential=potential_specification,
                source_archive_sha256=file_sha256(source_archive),
                potential_sha256=file_sha256(potential_path),
                code_sha256={name: file_sha256(project_root / name) for name in source_files},
                numpy_version=np.__version__, scipy_version=scipy.__version__)


def save_recurrence(path: str | Path, arrays: dict[str, np.ndarray],
                    metadata: dict[str, Any], identity: dict[str, Any]) -> None:
    """Atomically persist numerical results and the complete calculation identity."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    record = {"schema_version": 1, "identity": identity, "result": metadata}
    np.savez_compressed(temporary, metadata_json=json.dumps(record, allow_nan=False), **arrays)
    temporary.replace(path)


def load_recurrence(path: str | Path, identity: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Reject stale inputs, settings, code or library versions before reuse."""
    with np.load(path, allow_pickle=False) as archive:
        record = json.loads(str(archive["metadata_json"]))
        if record.get("schema_version") != 1 or record.get("identity") != json.loads(json.dumps(identity)):
            raise ValueError("Recurrence cache identity changed; set recompute=True and rerun.")
        arrays = {key: archive[key].copy() for key in archive.files if key != "metadata_json"}
    times = arrays["times"]
    if times.ndim != 1 or times.size < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("Invalid recurrence time grid.")
    for label in ("DOP853", "DOP853_refined", "Radau"):
        if arrays[label + ".states"].shape != (2, times.size):
            raise ValueError("Invalid single-particle state history.")
        np.testing.assert_array_equal(arrays[label + ".states"][:, 0], arrays["initial_state"])
    if any(not np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError("Nonfinite recurrence archive.")
    return arrays, record["result"]


def export_recurrence_tables(directory: str | Path, metadata: dict[str, Any],
                             cycle_rows: list[dict[str, Any]], identity: dict[str, Any]) -> None:
    """Export all resolved minima, integer returns and threshold visits as CSV."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    windows = [{"threshold_fraction": threshold['fraction'], **window}
               for threshold in metadata['thresholds'] for window in threshold['windows']]
    for name, rows in (("spatial_returns", metadata['returns']), ("cycle_returns", cycle_rows),
                       ("threshold_windows", windows)):
        if rows:
            with (directory / f"{name}.csv").open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    (directory / 'metadata.json').write_text(
        json.dumps({"identity": identity, "result": metadata}, indent=2, allow_nan=False), encoding='utf-8')
