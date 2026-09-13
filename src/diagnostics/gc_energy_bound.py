"""HDF5 and CSV persistence for the single-orbit energy-envelope study."""

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

if TYPE_CHECKING:
    from studies.gc_energy_bound import GCEnergyBoundResult


def _json_value(value: Any) -> Any:
    """Serialize NumPy provenance without executable object pickles."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported metadata type: {type(value).__name__}")


def save_energy_bound_result(path: str | Path, result: "GCEnergyBoundResult") -> Path:
    """Save full aligned histories and the audited reference in a new artifact."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tables = {name: getattr(result, name) for name in ("summary", "envelopes", "blocks", "orders")}
    with h5py.File(destination, "x") as archive:
        archive.attrs["schema"] = "gc-energy-bound-v1"
        archive.attrs["metadata_json"] = json.dumps(result.metadata, default=_json_value)
        archive.attrs["tables_json"] = json.dumps(tables, default=_json_value)
        for name, values in result.arrays.items():
            archive.create_dataset(name, data=values, compression="gzip", shuffle=True)
    for name, rows in tables.items():
        if rows:
            columns = list(dict.fromkeys(key for row in rows for key in row))
            with destination.with_name(f"{destination.stem}_{name}.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
    return destination


def load_energy_bound_result(path: str | Path) -> "GCEnergyBoundResult":
    """Read a completed study for plotting, without recomputing trajectories."""
    from studies.gc_energy_bound import GCEnergyBoundResult

    arrays: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as archive:
        if archive.attrs.get("schema") != "gc-energy-bound-v1":
            raise ValueError("Unrecognized GC energy-bound artifact.")
        metadata = json.loads(archive.attrs["metadata_json"])
        tables = json.loads(archive.attrs["tables_json"])

        def collect(name: str, value: Any) -> None:
            if isinstance(value, h5py.Dataset):
                arrays[name] = np.asarray(value)

        archive.visititems(collect)
    return GCEnergyBoundResult(arrays, metadata, **tables)
