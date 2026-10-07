"""Versioned, pickle-free Solution archives on disk or configured buckets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import tempfile
from typing import Any

import numpy as np
import scipy

from contracts.configuration import InitialConfiguration
from contracts.result import DiagnosticValue
from contracts.state_layout import FCStateLayout, GCStateLayout
from diagnostics.paths import solution_destination
from diagnostics.storage import ArtifactStore, StorageError
from initial_conditions.fc import FCInitialConfiguration
from initial_conditions.gc import GCInitialConfiguration
from potential.load import GC2DH5Metadata
from potential.grid import Grid
from potential.potential import Potential
from solution import Solution


@dataclass(frozen=True)
class StoredSolution:
    """Loaded trajectory, experiment metadata and optional sampled potential.

    ``potential`` is exactly the field passed to ``save_solution``. Store the
    base field and record ``rho`` in experiment metadata when using GC dynamics;
    the reader can then reconstruct the effective field without double averaging.
    """

    solution: Solution
    metadata: dict[str, Any]
    potential: Potential | None


def _json_default(value: object) -> object:
    """Convert numerical metadata explicitly, refusing arbitrary Python objects."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Unsupported metadata type: {type(value).__name__}.")


def _write_json(path: Path, record: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(record, default=_json_default, allow_nan=False,
                               indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_archive(
    directory: Path, solution: Solution, metadata: Mapping[str, Any],
    potential: Potential | None,
) -> None:
    """Encode physical layout, diagnostics and the actual sampled potential."""
    layout = solution.layout
    if type(layout) is GCStateLayout:
        layout_name = "gc_component_major_xy"
    elif type(layout) is FCStateLayout:
        layout_name = "fc_component_major_xyvxvy"
    else:
        raise TypeError("Persistence currently supports the canonical GC and FC layouts.")
    arrays = {"t": solution.t, "states": solution.states}
    initial_state = solution.initial_state
    if initial_state is not None:
        arrays["initial_state"] = initial_state
    diagnostics: dict[str, Any] = {}
    for index, (name, value) in enumerate(solution.diagnostics.items()):
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise TypeError("Object arrays cannot be persisted.")
            key = f"diagnostic_{index}"
            arrays[key] = value
            diagnostics[name] = {"array": key}
        elif isinstance(value, (float, int, str, bool, np.generic)):
            diagnostics[name] = {"scalar": value}
        else:
            raise TypeError(f"Unsupported diagnostic value: {name}.")
    np.savez_compressed(directory / "solution.npz", **arrays)
    potential_record = None
    if potential is not None:
        potential_arrays = {"mean": potential.mean, "modes": potential.modes,
                            "frequencies": potential.frequencies}
        provenance = potential.metadata
        # Preserve typed HDF5 provenance as well as the actual processed arrays.
        if isinstance(provenance, GC2DH5Metadata):
            attribute_arrays = {}
            for index, (name, value) in enumerate(provenance.attributes.items()):
                if value.dtype.hasobject:
                    raise TypeError("HDF5 provenance attributes must be pickle-free arrays.")
                key = f"attribute_{index}"
                attribute_arrays[name] = key
                potential_arrays[key] = value
            provenance = {name: getattr(provenance, name)
                          for name in provenance.__dataclass_fields__ if name != "attributes"}
            provenance["attribute_arrays"] = attribute_arrays
            provenance_kind = "gc2d_h5"
        else:
            provenance_kind = "json"
        potential_record = {
            "grid": asdict(potential.grid),
            "interpolation_order": potential.interpolation_order,
            "provenance_kind": provenance_kind, "provenance": provenance,
        }
        np.savez_compressed(directory / "potential.npz", **potential_arrays)
    _write_json(directory / "metadata.json", {
        "schema_version": 1, "artifact_kind": "gc2d_solution",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "layout": layout_name, "diagnostics": diagnostics,
        "has_initial_state": initial_state is not None,
        "potential": potential_record, "experiment": dict(metadata),
        "software": {"python": platform.python_version(),
                     "numpy": np.__version__, "scipy": scipy.__version__},
    })
    files = ["solution.npz", "metadata.json"]
    if potential is not None:
        files.append("potential.npz")
    _write_json(directory / "manifest.json", {
        "schema_version": 1, "artifact_kind": "gc2d_solution",
        "files": {name: {"sha256": _digest(directory / name),
                         "bytes": (directory / name).stat().st_size} for name in files},
    })


def save_solution(
    solution: Solution, destination: str | Path | None = None, *,
    metadata: Mapping[str, Any], potential: Potential | None = None,
    experiment_path: str | Path | None = None, run_id: str | None = None,
) -> str:
    """Save to the default bucket, or to an explicitly supplied destination.

    Without ``destination``, supply the experiment directory relative to
    ``notebooks/`` and its ``run_id``. Explicit destinations may be local or
    remote. Use ``solution_destination(..., storage="local")`` for local paths.
    The manifest is published last. Existing completed runs are never replaced.
    Failed remote transfers retain a complete local archive for manual retry.
    Concurrent writers must use distinct run prefixes.
    """
    if destination is None:
        if experiment_path is None or run_id is None:
            raise ValueError("Default bucket storage requires experiment_path and run_id.")
        destination = solution_destination(experiment_path, run_id)
    elif experiment_path is not None or run_id is not None:
        raise ValueError("Supply either destination or experiment_path and run_id.")
    store = ArtifactStore(destination)
    staging = Path(tempfile.mkdtemp(prefix="gc2d-solution-"))
    preserve = False
    try:
        _write_archive(staging, solution, metadata, potential)
        try:
            if store.remote:
                if store.exists("manifest.json"):
                    raise FileExistsError("The remote run already contains a manifest; choose a new run.")
            else:
                Path(store.location).mkdir(parents=True, exist_ok=False)
            for path in sorted(staging.iterdir()):
                if path.name != "manifest.json":
                    store.put(path, path.name)
            store.put(staging / "manifest.json", "manifest.json")
        except FileExistsError:
            raise
        except (OSError, StorageError) as exc:
            preserve = True
            raise StorageError(f"Publication failed. Complete local archive retained at {staging}.") from exc
    finally:
        if not preserve:
            shutil.rmtree(staging)
    return store.location


def _manifest_inventory(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a solution manifest and return its fixed artifact inventory."""
    if manifest.get("schema_version") != 1 or manifest.get("artifact_kind") != "gc2d_solution":
        raise ValueError("Unsupported solution manifest.")
    records = manifest.get("files", {})
    required = {"solution.npz", "metadata.json"}
    if set(records) not in (required, required | {"potential.npz"}):
        raise ValueError("Incomplete or unsupported manifest file inventory.")
    return dict(records)


def _load_solution_arrays(directory: Path, description: Mapping[str, Any]) -> Solution:
    """Validate solution metadata and reconstruct its canonical state layout."""
    if description.get("schema_version") != 1 or description.get("artifact_kind") != "gc2d_solution":
        raise ValueError("Unsupported solution metadata.")
    with np.load(directory / "solution.npz", allow_pickle=False) as saved:
        initial = saved["initial_state"] if description["has_initial_state"] else None
        configuration: InitialConfiguration
        if description["layout"] == "gc_component_major_xy":
            configuration = GCInitialConfiguration(initial)
        elif description["layout"] == "fc_component_major_xyvxvy":
            configuration = FCInitialConfiguration(initial)
        else:
            raise ValueError("Unsupported saved state layout.")
        diagnostics: dict[str, DiagnosticValue] = {
            name: saved[value["array"]] if "array" in value else value["scalar"]
            for name, value in description["diagnostics"].items()
        }
        solution = Solution(t=saved["t"], states=saved["states"],
                            source=configuration, diagnostics=diagnostics)
    return solution


def load_solution(source: str | Path) -> StoredSolution:
    """Fetch, verify and reconstruct a complete result without running a solver.

    Every remote load downloads into a fresh temporary directory; it cannot
    silently use a stale local result. Only fixed schema filenames are accepted.
    """
    store = ArtifactStore(source)
    with tempfile.TemporaryDirectory(prefix="gc2d-load-") as temporary:
        directory = Path(temporary)
        store.get("manifest.json", directory / "manifest.json")
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        records = _manifest_inventory(manifest)
        for name, record in records.items():
            path = directory / name
            store.get(name, path)
            if path.stat().st_size != record["bytes"] or _digest(path) != record["sha256"]:
                raise ValueError(f"Artifact integrity check failed: {name}.")
        description = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        solution = _load_solution_arrays(directory, description)
        potential = None
        field = description["potential"]
        if (field is not None) != ("potential.npz" in records):
            raise ValueError("Potential metadata and manifest disagree.")
        if field is not None:
            provenance = field["provenance"]
            with np.load(directory / "potential.npz", allow_pickle=False) as saved:
                if field["provenance_kind"] == "gc2d_h5":
                    if provenance["source_path"] is not None:
                        provenance["source_path"] = Path(provenance["source_path"])
                    provenance["attributes"] = {
                        name: saved[key] for name, key in provenance.pop("attribute_arrays").items()
                    }
                    provenance = GC2DH5Metadata(**provenance)
                elif field["provenance_kind"] != "json":
                    raise ValueError("Unsupported potential provenance.")
                potential = Potential(Grid(**field["grid"]), mean=saved["mean"],
                                      modes=saved["modes"], frequencies=saved["frequencies"],
                                      interpolation_order=field["interpolation_order"],
                                      metadata=provenance)
        return StoredSolution(solution, description["experiment"], potential)


__all__ = ["StoredSolution", "save_solution", "load_solution"]
