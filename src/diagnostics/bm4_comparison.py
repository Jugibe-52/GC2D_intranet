"""Portable NPZ persistence of BM4 comparison arrays and provenance."""

import json
from pathlib import Path

import numpy as np


def save_bm4_comparison(
	path: str | Path, arrays: dict[str, np.ndarray], metadata: dict[str, object],
) -> None:
	"""Save every trajectory and multiplier without Python-object pickles."""
	target = Path(path)
	target.parent.mkdir(parents=True, exist_ok=True)
	np.savez_compressed(target, **arrays, metadata_json=np.asarray(json.dumps(metadata, indent=2)))


def load_bm4_comparison(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
	"""Read a self-contained comparison without importing study orchestration."""
	with np.load(path, allow_pickle=False) as data:
		metadata = json.loads(str(data["metadata_json"]))
		arrays = {name: data[name].copy() for name in data.files if name != "metadata_json"}
	if not isinstance(metadata, dict) or "times" not in arrays:
		raise ValueError("Invalid BM4 comparison archive.")
	return arrays, metadata
