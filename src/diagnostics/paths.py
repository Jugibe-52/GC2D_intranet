"""Diagnostic output paths derived reproducibly from a project notebook."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import re
from typing import Literal


_BLOCK_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
DEFAULT_RESULTS_BUCKET = "gc2d_data:gc2d-notebooks-data"


def solution_destination(
	experiment_path: str | Path,
	run_id: str,
	*,
	storage: Literal["bucket", "local"] = "bucket",
	bucket_root: str = DEFAULT_RESULTS_BUCKET,
	project_root: str | Path | None = None,
) -> str:
	"""Locate a run using its experiment directory relative to ``notebooks/``.

	The default is the configured Backblaze data bucket. Choose ``storage="local"``
	explicitly to use ``outputs/``. For example, an experiment path of
	``developements/my_study`` preserves that hierarchy in either destination.
	"""
	path = Path(experiment_path)
	if (path.is_absolute() or len(path.parts) < 2 or ".." in path.parts
		or path.parts[0] == "notebooks" or ":" in str(path) or "\\" in str(path)):
		raise ValueError("Use an experiment directory relative to notebooks/, such as developements/my_study.")
	if not isinstance(run_id, str) or not _BLOCK_NAME.fullmatch(run_id):
		raise ValueError("run_id must contain only letters, numbers, '_' and '-'.")
	if storage == "bucket":
		if not re.fullmatch(r"[A-Za-z0-9_-]+:[A-Za-z0-9_.-]+", bucket_root):
			raise ValueError("bucket_root must have the form 'remote:bucket'.")
		return f"{bucket_root}/{path.as_posix()}/{run_id}"
	if storage == "local":
		root = find_project_root(Path.cwd()) if project_root is None else Path(project_root).expanduser().resolve()
		return str(root / "outputs" / path / run_id)
	raise ValueError("storage must be 'bucket' or 'local'.")


def find_project_root(start: str | Path) -> Path:
	"""Find the nearest ancestor containing the project configuration."""
	location = Path(start).expanduser().resolve()
	if location.is_file():
		location = location.parent
	for candidate in (location, *location.parents):
		if (candidate / "pyproject.toml").is_file():
			return candidate
	raise ValueError(f"Could not find the project root above {location}.")


def notebook_output_directory(
	notebook_path: str | Path,
	*,
	project_root: str | Path | None = None,
	run_date: date | str | None = None,
) -> Path:
	"""Map a notebook to ``outputs/<folder>/<notebook>/<date>``.

	For example, ``notebooks/developements/gc_symplecticity.ipynb`` maps to
	``outputs/developements/gc_symplecticity/2026-07-20``. The notebook must live
	under this project's ``notebooks`` tree so unrelated paths cannot write into
	the diagnostic output hierarchy accidentally.
	"""
	root = (
		find_project_root(Path.cwd())
		if project_root is None
		else Path(project_root).expanduser().resolve()
	)
	notebook = Path(notebook_path).expanduser()
	if not notebook.is_absolute():
		notebook = root / notebook
	notebook = notebook.resolve()
	try:
		relative = notebook.relative_to(root / "notebooks")
	except ValueError as exc:
		raise ValueError("The notebook must be located below the project notebooks directory.") from exc
	if relative.suffix != ".ipynb":
		raise ValueError("`notebook_path` must identify an .ipynb file.")

	if run_date is None:
		date_label = date.today().isoformat()
	elif isinstance(run_date, date):
		date_label = run_date.isoformat()
	else:
		date_label = date.fromisoformat(run_date).isoformat()
	return root / "outputs" / relative.parent / relative.stem / date_label


def validate_block_name(block_name: str) -> str:
	"""Reject names that could escape or ambiguously structure output files."""
	if not _BLOCK_NAME.fullmatch(block_name):
		raise ValueError("`block_name` may contain only letters, numbers, '_' and '-'.")
	return block_name


def next_block_index(output_directory: Path, block_name: str) -> int:
	"""Return the next unused five-digit index across all files in a block."""
	name = validate_block_name(block_name)
	indices: list[int] = []
	pattern = re.compile(rf"^{re.escape(name)}_[a-z]+_(\d{{5}})\.[^.]+$")
	if output_directory.exists():
		for path in output_directory.iterdir():
			match = pattern.match(path.name)
			if match is not None:
				indices.append(int(match.group(1)))
	return max(indices, default=-1) + 1


__all__ = ["DEFAULT_RESULTS_BUCKET", "find_project_root", "notebook_output_directory", "solution_destination"]
