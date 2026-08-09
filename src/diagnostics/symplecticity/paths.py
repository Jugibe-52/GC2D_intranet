"""Output paths derived reproducibly from a development notebook."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import re


_BLOCK_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


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


__all__ = ["find_project_root", "notebook_output_directory"]
