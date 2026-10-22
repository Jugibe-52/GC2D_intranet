"""Local manifests and resumable case execution for the fixed Poincare study."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from studies._validation import require_uncoupled_bm4
from diagnostics.io.persistence import load_solution, save_solution, StoredSolution
from execution.execution import Execution
from studies.poincare_automation import (
	AutomationConfig, AutomationCase, AutomationField, PreparedAutomationCase, PARTICLE_SET,
	canonical_json, load_automation_field, prepare_automation_case, run_automation_case,
	validate_saved_automation, automation_execution_context,
)


def case_key(config: AutomationConfig, case: AutomationCase) -> str:
	"""Include all resolved controls so closely spaced parameters never alias."""
	return sha256(canonical_json(dict(config=config.to_dict(), case=asdict(case),
		particle_set=PARTICLE_SET)).encode()).hexdigest()


def _write_json(path: Path, record: Any) -> None:
	"""Publish a whole JSON record atomically in its destination directory."""
	temporary: Path = path.with_name(path.name + ".tmp")
	temporary.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
	temporary.replace(path)


@contextmanager
def _run_lock(directory: Path) -> Iterator[None]:
	"""OS releases the local writer lock even if a process is interrupted."""
	with (directory / ".run.lock").open("a+") as lock:
		try:
			fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
		except BlockingIOError as exc:
			raise RuntimeError("This study already has an active writer.") from exc
		try:
			yield
		finally:
			fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def prepare_automation(config: AutomationConfig, project_root: Path,
		destination: Path | None = None) -> Path:
	"""Save resolved cases without opening the source or running any integrator."""
	require_uncoupled_bm4(config.coupling_frequency)
	root: Path = project_root.resolve()
	if destination is None:
		stamp: str = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
		destination = root / "notebooks/developements/poincare_section" / f"automation-{stamp}-{uuid4().hex[:8]}"
	directory: Path = destination.expanduser().resolve()
	expected: dict[str, object] = dict(schema_version=1, particle_set=PARTICLE_SET,
		project_root=str(root), configuration=config.to_dict())
	if directory.exists():
		if not (directory / "configuration.json").is_file():
			raise FileExistsError("Destination exists without a matching automation configuration.")
		if json.loads((directory / "configuration.json").read_text()) != expected:
			raise FileExistsError("Destination belongs to different inputs; choose another directory.")
		read_automation(directory)
		return directory
	directory.mkdir(parents=True, exist_ok=False)
	_write_json(directory / "configuration.json", expected)
	_write_json(directory / "cases.json", dict(schema_version=1, source_sha256=None, cases={
		case_key(config, case): dict(case=asdict(case), status="requested", error="")
		for case in config.cases()}))
	return directory


def read_automation(directory: Path) -> tuple[AutomationConfig, Path, dict[str, Any]]:
	"""Read without creating files; reject altered case inventories and unsafe records."""
	record: Any = json.loads((directory / "configuration.json").read_text())
	if record.get("schema_version") != 1 or record.get("particle_set") != PARTICLE_SET:
		raise ValueError("Unsupported automation configuration.")
	config: AutomationConfig = AutomationConfig(**record["configuration"])
	manifest: Any = json.loads((directory / "cases.json").read_text())
	if manifest.get("schema_version") != 1:
		raise ValueError("Unsupported automation manifest.")
	expected = {case_key(config, c): asdict(c) for c in config.cases()}
	if set(manifest.get("cases", {})) != set(expected):
		raise ValueError("The case inventory does not match the configuration.")
	for key, value in manifest["cases"].items():
		if value.get("case") != expected[key] or value.get("status") not in ("requested", "running", "failed", "complete"):
			raise ValueError("Invalid case parameters or status.")
	return config, Path(record["project_root"]), dict(manifest)


def verified_automation_field(config: AutomationConfig, root: Path,
		manifest: dict[str, Any]) -> AutomationField:
	field: AutomationField = load_automation_field(config, root)
	if manifest.get("source_sha256") not in (None, field.source_sha256):
		raise ValueError("The source file changed since this study started; choose a new study.")
	return field


def _archive_path(directory: Path, key: str, prepared: PreparedAutomationCase) -> Path:
	return directory / "cases" / key / str(prepared.metadata["input_digest"])


def load_completed_case(directory: Path, prepared: PreparedAutomationCase) -> StoredSolution:
	path: Path = _archive_path(directory, case_key(prepared.config, prepared.case), prepared)
	stored: StoredSolution = load_solution(path)
	validate_saved_automation(stored, prepared)
	return stored


def run_automation(directory: Path, *, execution: Execution | None = None) -> dict[str, Any]:
	"""Finish missing cases, preserving completed cases and reporting each failure.

	A caller may supply an explicitly configured existing executor. The command
	line uses local execution only; it never provisions or silently falls back.
	"""
	with _run_lock(directory):
		config: AutomationConfig
		root: Path
		manifest: dict[str, Any]
		config, root, manifest = read_automation(directory)
		require_uncoupled_bm4(config.coupling_frequency)
		with automation_execution_context(config, execution):
			pass  # Resolve local optional backend/device before loading fields or submitting cases.
		field: AutomationField = verified_automation_field(config, root, manifest)
		manifest["source_sha256"] = field.source_sha256
		_write_json(directory / "cases.json", manifest)
		for case in config.cases():
			key: str = case_key(config, case)
			record = manifest["cases"][key]
			prepared: PreparedAutomationCase = prepare_automation_case(config, case, field)
			path: Path = _archive_path(directory, key, prepared)
			print(f"{case.method}: rho={case.rho:g}, B={case.B:g}", flush=True)
			try:
				if (path / "manifest.json").is_file():
					load_completed_case(directory, prepared)
					record.update(status="complete", error="")
				elif record["status"] == "complete":
					raise ValueError("A completed case lost its archive; restore it or use a new study.")
				else:
					if path.exists():
						# Preserve partial artifacts for inspection before retrying this case.
						path.rename(path.with_name(path.name + ".incomplete-" + uuid4().hex[:8]))
					record.update(status="running", error="")
					_write_json(directory / "cases.json", manifest)
					stored: StoredSolution = run_automation_case(prepared, execution=execution)
					save_solution(stored.solution, path, metadata=stored.metadata)
					load_completed_case(directory, prepared)
					record.update(status="complete", error="")
			except Exception as exc:
				record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
				print(record["error"], flush=True)
			_write_json(directory / "cases.json", manifest)
		return manifest


__all__ = ["case_key", "prepare_automation", "read_automation", "verified_automation_field",
	"load_completed_case", "run_automation"]
