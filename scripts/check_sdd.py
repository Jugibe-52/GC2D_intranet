#!/usr/bin/env python3
"""Check SDD records without importing project code or executing recorded checks.

Only canonical Markdown records under specs/current and specs/changes are scanned;
local Spanish reading copies are excluded by filename. New changes use plain
Markdown; current specifications and older changes retain their existing checks.
Scientific correctness and the authenticity of approvals/evidence require review.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
import re
import sys
import tomllib
from typing import Any


STATES = ("draft", "approved", "implementing", "verified")
RESULTS = ("pending", "passed", "failed", "blocked")
REQUIREMENT_ID = re.compile(r"[A-Z][A-Z0-9]*-[0-9]{3,}")
SPANISH_COPY = re.compile(r".+\.es(?:\.[0-9]+)?\.md")
CHANGE_SECTIONS = {
	"spec": ("Problem", "Scope", "Proposed behavior"),
	"plan": ("Design", "Compatibility", "Verification plan"),
	"tasks": ("Execution", "Outcome and limitations"),
}
COMMON_FIELDS = {"schema_version", "kind", "id", "sources", "documents"}
PREPARATION_FILES = {"goal.md", "proposal.md", "impact.md", "steps.md"}
PLAIN_CHANGE_FILES = PREPARATION_FILES | {"results.md"}
LEGACY_CHANGE_FILES = {"spec.md", "plan.md", "tasks.md"}
# Historical records keep the paths that existed when their work was accepted.
# Resolve these retired sources only for change history, never current specs.
HISTORICAL_SOURCE_MOVES = {
	"src/methods/_jax_dispatch.py": "src/execution/_jax_step.py",
	"src/formulations/layout.py": "src/dynamics/coordinates.py",
	"src/contracts/formulations/layout.py": "src/contracts/dynamics/coordinates.py",
}


@dataclass(frozen=True)
class Record:
	"""One parsed document, retaining its path for actionable diagnostics."""

	path: Path
	metadata: dict[str, Any]
	body: str


class Checker:
	"""Collect independent format and reference errors in one read-only pass."""

	def __init__(self, root: Path) -> None:
		self.root = root.resolve()
		self.errors: list[str] = []
		self.warnings: list[str] = []
		self.records: list[Record] = []
		self.requirements: dict[str, Path] = {}

	def error(self, path: Path, message: str) -> None:
		self.errors.append(f"{path.relative_to(self.root)}: {message}")

	def keys(self, record: Record, data: dict[str, Any], allowed: set[str], context: str) -> None:
		unknown = data.keys() - allowed
		if unknown:
			self.error(record.path, f"{context}: unknown fields {', '.join(sorted(unknown))}")

	def string(self, record: Record, data: dict[str, Any], key: str) -> str:
		value = data.get(key)
		if not isinstance(value, str) or not value.strip():
			self.error(record.path, f"{key} must be a non-empty string")
			return ""
		return value

	def strings(self, record: Record, data: dict[str, Any], key: str) -> list[str]:
		value = data.get(key)
		if not isinstance(value, list) or any(not isinstance(v, str) or not v.strip() for v in value):
			self.error(record.path, f"{key} must be an array of non-empty strings")
			return []
		if len(set(value)) != len(value):
			self.error(record.path, f"{key} contains duplicate entries")
		return value

	def tables(self, record: Record, key: str, allowed: set[str]) -> list[dict[str, Any]]:
		value = record.metadata.get(key)
		if not isinstance(value, list) or any(not isinstance(v, dict) for v in value):
			self.error(record.path, f"{key} must be an array of tables")
			return []
		identifiers: set[str] = set()
		for entry in value:
			self.keys(record, entry, allowed, key)
			identifier = self.string(record, entry, "id")
			prefix = {"criteria": "AC", "tasks": "T", "checks": "V"}.get(key)
			if prefix and not re.fullmatch(rf"{prefix}-[0-9]{{3,}}", identifier):
				self.error(record.path, f"invalid {key} id {identifier!r}; expected {prefix}-001 format")
			if identifier in identifiers:
				self.error(record.path, f"duplicate {key} id {identifier!r}")
			identifiers.add(identifier)
		return value

	def references(self, record: Record, data: dict[str, Any], key: str) -> None:
		"""Check repository-local files and optional static unittest selectors."""
		for reference in self.strings(record, data, key):
			filename, separator, selector = reference.partition("::")
			relative = PurePosixPath(filename)
			if not filename or relative.is_absolute() or ".." in relative.parts or "\\" in filename:
				self.error(record.path, f"{key}: expected a repository-relative file: {reference!r}")
				continue
			path = self.root / filename
			if (key == "sources" and not path.exists()
				and record.path.is_relative_to(self.root / "specs/changes")
				and filename in HISTORICAL_SOURCE_MOVES):
				path = self.root / HISTORICAL_SOURCE_MOVES[filename]
			try:
				if not path.resolve().is_relative_to(self.root) or not path.is_file():
					self.error(record.path, f"{key}: missing or external file {reference!r}")
					continue
				if key == "tests" and (relative.parts[0] != "tests" or path.suffix != ".py"):
					self.error(record.path, f"tests: expected a Python file under tests/: {reference!r}")
				if separator:
					if key != "tests" or not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?", selector):
						self.error(record.path, f"invalid test selector {reference!r}")
						continue
					nodes = ast.parse(path.read_text(encoding="utf-8")).body
					# Follow definitions lexically, never importing or running a test file.
					parts = selector.split(".")
					for index, name in enumerate(parts):
						found = next((node for node in nodes if isinstance(
							node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
						) and node.name == name), None)
						if found is None:
							self.error(record.path, f"missing test selector {reference!r}")
							break
						if index == 0 and len(parts) == 2 and not isinstance(found, ast.ClassDef):
							self.error(record.path, f"test selector must start with a class: {reference!r}")
							break
						nodes = found.body
					else:
						if not isinstance(found, (ast.FunctionDef, ast.AsyncFunctionDef)) or not found.name.startswith("test_"):
							self.error(record.path, f"selector is not a test function: {reference!r}")
			except (OSError, ValueError, SyntaxError, RuntimeError) as error:
				self.error(record.path, f"cannot inspect {reference!r}: {error}")

	def read(self, path: Path, kind: str) -> Record | None:
		try:
			if not path.resolve().is_relative_to(self.root):
				raise ValueError("record must remain inside the repository")
			lines = path.read_text(encoding="utf-8").splitlines()
			if not lines or lines[0] != "+++":
				raise ValueError("expected TOML frontmatter delimited by +++")
			end = lines.index("+++", 1)
			metadata = tomllib.loads("\n".join(lines[1:end]))
			record = Record(path, metadata, "\n".join(lines[end + 1:]))
		except (OSError, ValueError, RuntimeError) as error:
			self.error(path, f"cannot read record: {error}")
			return None
		version = 1 if kind == "spec" else 2
		if type(metadata.get("schema_version")) is not int or metadata["schema_version"] != version:
			self.error(path, f"schema_version must be integer {version}")
		if metadata.get("kind") != kind:
			self.error(path, f"kind must be {kind!r} in this directory")
		identifier = self.string(record, metadata, "id")
		if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", identifier):
			self.error(path, "id must use lowercase hyphen-separated words or numbers")
		if (path.stem if kind == "spec" else path.parent.name) != identifier:
			self.error(path, "current filename or change directory must match the record id")
		if not record.body.strip():
			self.error(path, "Markdown body must not be empty")
		for key in ("sources", "documents"):
			self.references(record, metadata, key)
		self.records.append(record)
		return record

	def specification(self, record: Record) -> None:
		self.keys(record, record.metadata, COMMON_FIELDS | {"requirements"}, "specification")
		entries = self.tables(record, "requirements", {"id", "tests", "verification"})
		if not entries:
			self.error(record.path, "a current specification must define requirements")
		for requirement in entries:
			identifier = self.string(record, requirement, "id")
			if not REQUIREMENT_ID.fullmatch(identifier):
				self.error(record.path, f"invalid requirement id {identifier!r}")
			if identifier in self.requirements:
				self.error(record.path, f"duplicate requirement id {identifier!r}")
			self.requirements[identifier] = record.path
			if not re.search(rf"^## {re.escape(identifier)}: \S", record.body, re.MULTILINE):
				self.error(record.path, f"missing requirement heading '## {identifier}: ...'")
			self.references(record, requirement, "tests")
			if "verification" in requirement:
				self.string(record, requirement, "verification")
			if not requirement.get("tests") and not requirement.get("verification"):
				self.error(record.path, f"{identifier}: provide tests or a manual verification procedure")

	def plain_change(self, folder: Path, paths: list[Path]) -> None:
		"""Check readable preparation documents without interpreting their prose."""
		names = {path.name for path in paths}
		for name in sorted(PREPARATION_FILES - names):
			self.error(folder, f"change is missing {name}")
		for path in paths:
			try:
				if not path.resolve().is_relative_to(self.root):
					raise ValueError("record must remain inside the repository")
				body = path.read_text(encoding="utf-8")
				lines = body.strip().splitlines()
				if lines and lines[0] in ("+++", "---"):
					raise ValueError("use plain Markdown without a metadata block")
				if not any(line.strip() and not line.lstrip().startswith("#") for line in lines):
					raise ValueError("document must contain text beyond a title")
			except (OSError, ValueError, RuntimeError) as error:
				self.error(path, f"cannot read document: {error}")
				continue
			self.records.append(Record(path, {}, body))

	def revision(self, record: Record, key: str) -> int:
		"""Read an explicit document revision; bool is not an integer revision."""
		value = record.metadata.get(key)
		if type(value) is not int or value < 1:
			self.error(record.path, f"{key} must be a positive integer")
			return 0
		return value

	def alignment(self, record: Record, key: str, expected: int, status: str) -> None:
		actual = self.revision(record, key)
		if actual and expected and actual != expected:
			message = f"stale {key}: records {actual}, current revision is {expected}"
			if status == "draft":
				self.warnings.append(f"{record.path.relative_to(self.root)}: {message}")
			else:
				self.error(record.path, message)

	def change(self, parts: dict[str, Record]) -> None:
		"""Validate one progressive proposal and its cross-document traceability."""
		if "spec" not in parts:
			for record in parts.values():
				self.error(record.path, "change is missing spec.md")
			return
		record = parts["spec"]
		data = record.metadata
		self.keys(record, data, COMMON_FIELDS | {
			"revision", "status", "requirements", "proposed_requirements", "open_questions",
			"approval", "criteria",
		}, "change specification")
		status = self.string(record, data, "status")
		if status not in STATES:
			self.error(record.path, f"unknown status {status!r}; expected {', '.join(STATES)}")
		spec_revision = self.revision(record, "revision")
		requirements = self.strings(record, data, "requirements")
		proposed = self.strings(record, data, "proposed_requirements")
		for identifier in requirements + proposed:
			if not REQUIREMENT_ID.fullmatch(identifier):
				self.error(record.path, f"invalid requirement id {identifier!r}")
		for identifier in proposed:
			if identifier not in requirements:
				self.error(record.path, f"proposed requirement {identifier} must appear in requirements")
		for identifier in requirements:
			if identifier not in self.requirements and (identifier not in proposed or status == "verified"):
				self.error(record.path, f"unknown current requirement {identifier}")
		approval = data.get("approval")
		if not isinstance(approval, str):
			self.error(record.path, "approval must be a string (empty until user acceptance)")
		elif status == "draft" and approval.strip():
			self.error(record.path, "draft approval must be empty; preserve previous acceptance in the narrative")
		if status in STATES[1:]:
			self.string(record, data, "approval")
			for name in ("plan", "tasks"):
				if name not in parts:
					self.error(record.path, f"approved or later change requires {name}.md")
		for name, part in parts.items():
			questions = self.strings(part, part.metadata, "open_questions")
			if status in STATES[1:]:
				if questions:
					self.error(part.path, "an approved or later change cannot have open questions")
				for section in CHANGE_SECTIONS[name]:
					match = re.search(rf"^## {re.escape(section)}\n(.*?)(?=^## |\Z)", part.body, re.MULTILINE | re.DOTALL)
					if match is None or not match.group(1).strip():
						self.error(part.path, f"missing or empty section: {section}")

		criteria = self.tables(record, "criteria", {"id", "description", "requirements"})
		criterion_ids: set[str] = set()
		covered_requirements: set[str] = set()
		for criterion in criteria:
			identifier = self.string(record, criterion, "id")
			criterion_ids.add(identifier)
			self.string(record, criterion, "description")
			for reference in self.strings(record, criterion, "requirements"):
				if reference not in requirements:
					self.error(record.path, f"{identifier}: requirement {reference} is not declared by this change")
				covered_requirements.add(reference)
		if status in STATES[1:]:
			if not criteria:
				self.error(record.path, "approved or later change must have criteria")
			for identifier in sorted(set(requirements) - covered_requirements):
				self.error(record.path, f"requirement {identifier} has no acceptance criterion")

		plan_revision = 0
		if "plan" in parts:
			plan = parts["plan"]
			self.keys(plan, plan.metadata, COMMON_FIELDS | {
				"revision", "spec_revision", "open_questions",
			}, "change plan")
			plan_revision = self.revision(plan, "revision")
			self.alignment(plan, "spec_revision", spec_revision, status)
		if "tasks" in parts:
			work = parts["tasks"]
			self.keys(work, work.metadata, COMMON_FIELDS | {
				"spec_revision", "plan_revision", "open_questions", "tasks", "checks",
			}, "change tasks")
			if "plan" not in parts:
				self.error(work.path, "tasks.md requires plan.md")
			self.alignment(work, "spec_revision", spec_revision, status)
			self.alignment(work, "plan_revision", plan_revision, status)
			self.work(work, status, criterion_ids)

	def work(self, record: Record, status: str, criterion_ids: set[str]) -> None:
		"""Check task ordering and criterion evidence in the execution document."""
		tasks = self.tables(record, "tasks", {"id", "description", "depends_on", "done"})
		checks = self.tables(record, "checks", {"id", "criteria", "procedure", "tests", "result", "evidence", "checked_on"})
		preceding: dict[str, bool] = {}
		for task in tasks:
			identifier = self.string(record, task, "id")
			self.string(record, task, "description")
			if type(task.get("done")) is not bool:
				self.error(record.path, "task done must be a Boolean")
			for dependency in self.strings(record, task, "depends_on"):
				# Requiring earlier dependencies also excludes missing IDs and cycles.
				if dependency not in preceding:
					self.error(record.path, f"{identifier}: dependency {dependency} must name an earlier task")
				elif task.get("done") is True and not preceding[dependency]:
					self.error(record.path, f"{identifier}: completed task has unfinished dependency {dependency}")
			preceding[identifier] = task.get("done") is True
			if status == "verified" and task.get("done") is not True:
				self.error(record.path, f"verified change has unfinished task {identifier}")
		covered_criteria: set[str] = set()
		for check in checks:
			identifier = self.string(record, check, "id")
			self.string(record, check, "procedure")
			self.references(record, check, "tests")
			targets = self.strings(record, check, "criteria")
			if not targets:
				self.error(record.path, f"{identifier}: check must cover at least one criterion")
			for target in targets:
				if target not in criterion_ids:
					self.error(record.path, f"{identifier}: unknown criterion {target}")
				covered_criteria.add(target)
			result = self.string(record, check, "result")
			if result not in RESULTS:
				self.error(record.path, f"{identifier}: invalid check result {result!r}")
			for key in ("evidence", "checked_on"):
				if not isinstance(check.get(key), str):
					self.error(record.path, f"{identifier}: {key} must be a string")
			if result in ("passed", "failed", "blocked"):
				self.string(record, check, "evidence")
				try:
					date.fromisoformat(self.string(record, check, "checked_on"))
				except ValueError:
					self.error(record.path, f"{identifier}: checked_on must be an ISO date")
			if status == "verified" and result != "passed":
				self.error(record.path, f"verified change has non-passing check {identifier}")
		if status in STATES[1:]:
			for name, entries in (("tasks", tasks), ("checks", checks)):
				if not entries:
					self.error(record.path, f"approved or later change must have {name}")
			for identifier in sorted(criterion_ids - covered_criteria):
				self.error(record.path, f"criterion {identifier} has no verification check")

	def run(self) -> list[str]:
		"""Check plain changes alongside existing specifications and history."""
		identifiers: set[str] = set()
		proposals: set[str] = set()
		for directory in ("current", "changes"):
			folder = self.root / "specs" / directory
			if not folder.is_dir():
				self.error(folder, "required directory is missing")
				continue
			# Reading copies may duplicate IDs or translate required headings; only
			# the canonical records participate in traceability and lifecycle checks.
			paths = sorted(path for path in folder.rglob("*.md") if not SPANISH_COPY.fullmatch(path.name))
			if directory == "current" and not paths:
				self.error(folder, "no current specifications found")
			groups: dict[str, dict[str, Record]] = {}
			plain_groups: dict[Path, list[Path]] = {}
			for path in paths:
				if directory == "changes":
					relative = path.relative_to(folder)
					if len(relative.parts) != 2 or path.name not in PLAIN_CHANGE_FILES | LEGACY_CHANGE_FILES:
						self.error(path, "expected changes/<change>/{goal,proposal,impact,steps,results}.md or an existing three-file record; migrate legacy flat records")
						continue
					if path.name in PLAIN_CHANGE_FILES:
						plain_groups.setdefault(path.parent, []).append(path)
						continue
				record = self.read(path, "spec" if directory == "current" else f"change-{path.stem}")
				if record is None:
					continue
				if directory == "current":
					identifier = self.string(record, record.metadata, "id")
					if identifier in identifiers:
						self.error(path, f"duplicate document id {identifier!r}")
					identifiers.add(identifier)
					self.specification(record)
				else:
					groups.setdefault(path.parent.name, {})[path.stem] = record
			for identifier, parts in groups.items():
				self.change(parts)
				if "spec" not in parts:
					continue
				record = parts["spec"]
				if identifier in identifiers:
					self.error(record.path, f"duplicate document id {identifier!r}")
				identifiers.add(identifier)
				for requirement in self.strings(record, record.metadata, "proposed_requirements"):
					if requirement in proposals:
						self.error(record.path, f"requirement {requirement} is proposed by multiple changes")
					proposals.add(requirement)
			for change_folder, plain_paths in plain_groups.items():
				identifier = change_folder.name
				if identifier in groups:
					self.error(change_folder, "cannot mix plain and legacy change documents in one folder")
				elif identifier in identifiers:
					self.error(change_folder, f"duplicate document id {identifier!r}")
				identifiers.add(identifier)
				self.plain_change(change_folder, plain_paths)
		return self.errors


def main(argv: list[str] | None = None) -> int:
	"""Return 0 for valid records, 1 for validation errors, or argparse's 2."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Repository root (defaults to this script's repository)")
	args = parser.parse_args(argv)
	checker = Checker(args.root)
	errors = checker.run()
	for warning in checker.warnings:
		print(f"SDD warning: {warning}", file=sys.stderr)
	if errors:
		for error in errors:
			print(error, file=sys.stderr)
		print(f"SDD validation failed: {len(errors)} error(s).", file=sys.stderr)
		return 1
	print(f"SDD validation passed: {len(checker.records)} records, {len(checker.requirements)} requirements.")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
