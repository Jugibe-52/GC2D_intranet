"""Check plain SDD documents, preserved history, and the read-only CLI."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
import unittest

from scripts.check_sdd import CHANGE_SECTIONS, Checker


REPOSITORY = Path(__file__).resolve().parents[1]


def render_record(metadata: dict[str, Any], body: str) -> str:
	"""Serialize the small TOML fixture vocabulary without a writer dependency."""
	lines = ["+++"]
	tables: list[tuple[str, list[dict[str, Any]]]] = []
	for key, value in metadata.items():
		if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
			tables.append((key, value))
		else:
			lines.append(f"{key} = {json.dumps(value)}")
	for key, entries in tables:
		for entry in entries:
			lines.append(f"\n[[{key}]]")
			lines.extend(f"{name} = {json.dumps(value)}" for name, value in entry.items())
	return "\n".join(lines) + "\n+++\n\n" + body


class SDDTests(unittest.TestCase):
	"""Use isolated repositories to test invalid records without touching GC2D."""

	def setUp(self) -> None:
		self.directory = tempfile.TemporaryDirectory()
		self.addCleanup(self.directory.cleanup)
		self.root = Path(self.directory.name)
		for folder in ("specs/current", "specs/changes", "tests", "docs", "src"):
			(self.root / folder).mkdir(parents=True)
		(self.root / "src/sample.py").write_text("VALUE = 1\n", encoding="utf-8")
		(self.root / "docs/contract.md").write_text("# Existing contract\n", encoding="utf-8")
		# A static reference check must never import even a syntactically valid file.
		(self.root / "tests/test_sample.py").write_text(
			"raise RuntimeError('This file must not be imported')\n"
			"class ContractTests:\n\tdef test_contract(self):\n\t\tpass\n",
			encoding="utf-8",
		)
		self.spec: dict[str, Any] = {
			"schema_version": 1, "kind": "spec", "id": "sample",
			"sources": ["src/sample.py"], "documents": ["docs/contract.md"],
			"requirements": [{"id": "SAM-001", "tests": ["tests/test_sample.py::ContractTests.test_contract"]}],
		}
		self.change: dict[str, Any] = {
			"schema_version": 2, "kind": "change-spec", "id": "sample-change", "revision": 1,
			"status": "draft", "sources": ["src/sample.py"], "documents": ["docs/contract.md"],
			"requirements": ["SAM-001"], "proposed_requirements": [], "open_questions": [], "approval": "",
			"criteria": [{"id": "AC-001", "description": "Preserve the contract.", "requirements": ["SAM-001"]}],
			"tasks": [{"id": "T-001", "description": "Apply the requested change.", "depends_on": [], "done": False}],
			"checks": [{
				"id": "V-001", "criteria": ["AC-001"], "procedure": "Review the contract and run its test.",
				"tests": ["tests/test_sample.py::ContractTests.test_contract"],
				"result": "pending", "evidence": "", "checked_on": "",
			}],
		}
		self.body = "# Sample change\n\n" + "\n\n".join(
			f"## {section}\n\nAn explicit description of {section.lower()}." for sections in CHANGE_SECTIONS.values() for section in sections
		)
		self.plan: dict[str, Any] = {
			"schema_version": 2, "kind": "change-plan", "id": "sample-change",
			"revision": 1, "spec_revision": 1, "sources": [], "documents": [], "open_questions": [],
		}
		self.work: dict[str, Any] = {
			"schema_version": 2, "kind": "change-tasks", "id": "sample-change",
			"spec_revision": 1, "plan_revision": 1, "sources": [], "documents": [], "open_questions": [],
		}
		self.folder = self.root / "specs/changes/sample-change"
		self.folder.mkdir()
		self.save_spec()
		self.save_change()

	def save_spec(self) -> None:
		(self.root / "specs/current/sample.md").write_text(render_record(
			self.spec, "# Sample\n\n## SAM-001: Contract\n\nThe observable requirement.\n",
		), encoding="utf-8")

	def save_change(self) -> None:
		"""Distribute fixture data across its actual document owners."""
		proposal = {key: value for key, value in self.change.items() if key not in ("tasks", "checks")}
		work = dict(self.work, tasks=self.change["tasks"], checks=self.change["checks"])
		for name, data in (("spec", proposal), ("plan", self.plan), ("tasks", work)):
			(self.folder / f"{name}.md").write_text(render_record(data, self.body), encoding="utf-8")

	def errors(self) -> str:
		return "\n".join(Checker(self.root).run())

	def save_plain_change(self) -> Path:
		"""Prepare a new change alongside an older one in the same repository."""
		folder = self.root / "specs/changes/plain-change"
		folder.mkdir(exist_ok=True)
		contents = {
			"goal.md": "# Easier field selection\n\nUsers need to choose the input fields explicitly.\n",
			"proposal.md": "Allow an ordered list of field indices and preserve that order.\n",
			"impact.md": "The loader, its tests, and example notebooks need updates.\n",
			"steps.md": "- [ ] Accept a list of indices.\n- [ ] Check reordered and empty selections.\n",
		}
		for name, content in contents.items():
			(folder / name).write_text(content, encoding="utf-8")
		return folder

	def approve(self, status: str = "approved") -> None:
		self.change["status"] = status
		self.change["approval"] = "2026-10-13: Fixture user accepted the specification and plan."

	def finish(self) -> None:
		self.approve("verified")
		self.change["tasks"][0]["done"] = True
		self.change["checks"][0].update(
			result="passed", evidence="Fixture contract checked; expected result observed.", checked_on="2026-10-13",
		)

	def test_valid_lifecycle_and_manual_requirement_verification(self) -> None:
		self.assertEqual(self.errors(), "")
		for status in ("approved", "implementing"):
			self.approve(status)
			self.save_change()
			self.assertEqual(self.errors(), "")
		self.finish()
		self.save_change()
		self.assertEqual(self.errors(), "")
		self.spec["requirements"][0].update(tests=[], verification="Review the documented retirement and compatibility.")
		self.save_spec()
		self.assertEqual(self.errors(), "")

	def test_draft_allows_explicit_open_decisions_and_empty_work_arrays(self) -> None:
		self.change.update(criteria=[], tasks=[], checks=[], open_questions=["Choose the intended behavior."])
		self.body = "# Investigated draft\nA decision remains open.\n"
		self.save_change()
		self.assertEqual(self.errors(), "")

	def test_spanish_reading_copies_do_not_add_records_or_duplicate_ids(self) -> None:
		canonical = self.root / "specs/current/sample.md"
		for suffix in ("es.md", "es.2.md"):
			canonical.with_name(f"sample.{suffix}").write_bytes(canonical.read_bytes())
			for name in ("spec", "plan", "tasks"):
				(self.folder / f"{name}.{suffix}").write_text("# Reading copy without metadata\n", encoding="utf-8")
		checker = Checker(self.root)
		self.assertEqual(checker.run(), [])
		self.assertEqual(len(checker.records), 4)
		self.assertEqual(set(checker.requirements), {"SAM-001"})
		self.change["status"] = "invalid"
		self.save_change()
		self.assertIn("unknown status", self.errors())

	def test_spanish_copies_cannot_replace_missing_canonical_documents(self) -> None:
		canonical = self.root / "specs/current/sample.md"
		canonical.rename(canonical.with_name("sample.es.md"))
		(self.folder / "spec.md").rename(self.folder / "spec.es.2.md")
		errors = self.errors()
		self.assertIn("no current specifications found", errors)
		self.assertIn("change is missing spec.md", errors)

	def test_other_markdown_names_are_still_validated(self) -> None:
		for name in ("sample.es.notes.md", "sample.en.md"):
			with self.subTest(name=name):
				path = self.root / "specs/current" / name
				path.write_text("# Unexpected document without metadata\n", encoding="utf-8")
				self.assertIn(f"specs/current/{name}: cannot read record", self.errors())
				path.unlink()

	def test_approval_requires_plan_criteria_tasks_checks_and_resolved_questions(self) -> None:
		self.approve()
		self.change.update(approval="", open_questions=["Unresolved."], criteria=[], tasks=[], checks=[])
		self.body = "# Missing plan\n"
		self.save_change()
		errors = self.errors()
		for text in ("approval", "open questions", "must have criteria", "must have tasks", "must have checks", "missing or empty section: Design"):
			self.assertIn(text, errors)

	def test_new_requirements_must_be_current_before_verification(self) -> None:
		self.approve()
		self.change["requirements"] = ["SAM-002"]
		self.change["proposed_requirements"] = ["SAM-002"]
		self.change["criteria"][0]["requirements"] = ["SAM-002"]
		self.save_change()
		self.assertEqual(self.errors(), "")
		self.finish()
		self.save_change()
		self.assertIn("unknown current requirement SAM-002", self.errors())
		self.spec["requirements"][0]["id"] = "SAM-002"
		(self.root / "specs/current/sample.md").write_text(
			render_record(self.spec, "# Contract\n\n## SAM-002: New contract\n\nImplemented behavior.\n"), encoding="utf-8",
		)
		self.assertEqual(self.errors(), "")

	def test_revised_draft_cannot_retain_approval_of_old_scope(self) -> None:
		self.finish()
		self.change["status"] = "draft"
		self.save_change()
		self.assertIn("draft approval must be empty", self.errors())
		self.change["approval"] = ""
		self.change["checks"][0].update(result="pending", evidence="", checked_on="")
		self.save_change()
		self.assertEqual(self.errors(), "")

	def test_duplicate_requirements_documents_and_proposals_are_rejected(self) -> None:
		other = deepcopy(self.spec)
		other["id"] = "other"
		(self.root / "specs/current/other.md").write_text(
			render_record(other, "## SAM-001: Duplicate\n\nConflicting requirement.\n"), encoding="utf-8",
		)
		self.assertIn("duplicate requirement id", self.errors())
		(self.root / "specs/current/nested").mkdir()
		(self.root / "specs/current/nested/sample.md").write_bytes((self.root / "specs/current/sample.md").read_bytes())
		self.assertIn("duplicate document id", self.errors())
		self.change["requirements"].append("SAM-002")
		self.change["proposed_requirements"] = ["SAM-002"]
		self.save_change()
		other_change = {k: deepcopy(v) for k, v in self.change.items() if k not in ("tasks", "checks")}
		other_change["id"] = "other-change"
		(self.root / "specs/changes/other-change").mkdir()
		(self.root / "specs/changes/other-change/spec.md").write_text(render_record(other_change, self.body), encoding="utf-8")
		self.assertIn("proposed by multiple changes", self.errors())

	def test_unknown_requirement_and_criterion_references_are_rejected(self) -> None:
		self.change["requirements"] = ["SAM-999"]
		self.change["checks"][0]["criteria"] = ["AC-999"]
		self.save_change()
		errors = self.errors()
		self.assertIn("unknown current requirement SAM-999", errors)
		self.assertIn("requirement SAM-001 is not declared", errors)
		self.assertIn("unknown criterion AC-999", errors)

	def test_approved_traceability_cannot_omit_requirements_or_criteria(self) -> None:
		self.approve()
		self.change["criteria"][0]["requirements"] = []
		self.change["criteria"].append({"id": "AC-002", "description": "Additional behavior.", "requirements": []})
		self.save_change()
		self.assertIn("requirement SAM-001 has no acceptance criterion", self.errors())
		self.assertIn("criterion AC-002 has no verification check", self.errors())

	def test_verified_requires_finished_tasks_and_passing_checks(self) -> None:
		self.finish()
		self.change["tasks"][0]["done"] = False
		for result in ("pending", "failed", "blocked"):
			with self.subTest(result=result):
				self.change["checks"][0]["result"] = result
				self.save_change()
				self.assertIn("unfinished task T-001", self.errors())
				self.assertIn("non-passing check V-001", self.errors())

	def test_a_pass_requires_evidence_and_a_valid_date(self) -> None:
		self.finish()
		self.change["checks"][0].update(evidence="", checked_on="2026-02-30")
		self.save_change()
		self.assertIn("evidence must be a non-empty string", self.errors())
		self.assertIn("checked_on must be an ISO date", self.errors())

	def test_invalid_metadata_is_reported_without_tracebacks(self) -> None:
		original = deepcopy(self.change)
		cases: tuple[tuple[str, object, str], ...] = (
			("schema_version", True, "schema_version"),
			("status", "done", "unknown status"),
			("status", [], "status must be"),
			("kind", "spec", "kind must be"),
			("requirements", 7, "requirements must be"),
			("tasks", [True], "tasks must be"),
			("checks", "a command", "checks must be"),
			("approval", False, "approval must be"),
			("unexpected", True, "unknown fields"),
		)
		for key, value, expected in cases:
			with self.subTest(key=key, value=value):
				self.change = deepcopy(original)
				self.change[key] = value
				self.save_change()
				self.assertIn(expected, self.errors())

	def test_nested_metadata_checks_types_and_duplicate_ids(self) -> None:
		self.change["tasks"][0]["done"] = "true"
		self.change["criteria"].append(deepcopy(self.change["criteria"][0]))
		self.change["checks"][0].update(result="success", evidence=False, checked_on=False)
		self.save_change()
		errors = self.errors()
		for expected in ("done must be a Boolean", "duplicate criteria id", "invalid check result", "evidence must be a string", "checked_on must be a string"):
			self.assertIn(expected, errors)

	def test_missing_files_test_selectors_and_requirement_prose_are_rejected(self) -> None:
		self.spec["sources"] = ["src/missing.py"]
		self.spec["requirements"][0]["tests"] = ["tests/test_sample.py::ContractTests.test_missing"]
		self.save_spec()
		errors = self.errors()
		self.assertIn("missing or external file", errors)
		self.assertIn("missing test selector", errors)
		path = self.root / "specs/current/sample.md"
		path.write_text(path.read_text().replace("## SAM-001:", "## SAM-002:"), encoding="utf-8")
		self.assertIn("missing requirement heading", self.errors())

	def test_references_stay_inside_repository_and_tests_directory(self) -> None:
		self.change["documents"] = ["../outside.md", "/etc/passwd", "docs\\contract.md"]
		self.change["checks"][0]["tests"] = ["src/sample.py"]
		self.save_change()
		self.assertIn("repository-relative file", self.errors())
		self.assertIn("expected a Python file under tests/", self.errors())
		with tempfile.TemporaryDirectory() as outside:
			secret = Path(outside) / "external.md"
			secret.write_text("This must not be inspected.", encoding="utf-8")
			(self.root / "docs/external.md").symlink_to(secret)
			self.change["documents"] = ["docs/external.md"]
			self.save_change()
			self.assertIn("missing or external file", self.errors())

	def test_nested_function_is_not_a_class_method_test_selector(self) -> None:
		(self.root / "tests/test_sample.py").write_text(
			"def outer():\n\tdef test_nested():\n\t\tpass\n", encoding="utf-8",
		)
		self.spec["requirements"][0]["tests"] = ["tests/test_sample.py::outer.test_nested"]
		self.save_spec()
		self.assertIn("test selector must start with a class", self.errors())

	def test_missing_or_invalid_frontmatter_and_missing_baseline_fail(self) -> None:
		path = self.root / "specs/current/sample.md"
		for content in ("# No metadata", "+++\nnot toml\n+++\n# Contract", "+++\nmissing_end=true"):
			with self.subTest(content=content):
				path.write_text(content, encoding="utf-8")
				self.assertIn("cannot read record", self.errors())
		path.unlink()
		self.assertIn("no current specifications found", self.errors())
		(self.root / "specs/current").rmdir()
		self.assertIn("required directory is missing", self.errors())

	def test_templates_support_four_preparation_files_and_later_results(self) -> None:
		folder = self.root / "specs/changes/new-change"
		folder.mkdir()
		for name in ("goal", "proposal", "impact", "steps"):
			content = (REPOSITORY / f".agents/skills/sdd/assets/{name}.md").read_text(encoding="utf-8")
			(folder / f"{name}.md").write_text(content, encoding="utf-8")
		self.assertEqual(self.errors(), "")
		self.assertFalse((folder / "results.md").exists())
		content = (REPOSITORY / ".agents/skills/sdd/assets/results.md").read_text(encoding="utf-8")
		(folder / "results.md").write_text(content, encoding="utf-8")
		self.assertEqual(self.errors(), "")
		content = (REPOSITORY / ".agents/skills/sdd/assets/contract.md").read_text(encoding="utf-8")
		(self.root / "specs/current/new-contract.md").write_text(content, encoding="utf-8")
		self.assertEqual(self.errors(), "")

	def test_plain_preparation_coexists_with_history_without_fixed_headings(self) -> None:
		folder = self.save_plain_change()
		checker = Checker(self.root)
		self.assertEqual(checker.run(), [])
		self.assertEqual(len(checker.records), 8)
		self.assertFalse((folder / "results.md").exists())
		# The document checker must not misrepresent failed scientific work as done.
		(folder / "results.md").write_text(
			"The loader was updated. Reordered selection passed; empty selection failed.\n"
			"Implementation remains incomplete.\n", encoding="utf-8",
		)
		checker = Checker(self.root)
		self.assertEqual(checker.run(), [])
		self.assertEqual(len(checker.records), 9)

	def test_results_and_translations_cannot_replace_preparation_files(self) -> None:
		folder = self.save_plain_change()
		for name in ("goal", "proposal", "impact", "steps"):
			(folder / f"{name}.md").rename(folder / f"{name}.es.md")
		(folder / "results.md").write_text("Execution is blocked: the proposal is missing.\n", encoding="utf-8")
		errors = self.errors()
		for name in ("goal", "proposal", "impact", "steps"):
			self.assertIn(f"change is missing {name}.md", errors)

	def test_plain_documents_need_content_but_no_metadata(self) -> None:
		folder = self.save_plain_change()
		for content in ("", " \n", "# Only a title\n\n", "+++\nstatus='draft'\n+++\nText\n"):
			with self.subTest(content=content):
				(folder / "goal.md").write_text(content, encoding="utf-8")
				self.assertIn("goal.md: cannot read document", self.errors())

	def test_plain_reading_copies_are_ignored_and_formats_cannot_be_mixed(self) -> None:
		folder = self.save_plain_change()
		for name in ("goal", "proposal", "impact", "steps", "results"):
			for suffix in ("es.md", "es.2.md"):
				(folder / f"{name}.{suffix}").write_text("+++\nNot canonical metadata\n", encoding="utf-8")
		checker = Checker(self.root)
		self.assertEqual(checker.run(), [])
		self.assertEqual(len(checker.records), 8)
		(folder / "spec.md").write_bytes((self.folder / "spec.md").read_bytes())
		self.assertIn("cannot mix plain and legacy", self.errors())

	def test_plain_documents_reject_external_files_and_extra_documents(self) -> None:
		folder = self.save_plain_change()
		with tempfile.TemporaryDirectory() as outside:
			path = Path(outside) / "external.md"
			path.write_text("External content must not be inspected.\n", encoding="utf-8")
			(folder / "goal.md").unlink()
			(folder / "goal.md").symlink_to(path)
			self.assertIn("record must remain inside the repository", self.errors())
		(folder / "extra.md").write_text("Unexpected extra document.\n", encoding="utf-8")
		self.assertIn("extra.md: expected changes/", self.errors())

	def test_plain_cli_never_writes_or_executes_documented_commands(self) -> None:
		folder = self.save_plain_change()
		(folder / "results.md").write_text("Pending check: `touch should-not-exist`.\n", encoding="utf-8")
		before = {p.relative_to(self.root): (p.read_bytes(), p.stat().st_mtime_ns) for p in self.root.rglob("*") if p.is_file()}
		completed = subprocess.run(
			[sys.executable, "-B", str(REPOSITORY / "scripts/check_sdd.py"), "--root", str(self.root)],
			cwd=self.root, capture_output=True, text=True, check=False,
		)
		self.assertEqual(completed.returncode, 0, completed.stderr)
		after = {p.relative_to(self.root): (p.read_bytes(), p.stat().st_mtime_ns) for p in self.root.rglob("*") if p.is_file()}
		self.assertEqual(before, after)

	def test_draft_may_omit_downstream_phases_but_approval_requires_all_three(self) -> None:
		for name in ("tasks", "plan"):
			(self.folder / f"{name}.md").unlink()
			self.assertEqual(self.errors(), "")
		self.approve()
		self.save_change()
		(self.folder / "tasks.md").unlink()
		(self.folder / "plan.md").unlink()
		self.assertIn("requires plan.md", self.errors())
		self.assertIn("requires tasks.md", self.errors())

	def test_orphan_documents_and_legacy_layout_are_rejected(self) -> None:
		(self.folder / "plan.md").unlink()
		self.assertIn("tasks.md requires plan.md", self.errors())
		(self.folder / "spec.md").unlink()
		self.assertIn("change is missing spec.md", self.errors())
		(self.root / "specs/changes/legacy.md").write_text("# Old record", encoding="utf-8")
		self.assertIn("migrate legacy flat records", self.errors())

	def test_revisions_warn_in_draft_and_block_approval_until_reconciled(self) -> None:
		self.change["revision"] = 2
		self.save_change()
		checker = Checker(self.root)
		self.assertEqual(checker.run(), [])
		self.assertEqual(len(checker.warnings), 2)
		self.approve()
		self.save_change()
		self.assertIn("stale spec_revision", self.errors())
		self.plan.update(spec_revision=2, revision=2)
		self.work["spec_revision"] = 2
		self.save_change()
		self.assertIn("stale plan_revision", self.errors())
		self.work["plan_revision"] = 2
		self.save_change()
		self.assertEqual(self.errors(), "")
		for value in (True, 0, -1, "2"):
			self.plan["revision"] = value
			self.save_change()
			self.assertIn("revision must be a positive integer", self.errors())

	def test_each_part_checks_identity_fields_questions_and_references(self) -> None:
		self.approve()
		for part in (self.plan, self.work):
			original = deepcopy(part)
			part.update(id="another-change", open_questions=["Unresolved"], documents=["missing.md"], status="draft")
			self.save_change()
			errors = self.errors()
			for expected in ("must match the record id", "open questions", "missing or external file", "unknown fields"):
				self.assertIn(expected, errors)
			part.clear()
			part.update(original)
		self.save_change()
		self.assertEqual(self.errors(), "")

	def test_local_identifiers_require_their_domain_and_numeric_suffix(self) -> None:
		for table in ("criteria", "tasks", "checks"):
			original = self.change[table][0]["id"]
			for identifier in ("arbitrary text", "X-001", "AC-1"):
				self.change[table][0]["id"] = identifier
				self.save_change()
				self.assertIn(f"invalid {table} id", self.errors())
			self.change[table][0]["id"] = original
		self.save_change()
		self.assertEqual(self.errors(), "")

	def test_removed_criterion_requires_downstream_reconciliation_even_in_drafts(self) -> None:
		self.change["revision"] = 2
		self.change["criteria"][0]["id"] = "AC-002"
		self.save_change()
		checker = Checker(self.root)
		self.assertIn("unknown criterion AC-001", "\n".join(checker.run()))
		self.assertEqual(len(checker.warnings), 2)
		self.plan.update(revision=2, spec_revision=2)
		self.work.update(spec_revision=2, plan_revision=2)
		self.change["checks"][0]["criteria"] = ["AC-002"]
		self.save_change()
		self.assertEqual(self.errors(), "")

	def test_task_dependencies_require_order_and_completed_prerequisites(self) -> None:
		self.change["tasks"].append({"id": "T-002", "description": "Dependent step.", "depends_on": ["T-001"], "done": False})
		self.save_change()
		self.assertEqual(self.errors(), "")
		self.change["tasks"][1]["done"] = True
		self.save_change()
		self.assertIn("unfinished dependency T-001", self.errors())
		self.change["tasks"][0]["done"] = True
		self.save_change()
		self.assertEqual(self.errors(), "")
		for dependencies in (["T-002"], ["T-999"], ["T-001"]):
			self.change["tasks"][0]["depends_on"] = dependencies
			self.save_change()
			self.assertIn("must name an earlier task", self.errors())

	def test_cli_reports_exit_codes_and_never_writes_or_executes_procedures(self) -> None:
		self.change["checks"][0]["procedure"] = "touch should-not-exist"
		self.save_change()
		before = {p.relative_to(self.root): (p.read_bytes(), p.stat().st_mtime_ns) for p in self.root.rglob("*") if p.is_file()}
		command = [sys.executable, "-B", str(REPOSITORY / "scripts/check_sdd.py"), "--root", str(self.root)]
		completed = subprocess.run(command, cwd=self.root, capture_output=True, text=True, check=False)
		self.assertEqual(completed.returncode, 0, completed.stderr)
		self.assertIn("SDD validation passed", completed.stdout)
		after = {p.relative_to(self.root): (p.read_bytes(), p.stat().st_mtime_ns) for p in self.root.rglob("*") if p.is_file()}
		self.assertEqual(before, after)
		self.change["status"] = "invalid"
		self.save_change()
		completed = subprocess.run(command, cwd=self.root, capture_output=True, text=True, check=False)
		self.assertEqual(completed.returncode, 1)
		self.assertIn("specs/changes/sample-change/spec.md: unknown status", completed.stderr)
		self.assertNotIn("Traceback", completed.stderr)


if __name__ == "__main__":
	unittest.main()
