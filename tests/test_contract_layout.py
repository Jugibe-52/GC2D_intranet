"""Contract ownership, isolated imports, and standalone worker packaging."""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"


class ContractLayoutTests(unittest.TestCase):
	"""Require canonical domain contracts without loading their implementations."""

	def test_structural_contracts_are_defined_only_under_contracts(self) -> None:
		for path in SOURCE.rglob("*.py"):
			for node in ast.walk(ast.parse(path.read_text())):
				if not isinstance(node, ast.ClassDef):
					continue
				bases = {ast.unparse(base).split("[")[0] for base in node.bases}
				if bases.intersection({"Protocol", "TypedDict"}):
					self.assertTrue(path.is_relative_to(SOURCE / "contracts"), str(path))

	def test_all_contracts_import_without_implementations_or_optional_backends(self) -> None:
		modules = sorted(
			".".join(path.relative_to(SOURCE).with_suffix("").parts)
			for path in (SOURCE / "contracts").rglob("*.py")
			if path.name != "__init__.py"
		)
		script = """
import builtins, importlib, sys
original = builtins.__import__
blocked = {'dynamics', 'execution', 'formulations', 'methods', 'solvers', 'states',
           'studies', 'diagnostics', 'visualization', 'simulation', 'potential',
           'problem', 'request', 'jax', 'modal', 'matplotlib'}
def independent(name, *args, **kwargs):
    if name.split('.')[0] in blocked:
        raise ImportError('Contract imported implementation: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = independent
for name in sys.argv[1:]:
    importlib.import_module(name)
"""
		for order in (modules, list(reversed(modules))):
			completed = subprocess.run([sys.executable, "-c", script, *order],
				cwd=ROOT, capture_output=True, text=True)
			self.assertEqual(completed.returncode, 0, completed.stderr)

	def test_public_facades_export_canonical_contracts(self) -> None:
		for facade, module, names in (
			("dynamics", "contracts.dynamics.protocols", ("DynamicalSystem", "HamiltonianSystem")),
			("methods", "contracts.methods.base", ("NumericalMethod",)),
			("execution", "contracts.execution.controllers", ("StepController",)),
			("formulations", "contracts.formulations.base", ("DirectAdjointFormulation",)),
			("formulations", "contracts.formulations.projection", ("ProjectionFormulation",)),
			("simulation", "contracts.methods.observation", ("IntegrationStep", "IntegrationStage")),
			("simulation", "contracts.methods.extended.abba", ("ABBA4ImplicitIntegrationStep",)),
			("visualization", "contracts.visualization.trajectory_accuracy", ("AccuracySeriesView",)),
		):
			for name in names:
				with self.subTest(facade=facade, name=name):
					contract = getattr(importlib.import_module(module), name)
					self.assertIs(getattr(importlib.import_module(facade), name), contract)
					self.assertEqual(contract.__module__, module)

	def test_replaced_flat_modules_are_removed(self) -> None:
		for name in ("configuration", "compiled", "step", "nonlinear", "result",
			"observation", "comparison", "study_results", "_study_validation"):
			with self.assertRaises(ModuleNotFoundError):
				importlib.import_module("contracts." + name)
		with self.assertRaises(ModuleNotFoundError):
			importlib.import_module("dynamics.protocols")
		# The old execution.py route is now a domain package, without retired values.
		self.assertFalse(hasattr(importlib.import_module("contracts.execution"), "ExecutionOptions"))

	def test_cloud_archive_imports_contracts_without_installed_project(self) -> None:
		"""Exercise the real builder and extracted transfer script in a clean Python."""
		runtime = SOURCE / "studies" / "poincare_batch_runtime"
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory) / "experiment"
			(root / "aws").mkdir(parents=True)
			(root / "assets").mkdir()
			for name in ("calculo.ipynb", "visualizacion.ipynb", "run.py", "parallel_calculation.py",
				"checkpoint_calculation.py", "newton_diagnostics.py", "experiment.py", "poincare_batch.py",
				"poincare_selector.py", "_poincare_selector.html", "requirements.txt", "README.md",
				"validate_restart.py", "aws/validate_checkpoint_store.py", "aws/validation_restart.json",
				"aws/validation_cloud_store.json"):
				(root / name).write_text("")
			for name in ("build_package.py", "cloud_job.py"):
				shutil.copyfile(runtime / name, root / "aws" / name)
			shutil.copyfile(runtime / "study_io.py", root / "study_io.py")
			(root / "experiment.json").write_text(json.dumps({"run_id": "test_contracts",
				"bucket": "unused-offline-test", "deadline_hours": 1, "service": "test.service", "method": "RK4"}))
			built = subprocess.run([sys.executable, str(root / "aws/build_package.py")],
				cwd=ROOT, capture_output=True, text=True)
			self.assertEqual(built.returncode, 0, built.stderr)
			destination = Path(directory) / "extracted"
			with tarfile.open(root / "aws/experiment_aws.tar.gz") as archive:
				archive.extractall(destination, filter="data")
			bundle = destination / "experiment"
			self.assertEqual(
				{path.relative_to(bundle / "contracts") for path in (bundle / "contracts").rglob("*.py")},
				{path.relative_to(SOURCE / "contracts") for path in (SOURCE / "contracts").rglob("*.py")},
			)
			script = """
import runpy, sys
runpy.run_path(sys.argv[1], run_name='contract_import_test')
from contracts.studies.poincare_batch_runtime.cloud_job import S3Client
from contracts.studies.poincare_batch_runtime.newton_diagnostics import DiagnosticTableMetadata
assert S3Client.__module__.startswith('contracts.')
assert 'complete_steps' in DiagnosticTableMetadata.__annotations__
assert not any(name.split('.')[0] in {'numpy', 'pandas', 'jax', 'studies'} for name in sys.modules)
"""
			imported = subprocess.run([sys.executable, "-I", "-S", "-c", script,
				str(bundle / "aws/cloud_job.py")], cwd=bundle, capture_output=True, text=True)
			self.assertEqual(imported.returncode, 0, imported.stderr)
			# A scientific worker must see the same complete contracts package.
			scientific = subprocess.run([sys.executable, "-c",
				"import contracts; from simulation import RK4; "
				"from contracts.methods.base import NumericalMethod; "
				"assert isinstance(RK4(), NumericalMethod); "
				"assert 'extracted' in contracts.__file__"],
				cwd=bundle, capture_output=True, text=True)
			self.assertEqual(scientific.returncode, 0, scientific.stderr)


if __name__ == "__main__":
	unittest.main()
