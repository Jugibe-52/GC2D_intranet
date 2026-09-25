"""Supported package layout and optional-dependency contracts."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

import simulation
import methods
import formulations
import integration
from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import ABBA2Midpoint, ABBA2Implicit, ABBA4Implicit, ABBA6Implicit, ABBA4_PROJECTION_PLACEMENTS, ABBA_PROJECTION_FORMULATIONS, ABBA_STATE_EXTENSIONS, BM4Implicit, BM4Midpoint, ExplicitEuler, RK4
from methods import extended, classical


class PackageLayoutTests(unittest.TestCase):
	"""Keep the supported source hierarchy and remove superseded namespaces."""

	def test_supported_packages_are_present(self) -> None:
		project_root = Path(__file__).resolve().parents[1]
		source_root = project_root / "src"
		for package in (
			"contracts",
			"diagnostics",
			"dynamics",
			"formulations",
			"initial_conditions",
			"integration",
			"methods",
			"potential",
			"simulation",
			"studies",
			"visualization",
		):
			self.assertTrue((source_root / package / "__init__.py").is_file())
		self.assertTrue((source_root / "solution.py").is_file())
		self.assertFalse((source_root / "simulation" / "methods").exists())
		self.assertFalse((source_root / "simulation" / "formulations").exists())
		for removed in ("gc2d", "classes", "research", "workflows"):
			self.assertFalse((source_root / removed).exists())

		self.assertIsNotNone(GuidingCenterDynamics)
		self.assertIsNotNone(GCInitialConfiguration)
		self.assertIsNotNone(Potential)
		self.assertIsNotNone(ABBA2Midpoint)
		self.assertIsNotNone(ABBA2Implicit)
		self.assertIsNotNone(ABBA4Implicit)
		self.assertIsNotNone(ABBA4Implicit)
		self.assertIsNotNone(ABBA6Implicit)
		self.assertIsNotNone(BM4Implicit)
		self.assertEqual(
			ABBA4_PROJECTION_PLACEMENTS,
			("around_complete_composition",),
		)
		self.assertEqual(
			ABBA_PROJECTION_FORMULATIONS,
			("reduced_multiplier", "simultaneous_state_multiplier"),
		)
		self.assertEqual(
			ABBA_STATE_EXTENSIONS,
			("physical",),
		)
		self.assertIsNotNone(RK4)
		self.assertIsNotNone(ExplicitEuler)
		for name in extended.__all__:
			self.assertIs(getattr(extended, name), getattr(simulation, name))
			self.assertIs(getattr(extended, name), getattr(methods, name))
		self.assertIs(classical.RK4, RK4)
		for module in (
			"contracts.configuration", "contracts.problem", "contracts.request",
			"contracts.observation", "contracts.result", "contracts.step",
			"formulations.gc", "formulations.fc", "formulations.state",
			"integration.core", "methods.extended", "methods.classical",
			"methods.adaptive", "methods.hbvm", "simulation.runner", "solution",
		):
			with self.subTest(module=module):
				self.assertIsNotNone(importlib.util.find_spec(module))

	def test_public_exports_are_explicit_and_resolve(self) -> None:
		for namespace in (simulation, methods, extended, classical, formulations, integration):
			with self.subTest(namespace=namespace.__name__):
				self.assertEqual(len(namespace.__all__), len(set(namespace.__all__)))
				for name in namespace.__all__:
					self.assertFalse(name.startswith("_"), name)
					self.assertTrue(hasattr(namespace, name), name)

	def test_internal_imports_do_not_reenter_the_public_simulation_facade(self) -> None:
		"""Keep the execution facade above its implementation dependencies."""
		source_root = Path(__file__).resolve().parents[1] / "src"
		for path in source_root.rglob("*.py"):
			for node in ast.walk(ast.parse(path.read_text())):
				if isinstance(node, ast.ImportFrom):
					self.assertNotEqual(node.module, "simulation", str(path))
					self.assertTrue(all(alias.name != "*" for alias in node.names), str(path))
				elif isinstance(node, ast.Import):
					self.assertTrue(all(alias.name != "simulation" for alias in node.names), str(path))

	def test_removed_namespaces_are_not_importable(self) -> None:
		for package in ("gc2d", "classes", "research", "workflows"):
			self.assertIsNone(importlib.util.find_spec(package), package)
		for module in (
			"simulation.methods", "simulation.formulations", "simulation.integration",
			"simulation.configuration", "simulation.problem", "simulation.request",
			"simulation.observation", "simulation.solution", "simulation._result",
			"simulation._fixed", "simulation._compat", "methods.abba", "methods.bm4",
			"methods._fully_extended", "methods._abba_coefficients",
			"methods.extended.order4_implicit_single_projection",
		):
			with self.subTest(module=module):
				with self.assertRaises(ModuleNotFoundError):
					importlib.import_module(module)
		for namespace in (
			simulation,
			methods,
			extended,
		):
			for name in (
				"SymmetricProjectedABBA",
				"MidpointABBA",
				"ImplicitABBA1",
				"ImplicitABBA2",
				"ABBA4Implicit1",
				"ABBA4SingleProjectionImplicit1",
				"ABBA6",
				"ABBA_implicit2",
				"ABBA4_implicit2",
				"ABBA2SharedTimeExtendedImplicit",
				"ABBA2FullyExtendedImplicit",
				"ABBA4FullyExtendedImplicit",
				"BM4Composition",
				"BM4Implicit1",
				"BM4Implicit2",
				"BM4_implicit2",
				"MidpointBM4",
				"ProjectedBM4Composition",
			):
				self.assertFalse(
					hasattr(namespace, name),
					f"{namespace.__name__}.{name}",
				)

	def test_retired_public_interfaces_are_absent(self) -> None:
		for package, names in {
			"simulation": ("SimulationRunner", "ABBA4ImplicitSingleProjection",
				"FullyExtendedImplicitIntegrationStep", "FullyExtendedBaseMap"),
			"initial_conditions": ("Trajectory", "TrajectoryGC", "TrajectoryFC"),
			"studies": ("run_fully_extended_implicit_study", "centered_gc_trajectory",
				"run_abba4_projection_comparison_study"),
		}.items():
			module = importlib.import_module(package)
			for name in names:
				with self.subTest(package=package, name=name):
					self.assertFalse(hasattr(module, name))

	def test_core_packages_do_not_require_matplotlib(self) -> None:
		project_root = Path(__file__).resolve().parents[1]
		script = """
import builtins

original_import = builtins.__import__

def import_without_matplotlib(name, *args, **kwargs):
    if name == "matplotlib" or name.startswith("matplotlib."):
        raise ModuleNotFoundError("matplotlib intentionally unavailable")
    return original_import(name, *args, **kwargs)

builtins.__import__ = import_without_matplotlib
import dynamics
import initial_conditions
import potential
import simulation
assert dynamics.GuidingCenterDynamics is not None
assert initial_conditions.GCInitialConfiguration is not None
assert potential.Potential is not None
assert simulation.RK4 is not None
assert simulation.ExplicitEuler is not None
"""
		completed = subprocess.run(
			[sys.executable, "-c", script],
			cwd=project_root,
			check=False,
			capture_output=True,
			text=True,
		)
		self.assertEqual(completed.returncode, 0, completed.stderr)

	def test_visualization_public_api_imports_in_a_clean_interpreter(self) -> None:
		"""Prevent study type imports from re-entering a partial public package."""
		project_root = Path(__file__).resolve().parents[1]
		completed = subprocess.run(
			[
				sys.executable,
				"-c",
				"from visualization import animate_gc_particle_solution; "
				"assert animate_gc_particle_solution is not None",
			],
			cwd=project_root,
			check=False,
			capture_output=True,
			text=True,
		)
		self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
	unittest.main()
