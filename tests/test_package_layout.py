"""Supported package layout and optional-dependency contracts."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

import simulation
from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import ExplicitEuler, ImplicitABBA1, RK4
from simulation.methods import abba as abba_methods
from simulation.methods import bm4 as bm4_methods
from simulation.methods import classical as classical_methods


class PackageLayoutTests(unittest.TestCase):
	"""Keep the supported source hierarchy and remove superseded namespaces."""

	def test_supported_packages_are_present(self) -> None:
		project_root = Path(__file__).resolve().parents[1]
		source_root = project_root / "src"
		for package in (
			"diagnostics",
			"dynamics",
			"initial_conditions",
			"potential",
			"simulation",
			"studies",
			"visualization",
		):
			self.assertTrue((source_root / package / "__init__.py").is_file())
		for removed in ("gc2d", "classes", "research", "workflows"):
			self.assertFalse((source_root / removed).exists())

		self.assertIsNotNone(GuidingCenterDynamics)
		self.assertIsNotNone(GCInitialConfiguration)
		self.assertIsNotNone(Potential)
		self.assertIsNotNone(ImplicitABBA1)
		self.assertIsNotNone(RK4)
		self.assertIsNotNone(ExplicitEuler)
		self.assertIs(abba_methods.ImplicitABBA1, ImplicitABBA1)
		self.assertIs(bm4_methods.BM4Implicit1, simulation.BM4Implicit1)
		self.assertIs(classical_methods.RK4, RK4)
		for module in (
			"simulation.methods.abba",
			"simulation.methods.abba._core",
			"simulation.methods.abba._projection",
			"simulation.methods.abba._implicit",
			"simulation.methods.abba.midpoint",
			"simulation.methods.abba.implicit_1",
			"simulation.methods.abba.implicit_2",
			"simulation.methods.abba.order4_implicit_1",
			"simulation.methods.abba.order4_single_projection_implicit_1",
			"simulation.methods.abba.order6",
			"simulation.methods.abba.fully_extended",
			"simulation.methods.bm4",
			"simulation.methods.bm4._core",
			"simulation.methods.bm4._implicit",
			"simulation.methods.bm4.midpoint",
			"simulation.methods.bm4.implicit_1",
			"simulation.methods.bm4.implicit_2",
			"simulation.methods.bm4.fully_extended",
			"simulation.methods.classical",
			"simulation.methods.classical.euler",
			"simulation.methods.classical.rk4",
		):
			self.assertIsNotNone(importlib.util.find_spec(module), module)

	def test_removed_namespaces_are_not_importable(self) -> None:
		for package in ("gc2d", "classes", "research", "workflows"):
			self.assertIsNone(importlib.util.find_spec(package), package)
		for module in (
			"simulation.methods._implicit_abba",
			"simulation.methods._projected_abba",
			"simulation.methods.abba_midpoint",
			"simulation.methods.abba_implicit_1",
			"simulation.methods.abba_implicit_2",
			"simulation.methods.abba4_implicit_1",
			"simulation.methods.abba4_single_projection_implicit_1",
			"simulation.methods.abba6",
			"simulation.methods.abba_tangent_taylor",
			"simulation.methods.abba.tangent_taylor",
			"simulation.methods.bm4_midpoint",
			"simulation.methods.bm4_implicit_1",
			"simulation.methods.bm4_implicit_2",
			"simulation.methods._implicit_bm4",
			"simulation.methods._fully_extended_implicit",
			"simulation.methods.euler",
			"simulation.methods.rk4",
		):
			self.assertIsNone(importlib.util.find_spec(module), module)
		self.assertFalse(hasattr(simulation, "SymmetricProjectedABBA"))

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
