"""Flattened package layout and optional-dependency contracts."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import RK4


class PackageLayoutTests(unittest.TestCase):
	"""Keep public packages flat and remove every superseded namespace."""

	def test_flattened_packages_are_the_only_source_roots(self) -> None:
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
		self.assertIsNotNone(RK4)

	def test_removed_namespaces_are_not_importable(self) -> None:
		for package in ("gc2d", "classes", "research", "workflows"):
			self.assertIsNone(importlib.util.find_spec(package), package)

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
"""
		completed = subprocess.run(
			[sys.executable, "-c", script],
			cwd=project_root,
			check=False,
			capture_output=True,
			text=True,
		)
		self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
	unittest.main()
