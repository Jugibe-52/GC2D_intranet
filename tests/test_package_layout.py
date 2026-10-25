"""Supported package layout and optional-dependency contracts."""

from __future__ import annotations

import numpy as np

import ast
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

import simulation
import methods
import formulations
import execution
import diagnostics
from dynamics import GuidingCenterDynamics

from potential import Potential
from simulation import (
	ABBA2Midpoint, ABBA2Implicit, ABBA4Implicit, ABBA4_PROJECTION_PLACEMENTS,
	ABBA_PROJECTION_FORMULATIONS, ABBA_STATE_EXTENSIONS, BM4Implicit, BM4Midpoint, ExplicitEuler, RK4,
)
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
			"execution",
			"formulations",
			"states",
			"methods",
			"potential",
			"simulation",
			"studies",
			"visualization",
		):
			self.assertTrue((source_root / package / "__init__.py").is_file())
		self.assertTrue((source_root / "request.py").is_file())
		self.assertTrue((source_root / "problem.py").is_file())
		self.assertFalse((source_root / "simulation" / "methods").exists())
		self.assertFalse((source_root / "simulation" / "formulations").exists())
		for removed in ("gc2d", "classes", "research", "workflows", "integration"):
			self.assertFalse((source_root / removed).exists())

		self.assertIsNotNone(GuidingCenterDynamics)
		self.assertIsNotNone(np.ndarray)
		self.assertIsNotNone(Potential)
		self.assertIsNotNone(ABBA2Midpoint)
		self.assertIsNotNone(ABBA2Implicit)
		self.assertIsNotNone(ABBA4Implicit)
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
			"contracts.dynamics.coordinates", "problem", "request", "execution.options",
			"contracts.methods.observation", "contracts.states.history", "contracts.methods.step",
			"methods.extended.maps", "methods.splitting", "formulations.state",
			"execution._host", "methods.base", "methods.extended", "methods.classical",
			"methods.adaptive", "methods.hbvm", "execution.execution", "simulation.runner", "states.history", "execution.state", "dynamics.coordinates",
		):
			with self.subTest(module=module):
				self.assertIsNotNone(importlib.util.find_spec(module))

	def test_removed_abba6_interfaces_are_absent(self) -> None:
		import studies
		from contracts.methods.extended import abba as records
		for namespace, names in (
			(simulation, ('ABBA6Implicit', 'ABBA6ImplicitIntegrationStep')),
			(records, ('ABBA6ImplicitIntegrationStep',)),
			(diagnostics, ('abba6_implicit_step_particle_jacobians',)),
			(studies, ('ABBA6AccuracyConfig', 'ABBA6AccuracyResult', 'run_abba6_accuracy_study')),
		):
			for name in names:
				self.assertFalse(hasattr(namespace, name), (namespace.__name__, name))
		self.assertIsNone(importlib.util.find_spec('studies.abba6_accuracy'))

	def test_public_exports_are_explicit_and_resolve(self) -> None:
		for namespace in (simulation, methods, extended, classical, formulations, execution, diagnostics):
			with self.subTest(namespace=namespace.__name__):
				self.assertEqual(len(namespace.__all__), len(set(namespace.__all__)))
				for name in namespace.__all__:
					self.assertFalse(name.startswith("_"), name)
					self.assertTrue(hasattr(namespace, name), name)

	def test_diagnostic_public_names_resolve_to_process_implementations(self) -> None:
		"""Public observers and persistence keep one implementation after relocation."""
		for name, module in {
			"AdaptiveTrajectoryObserver": "methods.adaptive.trajectory",
			"ImplicitABBAIterationObserver": "methods.implicit_iterations",
			"ImplicitABBAJacobianObserver": "methods.extended.abba_jacobian.observer",
			"ImplicitABBAReversibilityObserver": "methods.extended.abba_reversibility",
			"GCGeneralizedEnergyObserver": "methods.extended.energy.observer",
			"GCTrajectorySymplecticityObserver": "methods.trajectory_symplecticity.observer",
			"load_solution": "io.persistence",
			"save_solution": "io.persistence",
			"load_reference_trajectory": "studies.reference_trajectory",
		}.items():
			with self.subTest(name=name):
				implementation = importlib.import_module("diagnostics." + module)
				self.assertIs(getattr(diagnostics, name), getattr(implementation, name))

	def test_diagnostics_follow_processes_without_retired_flat_modules(self) -> None:
		for module in (
			"execution.run_progress", "execution.reference_progress",
			"methods.projection.observer", "states.geometry",
			"studies.comparison", "studies.bm4_recurrence", "io.buffering", "io.storage",
		):
			with self.subTest(module=module):
				self.assertIsNotNone(importlib.util.find_spec("diagnostics." + module))
		for removed in (
			"adaptive_trajectory", "abba_jacobian", "abba_reversibility",
			"bm4_vectors", "bm4_energy_balance", "energy", "implicit_iterations",
			"jacobians", "symplecticity", "trajectory_symplecticity", "projection",
			"geometry", "run_progress", "reference_progress", "buffering", "output",
			"paths", "storage", "persistence", "comparison", "bm4_comparison",
			"bm4_recurrence", "dop853_energy_balance_npz", "five_method_comparison_csv",
			"gc_energy_bound", "parallel_bm4_recurrence_npz", "poincare_periodicity",
			"poincare_probe", "poincare_section", "reference_trajectory",
			"single_particle_recurrence",
		):
			with self.subTest(removed=removed):
				self.assertIsNone(importlib.util.find_spec("diagnostics." + removed))

	def test_numerical_processes_do_not_load_diagnostic_implementations(self) -> None:
		"""A core simulation must be importable even when diagnostics are unavailable."""
		script = """
import builtins
import sys
original_import = builtins.__import__
def without_diagnostics(name, *args, **kwargs):
    if name.split('.')[0] == 'diagnostics':
        raise ModuleNotFoundError('diagnostics intentionally unavailable')
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_diagnostics
import simulation
import solvers
import contracts.methods.observation
assert not any(name == 'diagnostics' or name.startswith('diagnostics.') for name in sys.modules)
"""
		completed = subprocess.run(
			[sys.executable, "-c", script],
			cwd=Path(__file__).resolve().parents[1],
			check=False, capture_output=True, text=True,
		)
		self.assertEqual(completed.returncode, 0, completed.stderr)

	def test_method_and_execution_import_orders_without_optional_backends(self) -> None:
		"""Preparation and controller imports must not create package cycles."""
		project_root = Path(__file__).resolve().parents[1]
		script = """
import builtins
import importlib
import pickle
import sys
import numpy as np

original_import = builtins.__import__
def without_optional_backends(name, *args, **kwargs):
    if name.split(".")[0] in ("jax", "modal"):
        raise ModuleNotFoundError("optional backend intentionally unavailable")
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_optional_backends
importlib.import_module(sys.argv[1])
from execution import Execution, FixedStepController, IntegrationState
from execution.options import ExecutionOptions
from methods import DOP853, IntegrationMethod, RK4
from potential import Potential
from dynamics import GuidingCenterDynamics
from problem import InitialValueProblem
from request import SimulationRequest

problem = InitialValueProblem(GuidingCenterDynamics(
    Potential.random(A=.01, M=2, nx=8, ny=8, seed=3), rho=.01),
    np.array([1., 1.1]))
request = SimulationRequest((0., .02), np.array([0., .005, .02]))
assert pickle.loads(pickle.dumps(ExecutionOptions())) == ExecutionOptions()
restored_problem = pickle.loads(pickle.dumps(problem))
np.testing.assert_array_equal(restored_problem.initial_state, problem.initial_state)
for method in (RK4(step=.01), DOP853(max_step=.01)):
    restored = pickle.loads(pickle.dumps(method))
    run = restored.new_run(problem.dynamics.vector_field, problem.initial_conditions, request,
        jacobian=problem.dynamics.particle_vector_field_jacobians, jacobian_structure='particle')
    assert isinstance(run, IntegrationMethod)
    result = Execution().run(problem, restored, request)
    np.testing.assert_array_equal(result.times, request.output_times)
assert isinstance(RK4().controller(), FixedStepController)
state = pickle.loads(pickle.dumps(IntegrationState(0., np.array([1., 2.]))))
assert not state.values.flags.writeable
assert not any(name.split(".")[0] in ("jax", "modal", "integration")
               for name in sys.modules)
"""
		for first in ("problem", "execution.options", "methods", "execution", "methods.base", "execution.state", "simulation"):
			with self.subTest(first=first):
				completed = subprocess.run(
					[sys.executable, "-c", script, first], cwd=project_root,
					check=False, capture_output=True, text=True,
				)
				self.assertEqual(completed.returncode, 0, completed.stderr)

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
		for package in ("gc2d", "classes", "research", "workflows", "integration"):
			self.assertIsNone(importlib.util.find_spec(package), package)
		for module in (
			"contracts.problem", "contracts.execution_options",
			"execution._fixed", "execution._jax_run", "dynamics._jax_host",
			"integration.core", "integration.state", "integration.jax_fixed",
			"simulation.methods", "simulation.formulations", "simulation.integration",
			"simulation.configuration", "simulation.problem", "simulation.request",
			"simulation.observation", "simulation.solution", "simulation._result",
			"simulation._fixed", "simulation._compat", "methods.abba", "methods.bm4",
			"methods._fully_extended", "methods._abba_coefficients",
			"methods.extended.abba_single_projection",
			"methods.extended.order2_implicit", "methods.extended.order4_implicit",
			"methods.extended.order6_implicit", "methods.extended.order2_midpoint",
			"methods.extended.bm4_midpoint", "methods.extended.coefficients",
			"methods.extended.abba_steps", "methods.extended.abba_composition",
			"methods.extended.bm4_composition", "methods.extended.abba_maps",
			"methods.extended.abba_outer", "methods.extended.abba_reduced",
			"methods.extended.abba_simultaneous",
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
				"ABBA6Implicit",
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
			"simulation": ("SingleFormulation", "SimulationRunner", "ABBA4ImplicitSingleProjection",
				"FullyExtendedImplicitIntegrationStep", "FullyExtendedBaseMap"),
			"states": ("Trajectory", "TrajectoryGC", "TrajectoryFC"),
			"studies": ("run_fully_extended_implicit_study", "centered_gc_trajectory",
				"run_abba4_projection_comparison_study"),
		}.items():
			module = importlib.import_module(package)
			for name in names:
				with self.subTest(package=package, name=name):
					self.assertFalse(hasattr(module, name))

	def test_methods_and_formulations_keep_function_and_workspace_boundaries(self) -> None:
		root = Path(__file__).resolve().parents[1] / 'src'
		for package in ('methods', 'formulations'):
			for path in (root / package).rglob('*.py'):
				for node in ast.walk(ast.parse(path.read_text())):
					if isinstance(node, ast.ImportFrom):
						module = node.module or ''
						# Fixed coordinate types carry array conventions, never physical equations.
						self.assertFalse((module.startswith('dynamics.') and module != 'dynamics.coordinates')
							or module == 'contracts.dynamics.protocols', str(path))
						if package == 'methods':
							self.assertFalse(module.startswith('diagnostics'), str(path))
						else:
							self.assertNotIn(module.split('.')[0], ('execution', 'methods', 'solvers'), str(path))
					if isinstance(node, ast.Attribute):
						self.assertNotEqual(node.attr, '__self__', str(path))
		self.assertFalse(hasattr(formulations, 'SingleFormulation'))

	def test_core_packages_do_not_require_matplotlib(self) -> None:
		project_root = Path(__file__).resolve().parents[1]
		script = """
import builtins
import sys

original_import = builtins.__import__

def import_without_matplotlib(name, *args, **kwargs):
    if name == "matplotlib" or name.startswith("matplotlib."):
        raise ModuleNotFoundError("matplotlib intentionally unavailable")
    return original_import(name, *args, **kwargs)

builtins.__import__ = import_without_matplotlib
import dynamics
import states
import potential
import simulation
import studies
assert dynamics.GuidingCenterDynamics is not None
assert states.States is not None
assert potential.Potential is not None
assert simulation.RK4 is not None
assert simulation.ExplicitEuler is not None
assert studies.run_area_comparison is not None
assert studies.run_abba_symplecticity_study is not None
assert studies.run_generalized_energy_comparison is not None
assert not any(name == "visualization" or name.startswith("visualization.") for name in sys.modules)
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
				"import visualization; "
				"assert len(visualization.__all__) == len(set(visualization.__all__)); "
				"assert all(hasattr(visualization, name) for name in visualization.__all__)",
			],
			cwd=project_root,
			check=False,
			capture_output=True,
			text=True,
		)
		self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
	unittest.main()
