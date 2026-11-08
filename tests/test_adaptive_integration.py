"""Adaptive solvers belong to independent study references, not the method API."""
import importlib.util
import unittest
from unittest.mock import patch

import numpy as np
from scipy.integrate import solve_ivp

from studies.reference_trajectory import _solve_adaptive
from tests.test_method_integration import _problem


class ReferenceSolverBoundaryTests(unittest.TestCase):
	def test_adaptive_method_api_is_retired(self):
		import methods
		import simulation
		for module in (methods, simulation):
			for name in ('DOP853','Radau'):
				self.assertFalse(hasattr(module,name))
		self.assertIsNone(importlib.util.find_spec('methods.adaptive'))
		self.assertIsNone(importlib.util.find_spec('execution._scipy'))

	def test_direct_reference_matches_scipy_on_irregular_samples_and_reports_progress(self):
		problem = _problem()
		times = np.array([.3,.301,.33,.34,.4])
		for method in ('DOP853','Radau'):
			events=[]
			with patch('execution.Execution.run',side_effect=AssertionError('Reference used simulation runtime')):
				states, summary = _solve_adaptive(problem.dynamics,problem.initial_state,times,
					relative_tolerance=1e-11,absolute_tolerance=1e-13,maximum_step=.02,
					method=method,progress_callback=lambda *args:events.append(args))
			reference = solve_ivp(problem.dynamics.vector_field,(times[0],times[-1]),problem.initial_state,
				method=method,t_eval=times,rtol=1e-11,atol=1e-13,max_step=.02)
			np.testing.assert_array_equal(states,reference.y)
			self.assertEqual(summary.function_evaluations,reference.nfev)
			self.assertEqual(summary.jacobian_evaluations,reference.njev)
			self.assertEqual(summary.lu_decompositions,reference.nlu)
			self.assertEqual(events[0][-1],'started')
			self.assertEqual(events[-1][-1],'finished')
			self.assertEqual(events[-1][1],times[-1])

	def test_reference_callback_failure_propagates(self):
		problem=_problem()
		def fail(*args):
			raise RuntimeError('Reference cancelled')
		with self.assertRaisesRegex(RuntimeError,'cancelled'):
			_solve_adaptive(problem.dynamics,problem.initial_state,np.array([.3,.4]),
				relative_tolerance=1e-11,absolute_tolerance=1e-13,maximum_step=.02,
				method='DOP853',progress_callback=fail)
