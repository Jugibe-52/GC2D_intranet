"""Run a short symmetric-projected ABBA trajectory with exact field Jacobians."""

from states import InitialConditions
from solvers import NewtonSolver
from formulations import ReducedProjection


from dynamics import GuidingCenterDynamics

from potential import Potential
from simulation import (
	InitialValueProblem,
	ABBA2Implicit,
	SimulationRequest,
	simulate,
)


potential = Potential.random(
	A=0.08,
	M=3,
	nx=16,
	ny=16,
	seed=27,
	interpolation_order=5,
)
configuration = InitialConditions.gc(x=[1.0], y=[1.2])
problem = InitialValueProblem(GuidingCenterDynamics(potential, rho=0.05), configuration)
request = SimulationRequest.uniform(
	t_span=(0.0, 0.2),
	max_step=0.05,
	sample_count=5,
)
solution = simulate(
	problem,
	ABBA2Implicit(projection_formulation=ReducedProjection(), solver=NewtonSolver()),
	request,
)

print(f"steps: {solution.diagnostics['step_count']}")
print(f"formulation: {solution.diagnostics['projection_formulation']}")
print(f"Newton iterations: {solution.diagnostics['newton_iterations']}")
print(f"final state: {solution.states[:, -1]}")
