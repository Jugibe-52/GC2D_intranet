# Place the request in problem.py

Move SimulationRequest, its validation helpers and existing default constants
from src/request.py to src/problem.py. Keep InitialValueProblem and
SimulationRequest as separate classes. InitialValueProblem still references a
dynamics object and InitialConditions; their implementations remain in dynamics/
and states/initial.py. Keep simulate(problem, method, request) unchanged.

Internal imports become `from problem import SimulationRequest`. Public imports
from simulation remain supported. Remove request.py without a compatibility
alias. Preserve all validation rules, default values and numerical behavior.
Keep step controls on methods; relocating their existing default constants does
not change that ownership. No new package or request attribute is introduced.
