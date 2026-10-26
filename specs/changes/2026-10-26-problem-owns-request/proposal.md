# Compose the problem from its existing objects

Restore request.py with SimulationRequest, its validators and defaults unchanged.
InitialValueProblem gains a validated request keyword and a request attribute.
Omitting the keyword creates SimulationRequest.uniform(), preserving existing
physical constructors with a complete default time configuration. A with_request
method returns a problem with the same dynamics and initial-condition owner and
a different request; it never mutates the source problem.

The public call becomes simulate(problem, method, *, execution=None, options=None,
observers=()). Execution.run and Execution_Modal.run similarly obtain the request
from the problem. Methods still receive functions, InitialConditions and
SimulationRequest separately. No distributions, equations or numerical controls
move into the problem.

For example: request = SimulationRequest.uniform(t_span=(0.0, 10.0));
problem = InitialValueProblem(dynamics, initial_state, request=request);
solution = simulate(problem, method). Existing simulation facade exports remain.
The former separate request argument is migrated in supported consumers.
