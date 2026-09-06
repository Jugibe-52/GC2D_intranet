# S54b SDIRK4 simulation architecture

## Public method

[`SDIRK4`](../../../../src/simulation/methods/classical/sdirk.py) is exported
from the public `simulation` package. It consumes `DynamicalSystem` directly
and has the same Newton controls and analytic/finite-difference Jacobian
selection convention as `GaussLegendre4`.

```python
from simulation import InitialValueProblem, SDIRK4, SimulationRequest, simulate

solution = simulate(
    problem,
    SDIRK4(
        newton_absolute_tolerance=1e-12,
        newton_relative_tolerance=1e-11,
        newton_max_iterations=40,
        newton_jacobian_method="analytic",
    ),
    SimulationRequest.uniform(
        t_span=(0.0, 100.0),
        max_step=0.1,
        sample_count=1001,
    ),
)
```

## Step lifecycle

The S54b tableau has five stages and common diagonal coefficient
`gamma = 1/4`. One complete step performs five sequential nonlinear solves.
At stage `i`, fields from stages `j < i` form the known right-hand side; Newton
then solves only for the current physical stage. With analytic planar
guiding-centre derivatives, the correction is vectorized across particles as
independent `2 x 2` systems. Generic dynamics use a dense centered-difference
Jacobian.

The accepted state uses the tableau weights. Because S54b is stiffly accurate,
the update is also the fifth stage to the configured nonlinear tolerance.
When `track_energy=True`, the time-conjugate momentum uses the same stage
states, nodes, and weights without enlarging the physical nonlinear systems.

## Fixed grid and observations

[`integrate_fixed_grid`](../../../../src/simulation/_fixed.py) keeps the main
integration grid independent of the output schedule. Main steps alone append
Newton diagnostics and notify the optional `step_observer`; shadow output
steps do neither. The generic `IntegrationStep.map_state` callback repeats the
same five-stage physical map for opt-in numerical differentiation.

## Diagnostics

The common one-dimensional arrays `nonlinear_iterations`,
`residual_evaluations`, `nonlinear_residual_norms`, and
`nonlinear_tolerances` contain one aggregate per complete step. Iterations and
evaluations are summed across the five stages; the residual norm is their
maximum. The corresponding `stage_*` arrays have shape `(step_count, 5)`.

Scalar metadata records the designed order, stage count, tableau name,
diagonal coefficient, stiff accuracy, and the deliberate absence of symmetry
and symplecticity. The exact tableau arrays are exported read-only as
`SDIRK4_TABLEAU_A`, `SDIRK4_TABLEAU_B`, and `SDIRK4_TABLEAU_C`.

## Verification

[`tests/test_sdirk4.py`](../../../../tests/test_sdirk4.py) checks all classical
order-four conditions, the failed symmetry and symplecticity identities,
observed fourth-order convergence, non-autonomous stage times, validation,
diagnostic shapes, and optional energy tracking.

[`tests/test_four_method_sdirk_comparison.py`](../../../../tests/test_four_method_sdirk_comparison.py)
checks the reusable four-method experiment contract and its aligned reference,
trajectory, energy, timing, and Newton results.
