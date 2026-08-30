# HBVM(4,2) numerical method and evaluation architecture

## Method definition

`simulation.HBVM42` is the Hamiltonian Boundary Value Method with four
Gauss--Legendre quadrature nodes and a degree-two polynomial path. Let
`c_i, b_i`, `i=1,...,4`, be Gauss--Legendre nodes and weights on `[0,1]`, and
let the first two orthonormal shifted Legendre polynomials be

\[
P_0(c)=1, \qquad P_1(c)=\sqrt{3}(2c-1).
\]

Define

\[
I_{i0}=c_i, \qquad I_{i1}=\sqrt{3}(c_i^2-c_i).
\]

The implementation solves for only two vector coefficients
`gamma = (gamma_0, gamma_1)`:

\[
Y_i=y_n+h\sum_{j=0}^{1} I_{ij}\gamma_j,
\]

\[
R_j(\gamma)=\gamma_j-
\sum_{i=1}^{4}b_iP_j(c_i)f(t_n+c_i h,Y_i)=0.
\]

The accepted Runge--Kutta update is

\[
y_{n+1}=y_n+h\sum_{i=1}^{4}b_i f(t_n+c_i h,Y_i).
\]

Equivalently, the Runge--Kutta matrix is

\[
A=\mathcal I_2\mathcal P_2^T\Omega,
\]

and has rank two. The nonlinear unknown therefore has size `2 * state_size`,
not `4 * state_size`.

## Numerical properties

- Classical order: `2s = 4`.
- Quadrature degree: seven, from four Gauss nodes.
- Polynomial energy conservation: an autonomous polynomial Hamiltonian of
  degree `nu` is conserved when `k >= nu*s/2`. With `k=4`, `s=2`, all
  Hamiltonians through degree four satisfy this condition.
- Symplecticity: HBVM(k,s) is generally not symplectic when `k > s`.
  HBVM(4,2) must therefore be assessed with a map-Jacobian defect rather than
  labeled symplectic by construction.

## Nonlinear solve

Each step uses a first-order stage predictor and Newton iterations on the
rank-two residual. The Jacobian is assembled from the four stage field
Jacobians. `jacobian_method="auto"` selects exact particle blocks when the
dynamics implements `GuidingCenterJacobianSystem`, and centered finite
differences otherwise. Backtracking is activated only if a full Newton update
increases the residual.

The stopping threshold is

\[
\varepsilon_{abs}+\varepsilon_{rel}
\max(1,\lVert y_n\rVert_\infty).
\]

Main-grid diagnostics include nonlinear iterations, residual norms, effective
tolerances, residual and Jacobian evaluations, and vector-field evaluations.
Shadow output samples are deliberately excluded from these arrays, consistent
with the shared fixed-grid integration contract.

## Energy diagnostics

`track_energy=True` stores

- `hamiltonian`: one row per particle and one column per saved time;
- `energy_drift`: `H(t_n,y_n)-H(t_0,y_0)`;
- `energy_error`: the maximum absolute drift.

These names describe physical Hamiltonian drift. For explicitly time-dependent
systems this is a diagnostic value, not an invariant error.

## Public usage

```python
from simulation import HBVM42, InitialValueProblem, SimulationRequest, simulate

solution = simulate(
    problem,
    HBVM42(
        absolute_tolerance=1e-14,
        relative_tolerance=1e-13,
        jacobian_method="auto",
        track_energy=True,
    ),
    SimulationRequest.uniform(
        t_span=(0.0, 8.0),
        max_step=0.1,
        sample_count=21,
    ),
)
```

## Notebook studies

`studies.run_hbvm42_evaluation` composes the individual experiment. For each
step it computes:

- trajectory-RMS and final-state error against an independent DOP853 solve;
- median and minimum execution time over explicit warm-up/repetition counts;
- maximum absolute and relative energy error;
- centered-difference symplecticity defects of one step and of the final flow;
- observed global orders and signed order reductions `4 - p_observed`;
- nonlinear iterations and field-evaluation work.

`studies.run_hbvm42_bm4_comparison` evaluates HBVM(4,2) and the existing
`BM4Composition(GCExtendedFormulation(...))` against the same DOP853 endpoint,
on identical complete fixed-step grids, with identical timing repetitions.
Only accuracy and execution time enter that comparison; energy and
symplecticity remain in the individual notebook.

The corresponding plots live in `visualization.hbvm42`. The notebooks contain
only explicit scientific configuration, study calls, tables, and
interpretation.

## Files

- `src/simulation/methods/hbvm/order4.py`: coefficients, reduced residual,
  Newton solve, integration, and diagnostics.
- `src/studies/hbvm42.py`: quartic validation system and both study runners.
- `src/visualization/hbvm42.py`: individual and comparison figures.
- `notebooks/developements/hbvm42_individual_evaluation.ipynb`: all individual
  metrics.
- `notebooks/developements/hbvm42_vs_bm4_accuracy_runtime.ipynb`: accuracy and
  runtime comparison.
