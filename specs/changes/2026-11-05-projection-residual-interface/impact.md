# Affected parts

- `methods.extended.core.projection` binds the complete residual and Jacobian
  and adapts them to the existing solvers. `methods.projection` and its contract
  rename the assembly helper; its direct test caller must migrate.
- ABBA2/ABBA4 use both formulations and BM4 uses reduced equations. Preserve
  analytic particle blocks, dense and centered-difference derivatives, initial
  simultaneous evaluation reuse, Newton/Broyden and JAX execution.
- Add independent derivative and trace-reuse checks, then exercise existing
  integration, observation and backend tests. Compare states and diagnostics
  with a pre-change baseline using the same physical inputs and tolerances.
- Update the solver/formulation guide, extended-model architecture, current
  simulation specification and relevant memory, with Spanish reading copies.
  Update the projection discussion and solver connection in TFM chapters 2/3,
  preserving existing edits, and rebuild its root `main.pdf`.

Configured method constructors, notebooks, study settings and saved trajectories
need no migration. The renamed helper is an internal equation operation;
the general solver API and numerical algorithms remain unchanged.
