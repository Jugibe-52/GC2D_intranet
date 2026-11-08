# Radau: standalone reference solver

Radau is no longer an IntegrationMethod or a simulation facade export. Studies use SciPy directly to generate independent physical references. `studies.reference_trajectory._solve_adaptive` samples accepted dense interpolants and reports SciPy work counters; `studies.single_particle_recurrence` uses `solve_ivp` with dense output for return searches.

The mathematical method remains documented in [theory.tex](../tex/theory.tex) and its [compiled PDF](../tex/theory.pdf). Historical saved results retain their original settings and energy conventions. The former adaptive execution controller, live method adapter, adaptive callback types and SciPy/JAX hybrid bridge are retired without aliases.

Specialized `studies.dop853_energy_balance` can reconstruct historical momentum separately using an eight-node Gauss rule over accepted interpolants. That quadrature has no independent adaptive error tolerance and must be audited by refinement. It is not the behavior of the generic EnergyObserver.

For current fixed-step simulations, see [execution](../../../simulation/execution.md) and [initial data and solvers](../../../simulation/initial-solvers-formulations.md). Their EnergyObserver evaluates H and positive analytic partial_t H only at physical endpoints and requested samples.

The [reference diagram](radau-simulation-architecture.svg) shows the current study-owned SciPy pipeline.
