# Independent state formulations

Make the choice between one physical state and two copies independent of the
physical model. Guiding-center (GC) coordinates and full-cyclotron (FC)
coordinates should use the same representation rules, with their dimensions
and component meanings supplied by the physical layout.

The current `PhysicalFormulation` and `DoubledFormulation` already derive sizes
from the problem, but also retain the problem and initial time and construct the
initial state. Specialized GC maps separately construct initial copies and
combine representation, elementary flows and projection helpers. This obscures
which object starts a run, prepares a step or selects a projection.

Introduce clear `SingleFormulation` and `DoubleFormulation` representations.
The numerical method owns run initialization and step preparation, selects its
projection, and invokes any required solver. A formulation supplies reusable
representation operations; it does not drive the simulation.

This proposal refines the completed
[initial-condition and solver change](../2026-10-21-initial-conditions-solvers-formulations/results.md)
and [palindromic-method change](../2026-10-21-palindromic-operations/results.md).
It preserves the currently supported numerical algorithms and their physical
results. Extending GC-only integrators to FC is a separate scientific change.
