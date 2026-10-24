# Methods receive functions; solvers and observers own their work

Simplify numerical methods by separating physical advancement, algebraic solves
and passive diagnostics. Execution supplies the functions needed by an algorithm;
methods no longer retain a physical problem or dynamics object. Ordinary methods
need no formulation object. Doubled methods retain only a small copy helper.

Concentrate nonlinear controls in the solver and energy tracking in an observer.
Remove duplicated configuration, initialization, capability checks and numerical
helpers rather than replacing them with a universal interface or extra adapters.
Measure the complete production-code change, including new contracts and observers.

Preserve physical trajectories, effective numerical settings, observer information
and saved scientific data. The physical coordinates and single accepted state
established by the [previous change](../2026-10-23-physical-state-and-generic-workspaces/proposal.md)
remain valid; ownership of passive momentum moves out of methods. This document
set prepares the change. Implementation and numerical verification remain pending.
