+++
schema_version = 2
kind = "change-spec"
revision = 2
status = "verified"
approval = "2026-10-20: User accepted revision 2 and explicitly requested SDD implementation."
id = "2026-10-20-unified-states"
sources = ["src/states/history.py", "src/diagnostics/geometry.py", "src/states/generators.py", "src/formulations/layout.py", "src/contracts/configuration.py", "src/problem.py"]
documents = ["specs/current/simulation.md", "specs/current/results.md", "docs/simulation/integration-architecture.md"]
requirements = ["SIM-001", "SOL-001", "SOL-002", "STA-001", "STA-002"]
proposed_requirements = ["STA-001", "STA-002"]
open_questions = []

[[criteria]]
id = "AC-001"
description = "One State type stores a finite nonempty packed snapshot with explicit layout and independent copies; GC and FC block order and units are unchanged."
requirements = ["SIM-001", "STA-001"]

[[criteria]]
id = "AC-002"
description = "Component and block operations round-trip single states and histories exactly and preserve NumPy/JAX backends in layouts."
requirements = ["STA-001"]

[[criteria]]
id = "AC-003"
description = "External radial and contour generators in states.generators preserve coordinates, ordering, defaults and validation; independent signed-area calculation preserves periodic handling."
requirements = ["STA-002"]

[[criteria]]
id = "AC-004"
description = "Problems retain state/layout snapshots, solutions retain immutable histories and provenance, and deterministic GC/FC simulations remain numerically equivalent."
requirements = ["SIM-001", "SOL-001", "SOL-002"]

[[criteria]]
id = "AC-005"
description = "Supported consumers and model documentation use states and states.generators; the initial_conditions package is removed after migration, removed imports are documented, and persisted numerical data is unchanged."
requirements = ["STA-001", "STA-002", "SOL-001"]
+++

# Unified states: specification

## Problem

StateConfiguration stores an initial vector, while GC/FC subclasses mainly select a layout and forward operations. Generic manipulation already lives in contracts.state_layout. Area additionally mixes storage, contour generation and area diagnostics. These responsibilities obscure the distinction between a physical state and its preparation.

## Scope

Replace built-in configuration storage with State in states; place external generator functions in states.generators and keep area evaluation as an independent diagnostic. Remove the initial_conditions package after migrating its callers. Migrate supported code, examples, tests, model documentation and notebook callers. No numerical equations, solvers, unit conversion, saved-history format or TFM text changes are included. Preparation writes records only.

## Proposed behavior

STA-001: State represents one instantaneous physical state, not necessarily the initial instant. Construction requires a finite nonempty one-dimensional vector and an explicit compatible layout. No unset state is allowed. GC order is [x_1,...,x_N,y_1,...,y_N]; FC appends vx then vy. N is positive. Storage and access use independent copies; the layout is captured independently. Changing the supplied arrays cannot change the object. A new state is constructed to replace values. Physical parameters and normalization remain owned by dynamics.

Expose construction from components and access to split components, positions, optional velocities and particle count. Keep generic packing and block operations in reusable layouts: histories retain shape (d*N,*sample_axes), component blocks (N,*sample_axes), and explicit blocks (d,N,*sample_axes), with d=2 or 4 for built-ins. State stores a NumPy snapshot; backend-preserving NumPy/JAX operations remain available on layouts without converting traced values to NumPy. Invalid velocity access for GC must fail clearly.

STA-002: radial_star, ranked_radial_star, square and circle are external functions in states.generators returning State, rather than methods on State or a separate initial_conditions package. Preserve existing parameters, defaults, ordering, central-point rules, endpoint rules and validation. Contours contain ordered boundary vertices rather than interior samples. Signed polygon area becomes an independent function, including existing optional minimum-image periodic reconstruction. It remains a local contour calculation, not a new general treatment of winding contours on a torus.

State can serve directly as initial data through the existing structural InitialConfiguration protocol. Its initial_state property is a compatibility accessor for the same snapshot. Preserve problem ownership and Solution history/provenance contracts. The integration core continues to operate on arrays without allocating State objects per numerical step; existing diagnostic replay may construct a problem snapshot. Coordinate units and particle identities remain unchanged.

Source references were updated for the 2026-10-20 states migration; prior recorded outcomes are unchanged.

Source locations were updated on 2026-10-22 after the contract ownership move;
the historical scope, acceptance and recorded checks above are unchanged.
