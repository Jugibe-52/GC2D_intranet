---
name: simular-poincare
description: Prepare, run or visualize GC2D Poincare simulations with the fixed 40-star plus 8-probe particle set and one shared viewer. Use for requests to automate these studies or sweep rho, B and supported methods; ordinary physics explanations and arbitrary particle geometries are outside this workflow.
---

# Fixed-particle Poincare simulations

Use the repository's tested automation, not copies of the historical notebooks.
Read [usage and parameter conventions](../../../docs/studies/poincare-automation.md)
when resolving parameters or execution choices. Locate the repository from this
skill's directory and use its available Python environment (`.venv/bin/python`
on the current host). Run commands from the repository root.

## Resolve the operation and inputs

- **Prepare / prepara:** save a configuration and list its cases without integration.
- **Run / ejecuta / simula:** prepare if needed, run requested cases, validate results,
  export and open the single viewer. An existing directory resumes missing cases.
- **View / visualiza / abre el visor:** use saved cases and reconstruct their field
  panels without integrating trajectories.
- **Status / estado or a question:** inspect inputs/results read-only. The CLI
  `status` operation writes nothing. Discussing the skill is not a run request.

Ask only for missing material inputs. Infer already specified choices from the
request/context; otherwise ask for one or more rho values and numerical methods
in one concise question. Suggest BM4Midpoint if the method is missing. The
physical model is GC; supported methods are BM4Midpoint, BM4Implicit, RK4 and
GaussLegendre4. Reject unsupported models/methods without substitution.

Use these defaults unless the user supplies an override: **5000 cycles**, **50
steps per cycle**, **B=1.5**, spatial and temporal potential rescaling enabled,
source PHI_2.h5 variable index 15 plus constant field 0, cubic interpolation,
no smoothing, zero coupling and local execution/storage. Backend defaults to
SciPy/CPU; JAX/CPU is available when selected and the CLI enables float64 for
that run. Explain any proposed backend change before running. Do not infer
cloud authorization from the reference notebooks' historical Modal/bucket use.

The same 48 initial positions, IDs and colors are fixed for every invocation.
Requested rho retains `rho_m = rho*0.06/(2*pi)` meters; it is not rho divided by
the cell width. The study maps it into the chosen runtime coordinates and
passes B only to GC dynamics. Do not add a legacy `1/(2*pi)` amplitude factor.
The shared code owns these conventions and particle coordinates.

## Execute through the shared entry point

For example, from the repository root:

```bash
.venv/bin/python scripts/run_poincare_automation.py prepare --rho 0.3 --method BM4Midpoint --B 1.5
.venv/bin/python scripts/run_poincare_automation.py run /absolute/path/printed/by/prepare
.venv/bin/python scripts/run_poincare_automation.py view /absolute/path/printed/by/prepare
.venv/bin/python scripts/run_poincare_automation.py status /absolute/path/printed/by/prepare
```

Use the actual directory returned by prepare. Multiple values after `--rho`,
`--B` or `--method` create their Cartesian product. Show the resolved choices,
case count, total integration steps and destination before running; a requested
run with resolved inputs needs no additional approval. Use `--help` and the
usage guide for supported overrides rather than inventing CLI flags.

A successful run saves unwrapped float64 cycle states and regenerates one
`viewer.html` for the invocation. Open that absolute file with the available
Codex preview/browser tool and provide its path. All 48 particles share one
return panel; the adjacent panel shows the effective potential and GC velocity.
The case selector changes rho/B/method within the same viewer. Missing or failed
cases remain unavailable. Report failures from `cases.json` and do not describe
a partial viewer as a completed sweep. Rerunning reuses validated complete cases;
never overwrite mismatched inputs or relabel historical archive clocks.

Creating this skill does not launch a study. Explicit execution of its function
authorizes necessary study outputs within the request; implicit selection alone
does not bypass the personal `&` rule. Creating/editing skills still requires
that prefix. Parameter changes within the supported contract need no new SDD
proposal; changes to physics, particle identity or method support do.
