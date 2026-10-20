+++
schema_version = 2
kind = "change-tasks"
id = "2026-10-20-single-execution-coordinator"
spec_revision = 1
plan_revision = 1
sources = ["src/execution/execution.py", "src/execution/_host.py", "src/execution/_jax.py", "src/execution/_jax_fixed.py", "src/methods/_jax_dispatch.py", "src/states/history.py"]
documents = ["specs/current/simulation.md", "specs/current/states.md", "docs/simulation/execution.md", "docs/dynamics/jax-potential-evaluation.md"]
open_questions = []

[[tasks]]
id = "T-001"
description = "Capture baseline and record the reviewed specification and plan."
depends_on = []
done = true

[[tasks]]
id = "T-002"
description = "Implement central lifecycle, bulk recording and reduced workers."
depends_on = ["T-001"]
done = true

[[tasks]]
id = "T-003"
description = "Merge JAX preparation/hybrid adaptation and simplify method dispatch."
depends_on = ["T-002"]
done = true

[[tasks]]
id = "T-004"
description = "Migrate callers and verify lifecycle, recording and numerical agreement."
depends_on = ["T-003"]
done = true

[[tasks]]
id = "T-005"
description = "Reconcile current contracts, model docs, Spanish copies and project memory."
depends_on = ["T-004"]
done = true

[[tasks]]
id = "T-006"
description = "Run required checks, inspect the package and close only with passing evidence."
depends_on = ["T-005"]
done = true

[[checks]]
id = "V-001"
criteria = ["AC-001", "AC-004", "AC-005"]
procedure = "Run lifecycle, states, controller and compiled-contract tests; inspect single allocation/finalization and failure paths."
tests = ["tests/test_execution.py", "tests/test_states.py", "tests/test_method_integration.py", "tests/test_compiled_step_contract.py"]
result = "passed"
evidence = "57 focused tests passed (13 execution, 13 states, 31 method/adaptive/compiled/package). Fixed and hybrid jobs allocate/finalize the same result exactly once. Allocation/finalization failures consume only the fresh run; invalid bulk histories do not publish data."
checked_on = "2026-10-20"

[[checks]]
id = "V-002"
criteria = ["AC-002", "AC-003", "AC-005"]
procedure = "Compare 23 captured deterministic cases exactly and run JAX/potential/adaptive tests."
tests = ["tests/test_jax_methods.py", "tests/test_adaptive_integration.py", "tests/test_jax_potential.py", "tests/test_potential_execution.py"]
result = "passed"
evidence = "All 23 deterministic cases matched times, states and every diagnostic exactly; maximum absolute state difference 0.0. The complete suite also passed standalone JaxPotential, tracing/cache, adaptive, convergence and sampling checks. Compiled arithmetic, controllers, sample extraction, progress and step/time helpers retain identical ASTs."
checked_on = "2026-10-20"

[[checks]]
id = "V-003"
criteria = ["AC-001", "AC-002", "AC-004", "AC-005", "AC-006"]
procedure = "Run mypy, complete unittest suite, both examples, build and clean-wheel smoke; inspect optional imports and mocked Modal."
tests = ["tests/test_package_layout.py", "tests/test_execution_modal.py"]
result = "passed"
evidence = "Python 3.12.3: 598 tests in 460.125 s, OK with 3 GPU-only skips; mypy passed 253 source files. Both examples succeeded. Isolated sdist/wheel build and clean-wheel CPU fixed/adaptive jobs passed; every packaged Python source matches the working tree and the three retired modules are absent. Optional-import and mocked Modal checks passed; no live Modal deployment."
checked_on = "2026-10-20"

[[checks]]
id = "V-004"
criteria = ["AC-006"]
procedure = "Run scripts/check_sdd.py; audit notebook imports, active docs, translations, memory and git diff."
tests = ["tests/test_sdd.py"]
result = "passed"
evidence = "SDD validates 38 records and 19 requirements. Audited 28 bilingual Markdown pairs for technical metadata, code blocks, links and ignored Spanish copies. Audited 251 notebooks and 1180 code cells without retired calls or notebook edits. Physics has no execution import; numerical algorithms and JaxPotential source are unchanged. MEMORY.md has 247 lines; git diff --check passed."
checked_on = "2026-10-20"
+++

# Single execution coordinator: tasks and evidence

## Execution

The working tree was clean at 1a21217. The user accepted the discussed scope and explicitly requested preparation and implementation. Baseline capture preceded runtime changes. Implementation is complete; the following evidence was observed on the final runtime source.

## Outcome and limitations

Implemented and verified on 2026-10-20. Execution.run now owns the complete lifecycle and physical result boundary. Three private modules and two independent complete coordinators are retired. JaxPotential remains public and independently usable; the simulation prepares it automatically without refitting splines. No numerical algorithm changed.

The exact comparison used a synthetic Potential.random(A=.08, M=3, nx=16, ny=16, seed=27), GC rho=.05 and FC rho=.2/eta=-.4, two particles, span (.3,.34), max_step .02, outputs [.3,.305,.32,.333,.34] and energy tracking. Cases covered all 13 CPU GC methods, four CPU FC methods, four JAX GC methods and two JAX FC methods. Captured from clean 1a21217; every diagnostic and physical history was compared exactly. The temporary procedure is /tmp/gc2d_coordinator_baseline.py; its evidence is /tmp/gc2d-coordinator-baseline.log.

Complete-suite evidence is /tmp/gc2d-coordinator-tests.log; type, package and semantic audits are /tmp/gc2d-coordinator-mypy.log, /tmp/gc2d-coordinator-audit.log and /tmp/gc2d-coordinator-semantics.log. Verification covered JAX CPU and mocked Modal transport. Three GPU checks were skipped because hardware was unavailable; no live Modal job or remote deployment was run. Canonical notebooks, TFM content and theory PDFs were unchanged. No commit or publication was performed.
