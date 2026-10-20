+++
schema_version = 2
kind = "change-tasks"
spec_revision = 1
plan_revision = 1
id = "2026-10-20-executor-owned-integration"
sources = ["src/execution/execution.py", "src/execution/_host.py", "src/methods/base.py", "src/methods/adaptive/scipy.py", "src/methods/_jax_dispatch.py"]
documents = ["specs/current/simulation.md", "specs/current/states.md", "docs/simulation/execution.md", "docs/simulation/jax-execution.md"]
open_questions = []

[[tasks]]
id = "T-001"
description = "Capture deterministic CPU/JAX baselines and inventory callers."
depends_on = []
done = true

[[tasks]]
id = "T-002"
description = "Implement executor orchestration, preparation contract and numerical adapters; remove CompiledFixedMethod."
depends_on = ["T-001"]
done = true

[[tasks]]
id = "T-003"
description = "Migrate consumers and verify lifecycle, external providers and Modal behavior."
depends_on = ["T-002"]
done = true

[[tasks]]
id = "T-004"
description = "Reconcile contracts, model documentation, Spanish copies and confirmed memory."
depends_on = ["T-003"]
done = true

[[tasks]]
id = "T-005"
description = "Run numerical baseline comparison and required checks; review evidence and close the change."
depends_on = ["T-004"]
done = true

[[checks]]
id = "V-001"
criteria = ["AC-001", "AC-003", "AC-004"]
procedure = "Run execution, compiled-step, method lifecycle, adaptive and Modal tests; audit removal of method integration entry points."
tests = ["tests/test_execution.py", "tests/test_compiled_step_contract.py", "tests/test_method_integration.py", "tests/test_adaptive_integration.py", "tests/test_execution_modal.py"]
result = "passed"
evidence = "Execution tests (11) passed; focused lifecycle, adaptive, compiled-map, extensibility and Modal tests ran. One obsolete integrate-only Modal fixture initially failed, was migrated, and all 18 Modal tests passed on rerun. The final full suite passed. Runtime inspection confirms methods._compiled is not importable and both BM4 classes directly inherit IntegrationMethod. Methods contain no complete-integration entry point or device resolution."
checked_on = "2026-10-20"

[[checks]]
id = "V-002"
criteria = ["AC-002"]
procedure = "Compare captured before/after CPU and JAX arrays and diagnostics with identical deterministic inputs; run existing cross-backend and method accuracy tests."
tests = ["tests/test_jax_methods.py", "tests/test_jax_rk4.py", "tests/test_adaptive_integration.py"]
result = "passed"
evidence = "Captured 23 deterministic cases before source edits: all 13 CPU methods on GC, four representative CPU FC cases, four JAX GC cases (RK4, BM4Implicit, DOP853, Radau) and two JAX FC cases (RK4, Radau), all with energy tracking and irregular times. Every saved time, physical value and original diagnostic matches exactly after the change; maximum absolute physical-state difference is 0. Existing cross-backend and numerical tests passed in the full run. CPU execution metadata is now supplied consistently by the executor."
checked_on = "2026-10-20"

[[checks]]
id = "V-003"
criteria = ["AC-005"]
procedure = "Audit supported source/notebook callers and docs, Spanish-copy alignment, retired imports, wheel contents and current contracts."
tests = ["tests/test_package_layout.py"]
result = "passed"
evidence = "Source callers migrated; audit found zero retired calls in notebook cells. Parsed 1169 ordinary notebook code cells without syntax errors (pre-existing invalid-escape SyntaxWarnings remain). Modified Markdown documents have ignored Spanish counterparts with matching link destinations/code blocks. Removed module absent from the wheel; isolated wheel imports and a rotation simulation passed. Current contracts and model prose updated; older rendered diagrams explicitly marked historical. Both affected ABBA theory PDFs compiled successfully and their changed pages were visually checked. Memory is 232 lines."
checked_on = "2026-10-20"

[[checks]]
id = "V-004"
criteria = ["AC-001", "AC-002", "AC-003", "AC-004", "AC-005"]
procedure = "Run .venv/bin/python -m mypy src; MPLBACKEND=Agg .venv/bin/python -m unittest discover -s tests -v; both supported examples; .venv/bin/python -m build; .venv/bin/python scripts/check_sdd.py; git diff --check."
tests = ["tests/test_execution.py", "tests/test_jax_methods.py"]
result = "passed"
evidence = "592 tests completed in 420.670 s: OK, with three GPU-only skips because no GPU is available. JAX CPU tests passed. Mypy passed for all 257 source files; examples/gc_orbit.py and examples/projected_abba.py passed; isolated sdist and wheel build succeeded; SDD checker and git diff --check passed. No live Modal deployment or remote submission was performed."
checked_on = "2026-10-20"
+++

# Executor-owned integration: tasks and evidence

## Execution

The specification and plan implement the ownership and class changes already reviewed in the conversation. The explicit request to prepare and execute authorizes this scope. The initial working tree was clean. Baselines preceded source edits. All 23 deterministic before/after cases match exactly, including original diagnostics (maximum physical-state difference 0). CPU results now consistently include execution_backend, execution_device and execution_device_index. The first focused test run exposed one obsolete integrate-only Modal test fixture; it was migrated to new_run and its test module rerun. The final full suite passed with three hardware-related GPU skips.

## Outcome and limitations

The executor now owns environment preparation, isolated runs and driver selection.
CompiledFixedMethod is removed; numerical methods no longer expose integrate or
_integrate_jax. Controllers, numerical formulas and state storage retain their
responsibilities. Supported consumers and current contracts are migrated. Complete
jobs use simulate or Execution.run; external method providers implement new_run
and jax_mode. Execution metadata is injected before finalization without copying
a completed history. All tasks and criteria are verified.

The before-change reference was captured from clean commit `7a0bf56583bddec4c5f2ec3faf585f54bc469472`.
Baseline field: Potential.random(A=.08, M=3, nx=16, ny=16, seed=27). GC uses rho=.05
and initial [1.,1.1,1.2,1.3]; FC uses rho=.2, eta=-.4 and appends velocities
[.2,.3,-.2,-.1]. All runs track energy on span (.3,.34), max_step=.02 and output
[.3,.305,.32,.333,.34]. CPU covers all 13 GC methods plus RK4/GaussLegendre4/DOP853/Radau
on FC. JAX covers RK4/BM4Implicit/DOP853/Radau on GC and RK4/Radau on FC.
The temporary baseline capture/comparison compared every original diagnostic,
including accepted-step work and energy histories, without relaxing tolerances.

GPU execution was unavailable and three existing GPU tests skipped. Modal was
verified through local worker and mocked service tests only. Existing rendered
diagrams and their sources remain historical; current Markdown describes the new
execution ownership. ABBA and ABBA4 theory sources and PDFs include the updated
entry point. No TFM files, numerical equations, deployment or stored study data
were changed. No commit or push was made.


Reference note (2026-10-20): source references now resolve to methods.base and execution after the package merger. Historical design and results in this record retain their original execution context.
