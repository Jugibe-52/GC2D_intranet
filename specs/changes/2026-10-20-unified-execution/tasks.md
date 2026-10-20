+++
schema_version = 2
kind = "change-tasks"
id = "2026-10-20-unified-execution"
spec_revision = 1
plan_revision = 1
sources = ["src/execution/execution.py", "src/execution/_host.py", "src/execution/state.py", "src/methods/base.py", "src/methods/adaptive/scipy.py"]
documents = ["specs/current/simulation.md", "specs/current/states.md", "docs/simulation/execution.md"]
open_questions = []

[[tasks]]
id = "T-001"
description = "Capture the clean baseline and inventory package, notebook and documentation consumers."
depends_on = []
done = true

[[tasks]]
id = "T-002"
description = "Move the numerical base into methods and the complete integration machinery into execution; remove integration."
depends_on = ["T-001"]
done = true

[[tasks]]
id = "T-003"
description = "Migrate consumers and packaging, and verify import order, lifecycle, sampling and numerical equality."
depends_on = ["T-002"]
done = true

[[tasks]]
id = "T-004"
description = "Reconcile current contracts, affected documentation, Spanish copies and project memory."
depends_on = ["T-003"]
done = true

[[tasks]]
id = "T-005"
description = "Run required checks, inspect the built wheel and record actual evidence."
depends_on = ["T-004"]
done = true

[[checks]]
id = "V-001"
criteria = ["AC-001", "AC-002", "AC-004"]
procedure = "Run package-layout, method lifecycle, executor and Modal tests; check clean-interpreter import orders and removed routes."
tests = ["tests/test_package_layout.py", "tests/test_method_integration.py", "tests/test_execution.py", "tests/test_execution_modal.py"]
result = "passed"
evidence = "All 593 tests passed with three GPU-only skips. The eight package-layout tests include five clean-interpreter import orders with JAX and Modal imports blocked, configured-method/state pickle round trips and fixed/adaptive simulations. Existing controller/provider, lifecycle, optional-import and mocked Modal tests passed. No integration package or aliases remain."
checked_on = "2026-10-20"

[[checks]]
id = "V-002"
criteria = ["AC-003"]
procedure = "Compare all 23 baseline cases with exact times, physical arrays and all diagnostics; run existing adaptive and JAX tests."
tests = ["tests/test_adaptive_integration.py", "tests/test_jax_methods.py", "tests/test_states.py"]
result = "passed"
evidence = "All 23 before/after cases match exactly: every saved time, physical value, diagnostic key and diagnostic value; maximum absolute physical-state difference 0. Baseline captured from clean commit 87f85d426af202049d763fb27772cc46ff81c9f3. Moved controller, sampling, recording and readonly-preparation helpers have identical syntax trees; state and progress/count utilities are byte-identical. Existing adaptive and CPU/JAX accuracy tests passed."
checked_on = "2026-10-20"

[[checks]]
id = "V-003"
criteria = ["AC-005"]
procedure = "Audit imports, notebook cells and documents; build and inspect a clean wheel; check translated Markdown and SDD references."
tests = ["tests/test_package_layout.py"]
result = "passed"
evidence = "Audited 1180 notebook code cells: no retired runtime imports, no notebook edits or long executions. All 35 affected Markdown pairs have Spanish copies with matching technical metadata, code blocks and link targets; existing translated prose metadata was preserved. Current and historical SDD references resolve. A clean sdist/wheel build passed, every packaged Python file matches final sources, the retired integration package is absent and extracted-wheel fixed/adaptive simulations passed without editable-source imports. Both affected DOP853/Radau theory PDFs compiled and their changed second pages were visually checked; no overfull or undefined-reference errors were reported. MEMORY.md has 239 lines."
checked_on = "2026-10-20"

[[checks]]
id = "V-004"
criteria = ["AC-001", "AC-002", "AC-003", "AC-004", "AC-005"]
procedure = "Run mypy, the complete headless unittest suite, both supported examples, package build, SDD checker and git diff --check."
tests = ["tests/test_execution.py", "tests/test_jax_rk4.py"]
result = "passed"
evidence = "Full suite: 593 tests in 466.787 seconds, OK with three unavailable-GPU skips. Mypy passed for 256 source files. Both supported examples, isolated source/wheel builds and clean-wheel simulations passed. SDD validator passed (35 records, 19 requirements) and git diff --check passed. No live Modal deployment/submission or TFM change was performed."
checked_on = "2026-10-20"
+++

# Unified execution: tasks and evidence

## Execution

The user reviewed the merger in the conversation and explicitly requested preparation and execution through SDD. The preceding project synchronization completed before this change began. Its baseline is clean commit 87f85d4. Task completion and verification evidence are recorded above.

## Outcome and limitations

Integration and execution are one subsystem. The separate integration package and
its packaging entry are removed. methods.base owns IntegrationMethod; execution
owns host/device loops, fixed control, samples, progress and current snapshots.
Public simulation/executor signatures and numerical methods are unchanged.
Controller/state extensions are available from execution and the numerical base
from methods. Old integration imports are retired without aliases. Current contracts,
model documentation, project instructions, memory and Spanish copies are updated.

Baseline: Potential.random(A=.08, M=3, nx=16, ny=16, seed=27); GC rho=.05 and
initial [1.,1.1,1.2,1.3]; FC rho=.2, eta=-.4 with velocities [.2,.3,-.2,-.1].
Span (.3,.34), max_step=.02 and outputs [.3,.305,.32,.333,.34], with energy enabled.
CPU cases cover all 13 GC methods and RK4/GaussLegendre4/DOP853/Radau on FC.
JAX cases cover RK4/BM4Implicit/DOP853/Radau on GC and RK4/Radau on FC.
All histories and the complete diagnostic mappings match the captured baseline exactly.

Three GPU tests could not run because this machine has no JAX GPU. Modal was
checked through local workers and mocked service tests only. Historical diagrams
and their generators remain historical; their current owning documents explain
the new structure. Saved physical archives are unchanged; serialized live snapshots
with retired module paths are not a supported archive format. No notebook outputs,
TFM content or scientific formulas changed. This implementation remains local;
the earlier synchronization published only its predecessor, commit 87f85d4.
