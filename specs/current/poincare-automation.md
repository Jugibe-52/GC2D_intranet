+++
schema_version = 1
kind = "spec"
id = "poincare-automation"
sources = ["src/studies/poincare_automation.py", "src/studies/poincare_automation_io.py", "src/visualization/poincare_automation.py", "scripts/run_poincare_automation.py", ".agents/skills/simular-poincare/SKILL.md"]
documents = ["docs/studies/poincare-automation.md"]

[[requirements]]
id = "PNC-001"
tests = ["tests/test_poincare_automation.py::AutomationTests.test_exact_reference_particles_and_default_schedule", "tests/test_poincare_automation_viewer.py::AutomationViewerTests.test_all_cases_in_one_viewer_with_original_colors_and_no_integration"]

[[requirements]]
id = "PNC-002"
tests = ["tests/test_poincare_automation.py"]

[[requirements]]
id = "PNC-003"
tests = ["tests/test_poincare_automation_viewer.py"]

[[requirements]]
id = "PNC-004"
tests = ["tests/test_poincare_automation.py::AutomationTests.test_cli_prepare_run_view_and_status"]
verification = "Review skill operation routing and missing-input prompts; validate frontmatter/discovery metadata with skill-creator quick_validate.py."
+++

# Fixed-particle Poincare studies

## PNC-001: Persistent particle identity

Every case uses the reference ranked 40-particle star (eight arms, radius
0.425 cell widths, first angle zero) followed by the eight literal probe
fractions owned by `studies.poincare_automation`. IDs are 1-48; positions and
ordering do not change with rho, B or method. The original 40-entry turbo
palette and eight fixed probe colors remain stable in the shared viewer.

## PNC-002: Reproducible physical composition and execution

Resolve rho/method lists explicitly, B=1.5 and both rescaling flags true by
default. Use 5000 cycles and 50 complete steps per cycle unless overridden.
The selected original HDF5 field has positive cyclic frequency; the physical
cycle is 1/f_ref. Preserve the user's rho definition through
rho_m=rho*reference_length/(2*pi), reference_length=0.06 m by default. Use the
loader's spatial/time/amplitude scales and apply B once in GC dynamics.

Integrate all 48 particles together with BM4Midpoint, BM4Implicit, RK4 or
GaussLegendre4. Save unwrapped float64 cycle states including both endpoints
and validate the requested schedule, initial state and method diagnostics.
Reject input/source/identity mismatches during archive reuse; preserve
completed cases and report failures. Local execution/storage is the CLI
baseline, with explicit optional backend choices and no silent fallback.

## PNC-003: One viewer for every case

One existing HTML viewer contains the potential/velocity panel and one return
panel with all 48 particles. Case choices stay within the same page. Preserve
cycle zero, every return, particle groups/colors, playback and start-cycle
rings. Unavailable cases stay disabled. Reconstruct fields from verified inputs
without integration, map H and velocity to cell/cycle units, and audit closure
across 51 phases. Folding changes only display copies.

## PNC-004: Bounded skill operations

The project-local simular-poincare skill resolves inputs and uses the tested
CLI. Prepare writes configuration only; run executes and exports; view
reconstructs from saved cases; status and consultation are read-only. Ask only
for missing material choices, retain fixed defaults, and do not treat creating
the skill as authorization to launch a production run. Creation/editing obeys
the personal prefix rule; explicit execution authorizes only its requested scope.
