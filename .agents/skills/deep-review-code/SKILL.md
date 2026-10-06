---
name: deep-review-code
description: Perform an in-depth, read-only analysis of specified code modules or paths for correctness, architectural coherence, and numerical concerns. Accept module names or paths alone after the skill name.
---

# Deep Review Code

Module names or paths are sufficient input:

```text
$deep-review-code potential
$deep-review-code potential dynamics
```

- Locate the specified modules or paths. Read related contracts, consumers, documentation, and tests to understand their relationships, keeping the analysis focused on the selected code. Ask for the target when it is missing or cannot be determined unambiguously.
- Analyze possible logic errors, edge cases, array shapes, units, signs, and mathematical consistency.
- Review responsibilities, dependencies, layer boundaries, duplication, and API consistency against the project's conventions and documented architecture.
- Where relevant, examine numerical stability, backend differences, potential performance costs, and gaps in test coverage. Read tests as evidence of intended coverage, not proof that they pass.
- Work strictly through read-only inspection. Do not modify files, run tests, simulations, benchmarks, or other execution-based validation, or apply corrections. Suggest implementation work separately.

Report the most relevant issues first, giving a file and line location, explanation, consequence, and suggested correction. Distinguish code-supported problems from suspicions requiring validation and optional improvements. Do not claim execution results or measured performance costs.

Briefly state what was reviewed and what static analysis could not confirm. Do not require a minimum number of findings or invent issues when none are supported.
