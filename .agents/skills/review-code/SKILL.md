---
name: review-code
description: Review specified code modules or paths without modifying files. Accept module names or paths alone after the skill name.
---

# Review Code

```text
$review-code potential
```

- Locate and review the specified modules or paths. Consult their contracts, consumers, documentation, and tests only as needed to understand the selected code. If the target is ambiguous, ask which module or path to review.
- Review only: do not modify files or apply fixes. Describe any suggested corrections; implementation requires a separate user request.
