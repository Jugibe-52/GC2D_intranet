+++
schema_version = 2
kind = "change-tasks"
id = "new-change"
spec_revision = 1
plan_revision = 1
sources = []
documents = []
open_questions = ["Derive concrete tasks and checks from the specification and plan."]

[[tasks]]
id = "T-001"
description = "Replace with an actionable task from the plan."
depends_on = []
done = false

[[checks]]
id = "V-001"
criteria = ["AC-001"]
procedure = "Replace with a reproducible command or manual verification procedure."
tests = []
result = "pending"
evidence = ""
checked_on = ""
+++

# Change title: tasks and evidence

## Execution

Describe relevant progress and resumption context. Order tasks so dependencies
appear before their consumers. Metadata owns completion and check results;
avoid a second checklist. Preserve previous outcomes here when revising scope.

## Outcome and limitations

Not implemented. Record actual results and remaining limitations when work is
performed. Do not replace observed results with expected results.
