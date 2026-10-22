# Affected code and use

Changes are private implementation details in `states`, `dynamics`,
`contracts.comparison`, `methods.classical`, `studies._comparison`,
`studies.poincare_automation`, and six visualization modules. No new public
arguments, exports, archive fields or notebook edits are planned.

Constructor compatibility includes Boolean versus integer policies, signed
physical parameters and energy errors, copied immutable arrays, preserved
history dtypes, supplied solver precedence and the first reported invalid input.
Recording state transitions and numerical evaluations must not move.

Plot preparation prevents partially created figures on invalid input, including
when an earlier series is valid. Existing open figures must remain untouched.
Valid data must still produce the same curves. Existing validation-contract,
state ownership, comparison, Poincare and plotting tests provide the baseline;
add focused regression coverage where this behavior is missing.

Update the integration architecture and current state/result specifications to
describe the implemented ownership of validation. No numerical theory or data
migration is required; consumers retain the same public API.
