# Affected parts

Replace the interim implementation in this same change folder. The coordinate
base, four concrete classes, GC/FC declarations, layout contract, snapshots,
method guards and public exports change together. Adapt external test layouts
to explicit classes and snapshots; keep generic numerical-method coverage.

Update guides, affected model simulation architecture, current contracts and
memory, including Spanish copies. Scan notebooks for retired imports; execute
only a reduced relevant subset if a notebook needs migration. Existing numerical
archives keep their GC/FC labels and require no rewriting. New objects must still
serialize correctly; older configured pickles are not migrated.

Retain prior results as evidence for the superseded index-based implementation.
Reuse its pre-change numerical baseline and archives, but rerun validation of
the final class hierarchy, ownership, JAX and integrations. Physical equations,
units, solvers and integration algorithms are unchanged. The separate
`2026-10-25-shared-fixed-methods` proposal is outside this change.
