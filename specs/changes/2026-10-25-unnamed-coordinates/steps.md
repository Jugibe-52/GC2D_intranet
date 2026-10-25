# Steps

- [x] Retain pre-change numerical baseline and GC/FC archives from the first run.
- [x] Implement stateless GC/FC coordinates and their respective JAX subclasses;
  migrate snapshots, contracts, model declarations, guards and consumers.
- [x] Verify fixed conventions, absent configurable fields, plain tuples,
  immutable snapshots, explicit custom layouts and NumPy/JAX agreement.
- [x] Compare final trajectories and diagnostics to the original baseline and
  load the pre-change archives; update permanent docs and Spanish copies.
- [ ] Run source mypy, the full headless suite, both examples, distribution build,
  whitespace checks and final SDD validation; record actual results and limits.
