# Steps

- [x] Capture pre-change ABBA2, ABBA4 and BM4 states and diagnostics for both
  formulations and solvers, analytic/difference Newton and NumPy/JAX.
- [x] Implement candidate-only residual/Jacobian functions and migrate assembly
  helpers, preserving composition reuse and particle blocks.
- [x] Check derivatives independently, candidate changes, trace reuse, backend
  agreement and regression against the captured baseline.
- [x] Update permanent documentation, Spanish copies, project memory and the
  TFM; compile `latexmk -pdf main.tex` in the TFM root.
- [x] Run the relevant tests, mypy, examples, build and SDD checker; record actual
  results and any limitations.
