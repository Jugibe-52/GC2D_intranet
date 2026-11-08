# Steps

- [x] Capture deterministic baseline runs for the five named methods and both
  generic recipe methods. Include nonzero initial time, several particles,
  irregular sampling, a shortened final step, energy observations, and
  representative reduced/simultaneous projection and Newton/Broyden settings
  where supported. Record the actual inputs and solver tolerances.
- [x] Move the family packages and split the seven concrete methods and three
  bases into the proposed modules. Preserve recipes, shared algorithm code,
  class inheritance, constructor defaults, and explicit public exports.
- [x] Migrate active consumers, diagnostic imports, and local notebook
  references. Verify the new defining modules, retired-path removal,
  serialization, and local worker preparation. Preserve existing Runge–Kutta
  work and the numerical test references.
- [x] Update affected current specifications and permanent documentation,
  including model paths, diagrams, theory references, and Spanish reading
  copies. Regenerate affected diagram assets and theory PDFs where needed.
- [x] Compare baseline and relocated results, including states, energies,
  projection multipliers, and iteration counts. Expect identical NumPy
  results for unchanged arithmetic; record measured maximum discrepancies.
  Run existing focused method/diagnostic and CPU/JAX tests using their explicit
  tolerances, followed by strict source typing, the full headless test suite,
  both smoke examples, and a distribution build. Inspect package contents and
  run only reduced notebook checks. Report unavailable backends as limitations.
- [x] Record acceptance, implementation, observed checks, and any limitations
  in `results.md`; complete only verified checklist items. Run
  `python scripts/check_sdd.py` and review the final changes for stale active
  imports, unintended numerical changes, and preservation of unrelated work.
