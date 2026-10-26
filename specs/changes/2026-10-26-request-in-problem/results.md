# Results

Accepted on 2026-10-26 through the explicit request to prepare and execute the
request relocation discussed in the conversation. Scope: separate input classes
in problem.py, migrated consumers and packaging, unchanged numerical behavior.

## Implemented

Moved SimulationRequest, its validators and default constants into problem.py;
removed request.py and migrated internal consumers, tests, packaging and four
Modal source lists. Public simulation imports and simulate's signature remain
unchanged. InitialValueProblem still references dynamics and InitialConditions,
whose implementations remain in their existing packages. Syntax-tree comparison
against HEAD confirmed that both classes and the moved validators are unchanged.

Updated current documentation and Spanish reading copies. The SDD checker now
resolves the retired source path only in historical records; current contracts
must use the new path. Its initial three missing-source failures are resolved
without rewriting history. Memory records the new owner and remains at 300 lines.

Migrated the import in notebooks/developements/bm4_simple_inspection.ipynb and
notebooks/developements/bm4_h5_simple_inspection/calculation.ipynb. Their affected
import/configuration cells executed successfully in isolated dictionaries. Both
notebooks remain ignored; historical from-b2 copies were preserved. Complete
notebook integrations were not run.

## Verification

Using .venv/bin/python on 2026-10-26:

- `-m unittest discover -s tests -p test_package_layout.py -v`: 13 passed,
  including public export identity, import isolation and removal of request.
- `-m unittest discover -s tests -p test_sdd.py -v`: 35 passed, including historical
  source resolution and rejection of retired paths in current specifications.
- `-m mypy src`: passed for 302 source files.
- With MPLBACKEND=Agg, examples/gc_orbit.py and examples/projected_abba.py passed
  with 20 and 4 steps respectively.
- `-m build --outdir /tmp/gc2d-request-build`: source archive and wheel built;
  inspected both for problem.py and absence of request.py. The earlier
  --no-isolation attempt failed because setuptools is absent from the project
  environment; the standard isolated build supplied its own dependencies.
- `git diff --check`: passed. Spanish copies and development notebooks remain
  excluded from Git.

- `MPLBACKEND=Agg .venv/bin/python -m unittest discover -s tests -v`: 712 tests
  in 646.116 seconds; 709 passed and 3 GPU checks skipped because no JAX GPU is
  available. The focused layout and SDD suites also passed after their final
  assertions were added.
- `.venv/bin/python scripts/check_sdd.py`: passed, 134 records and 22 requirements.

All planned implementation and local verification steps are complete.

## Compatibility

Old configured-object pickles referring to request.SimulationRequest must be
recreated. Remote clients and workers need the same revision. No remote jobs,
deployments, commits or pushes were performed; numerical archive schemas and
physical algorithms are unchanged. GPU checks depend on available hardware.

This completed relocation is superseded by [problem composition](../2026-10-26-problem-owns-request/proposal.md), which restores request.py and stores its request object inside InitialValueProblem. The checks above describe the earlier implementation.
