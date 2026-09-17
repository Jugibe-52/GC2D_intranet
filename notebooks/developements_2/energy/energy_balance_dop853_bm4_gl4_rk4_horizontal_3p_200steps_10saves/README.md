# Energy balance and DOP853 trajectory comparison

Run `calculation.ipynb` remotely with the current project source installed (`pip install -e ".[notebook]"` from the project root). Transfer `results.npz` back to this directory, then run `visualization.ipynb` locally. No timing campaign or Radau integration is performed. Computation remains pending.

The visualization writes `trajectories.html`, a standalone animation with method toggles. NPZ persistence contains no pickled objects. Both notebooks require the source modules added with this study, not only the notebooks themselves.

The three horizontal initial positions are defined explicitly in calculation.ipynb. Each fixed method performs 40,000 steps and saves 2,001 states. DOP853 uses an adaptive maximum step of 0.005 and the same output grid. Energy is the extended balance H+kappa, reconstructed/integrated with each method's stages.

`legacy_previous_study/` preserves the former notebooks, CSVs and logs for provenance. These results must not be loaded into the new study. It can be omitted from the remote transfer.

## AWS execution dispatched on 2026-09-08

Instance `i-0246fff3c5553a26e`, region `eu-south-2`. Remote root: `/home/ubuntu/gc2d-energy-balance-20260908`. The full source snapshot is recorded in `source_manifest.json` on the worker. A detached process executes `run_remote.py` with the existing Python 3.12 environment. It writes `calculation.log`, `execution_status.json`, `calculation_executed.ipynb` and, on success, `results.npz` in the study directory. This runner does not shut down the instance automatically. `remote_run.json` records dispatch details; its state is a snapshot, not a live monitor.
