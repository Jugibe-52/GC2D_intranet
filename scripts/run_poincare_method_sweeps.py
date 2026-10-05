"""Execute a saved, explicit Poincare batch specification with JAX on Modal."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from diagnostics.paths import find_project_root
from studies.poincare_gap_probes import GapProbeSeeds
from studies.poincare_rho_batch import RhoBatchJob, run_modal_rho_batch
from studies.poincare_rho_sweep import RhoStarConfig


def main(spec_path: Path) -> None:
    """Resume existing jobs and report failed runs without resubmitting them."""
    spec = json.loads(spec_path.read_text())
    root = find_project_root(spec_path.parent)
    seed_config = spec.get('probe_seeds')
    seeds = GapProbeSeeds(**seed_config) if seed_config is not None else None
    jobs = [RhoBatchJob(
        job_id=row['job_id'], experiment_path=row['experiment_path'], run_id=row['run_id'],
        config=RhoStarConfig(**{**row['config'], 'source_selection': tuple(row['config']['source_selection'])}),
        probe_seeds=seeds,
    ) for row in spec['jobs']]
    report = run_modal_rho_batch(
        spec['source'], jobs,
        record_directory=root / spec.get('record_directory', 'logs/execution_modal/poincare_method_sweeps'),
        progress_path=spec_path.with_name('progress.json'), max_workers=spec['max_workers'],
        app_name=spec['app_name'], storage=spec['storage'], project_root=root,
        save_folded_returns=True, expected_source_sha256=spec['expected_source_sha256'],
    )
    print(json.dumps(report, indent=2), flush=True)
    if any(row['status'] != 'completed' for row in report['jobs']):
        raise SystemExit('Some runs did not complete; inspect progress.json and recover confirmed calls.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec', type=Path)
    main(parser.parse_args().spec.resolve())
