"""Opt-in persistence of reduced periodicity diagnostics and their provenance."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from studies.poincare_periodicity import PoincarePeriodicity, SavedPoincareSection, ShortPeriodicity


def _write_records(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_poincare_periodicity(directory: str | Path, section: SavedPoincareSection,
                                result: PoincarePeriodicity, *, controls: dict[str, Any],
                                interpretation: str) -> Path:
    """Save tables and an auditable report separately from original run products."""
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    for name, rows in (('summary', result.summary), ('thresholds', result.thresholds),
                       ('candidates', result.candidates), ('initial_positions', section.initial_records)):
        _write_records(destination / f'{name}.csv', rows)
    rows = []
    for k, q in enumerate(result.lags):
        for j, pid in enumerate(section.particle_ids):
            values = result.metrics[k, j]
            rows.append(dict(particle=pid, q=int(q), rms_over_L=values[0], p95_over_L=values[1],
                             max_over_L=values[2], first_half_rms=values[3], second_half_rms=values[4]))
    _write_records(destination / 'lag_scan.csv', rows)
    metadata = dict(source=section.metadata, source_sha256=section.source_hashes,
                    controls=controls, effective_max_lag=int(result.lags[-1]),
                    selected_particle_ids=section.particle_ids,
                    sample_scope='Initial state and once-per-forcing-cycle returns only',
                    distance='Minimum-image Euclidean norm divided by cell period L')
    (destination / 'analysis_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (destination / 'interpretation.md').write_text('# Saved Poincare periodicity study\n\n' + interpretation + '\n')
    return destination


def export_short_periodicity(directory: str | Path, section: SavedPoincareSection,
                             result: ShortPeriodicity, *, controls: dict[str, Any],
                             interpretation: str) -> Path:
    """Persist short-rhythm diagnostics without ranking long recurrence minima."""
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('summary', 'peaks', 'multiples', 'blocks', 'revolutions', 'thresholds'):
        records = getattr(result, name)
        if records:
            _write_records(destination / f'{name}.csv', records)
    correlation_rows = [dict(particle=pid, lag=int(lag), correlation=result.correlations[k, j])
                        for k, lag in enumerate(result.lags) for j, pid in enumerate(section.particle_ids)]
    _write_records(destination / 'correlations.csv', correlation_rows)
    metadata = dict(source=section.metadata, source_sha256=section.source_hashes,
                    controls=controls, selected_particle_ids=section.particle_ids,
                    sample_scope='Initial state and once-per-forcing-cycle returns only',
                    candidate_selection='First qualifying short autocorrelation peak',
                    geometric_clock='Monotone unwrapped angle about circular centroid; interpolated complete turns',
                    template='Sin/cos torus observables; first-half fit, second-half evaluation')
    (destination / 'analysis_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (destination / 'interpretation.md').write_text('# Short recurring rhythms\n\n' + interpretation + '\n')
    return destination
