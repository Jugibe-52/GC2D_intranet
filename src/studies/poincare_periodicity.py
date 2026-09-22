"""Finite-record periodicity diagnostics for checksum-verified Poincare returns."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.signal import find_peaks, periodogram

from ._trajectory_distances import periodic_particle_distances


@dataclass(frozen=True)
class SavedPoincareSection:
    """Original IDs and normalized coordinates, including the initial state.

    ``positions`` has shape (cycles + 1, particles, 2), in units of cell period L.
    Only integer forcing phases are available; these are not full trajectories.
    """

    particle_ids: tuple[int, ...]
    positions: np.ndarray
    colors: tuple[str, ...]
    initial_records: list[dict[str, Any]]
    metadata: dict[str, Any]
    source_hashes: dict[str, str]


@dataclass(frozen=True)
class PoincarePeriodicity:
    """Lag reductions, spectra and conclusions over the complete saved record."""

    lags: np.ndarray
    metrics: np.ndarray  # (lags, particles, 5): RMS, p95, max, first/second RMS.
    spectra: dict[str, tuple[np.ndarray, np.ndarray]]
    summary: list[dict[str, Any]]
    thresholds: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    best_defects: list[np.ndarray]
    initial_distances: np.ndarray
    threshold_fractions: tuple[float, ...]


def _digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _particle_ids(values: Sequence[int]) -> tuple[int, ...]:
    if not len(values) or any(isinstance(p, (bool, np.bool_)) or int(p) != p for p in values):
        raise ValueError('Select a nonempty sequence of integer particle IDs.')
    ids = tuple(int(p) for p in values)
    if len(set(ids)) != len(ids):
        raise ValueError('Particle IDs must be unique.')
    return ids


def load_saved_poincare_section(
    study_directory: str | Path, run_id: str, particle_ids: Sequence[int],
) -> SavedPoincareSection:
    """Read selected original IDs without requiring the full trajectory archive.

    Verify the CSV and metadata against COMPLETE.json and initial states against
    the original-particle manifest hash. Missing, repeated or misaligned samples
    are errors. Git LFS pointer files must first be materialized by the caller.
    """
    ids = _particle_ids(particle_ids)
    if not run_id or Path(run_id).name != run_id or run_id in ('.', '..'):
        raise ValueError('run_id must be a single directory name.')
    root = Path(study_directory)
    directory = root / 'resultados' / run_id
    complete = json.loads((directory / 'COMPLETE.json').read_text())
    if complete['run_id'] != run_id:
        raise ValueError('Completion manifest belongs to another run.')
    hashes = {}
    for name in ('metadata.json', 'positions_after_each_cycle.csv'):
        hashes[name] = _digest(directory / name)
        if hashes[name] != complete['sha256'][name]:
            raise ValueError(f'Checksum mismatch or unmaterialized Git LFS file: {name}')
    meta = json.loads((directory / 'metadata.json').read_text())
    if meta['run_id'] != run_id or not set(ids).issubset(meta['particle_ids']):
        raise ValueError('Requested run or particle IDs do not match saved metadata.')
    manifest = root / 'assets' / 'original_particles.json'
    hashes['original_particles.json'] = _digest(manifest)
    if hashes['original_particles.json'] != meta['original_particles_sha256']:
        raise ValueError('Original-particle manifest checksum mismatch.')
    original = {int(p['particle']): p for p in json.loads(manifest.read_text())['particles']}
    period = float(meta['field_provenance']['grid']['period'])
    if not np.isfinite(period) or period <= 0:
        raise ValueError('Cell period must be finite and positive.')
    cycles = int(meta['cycles'])
    points = np.full((cycles + 1, len(ids), 2), np.nan)
    seen = np.zeros((cycles + 1, len(ids)), dtype=bool)
    initial = []
    for j, pid in enumerate(ids):
        p = original[pid]
        points[0, j] = np.array([p['x'], p['y']]) / period
        seen[0, j] = True
        initial.append(dict(particle=pid, x_over_L=points[0, j, 0],
                            y_over_L=points[0, j, 1], radius_over_L=p['radius_over_L'],
                            color=p['color']))
    indices = {pid: j for j, pid in enumerate(ids)}
    with (directory / 'positions_after_each_cycle.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            pid = int(row['particle'])
            if pid not in indices:
                continue
            j, n = indices[pid], int(row['cycle'])
            if not 1 <= n <= cycles or seen[n, j]:
                raise ValueError('Saved cycles must occur exactly once per selected particle.')
            expected_time = meta['t0'] + n * meta['cycle_duration']
            if not np.isclose(float(row['time_normalized']), expected_time, rtol=0, atol=1e-10):
                raise ValueError('Saved times do not match the forcing-cycle grid.')
            coordinates = np.array([float(row['x_over_L']), float(row['y_over_L'])])
            wrapped = np.array([float(row['x_wrapped']), float(row['y_wrapped'])]) / period
            if (not np.isfinite(coordinates).all() or np.any((coordinates < 0) | (coordinates > 1))
                    or not np.allclose(coordinates, wrapped, rtol=0, atol=1e-12)):
                raise ValueError('Invalid normalized periodic coordinates.')
            if row['color'] != original[pid]['color'] or not np.isclose(
                    float(row['initial_radius_over_L']), original[pid]['radius_over_L'],
                    rtol=0, atol=1e-14):
                raise ValueError('Particle identity differs from the original manifest.')
            points[n, j] = coordinates
            seen[n, j] = True
    if not seen.all() or not np.isfinite(points).all():
        raise ValueError('Incomplete or nonfinite saved Poincare section.')
    return SavedPoincareSection(ids, points, tuple(original[p]['color'] for p in ids),
                               initial, meta, hashes)


def section_distances(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Minimum-image distances / L for matching (samples, particles, xy) arrays."""
    first, second = np.asarray(first), np.asarray(second)
    if first.shape != second.shape or first.ndim != 3 or first.shape[-1] != 2:
        raise ValueError('Expected matching (samples, particles, 2) arrays.')
    samples, particles, _ = first.shape
    # Reuse the GC component-major distance contract [x_1..x_N, y_1..y_N].
    packed_a = first.transpose(2, 1, 0).reshape(2 * particles, samples)
    packed_b = second.transpose(2, 1, 0).reshape(2 * particles, samples)
    return periodic_particle_distances(packed_a, packed_b, period=1.0).T


def _spectrum(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    angle = 2 * np.pi * points
    observables = np.concatenate((np.sin(angle), np.cos(angle)), axis=-1)
    # Remove the mean explicitly so constant orbits have exactly zero power.
    observables -= observables[:1]
    observables -= observables.mean(axis=0)
    frequency, power = periodogram(observables, fs=1.0, window='hann',
                                   detrend=False, axis=0, scaling='spectrum')
    return frequency, power.sum(axis=-1)


def _dominant_frequency(spectrum: tuple[np.ndarray, np.ndarray], particle: int) -> float | None:
    frequency, power = spectrum
    peak = 1 + int(np.argmax(power[1:, particle]))
    return float(frequency[peak]) if power[peak, particle] > 0 else None


def analyze_poincare_periodicity(
    positions: np.ndarray, particle_ids: Sequence[int], *, max_lag: int = 1000,
    minimum_repetitions: int = 5,
    threshold_fractions: Sequence[float] = (1e-4, 1e-3, 1e-2),
) -> PoincarePeriodicity:
    """Scan integer forcing lags and compare independent first/second windows.

    Coordinates are in units of L; row zero is the initial state. The best lag
    minimizes RMS over all overlapping pairs, not distance to the initial point.
    Threshold tests use the maximum defect and do not estimate numerical error.
    """
    ids = _particle_ids(particle_ids)
    z = np.asarray(positions, dtype=float)
    if z.ndim != 3 or z.shape[1:] != (len(ids), 2) or len(z) < 11 or not np.isfinite(z).all():
        raise ValueError('Require finite (cycles + 1, particles, 2) data with at least ten cycles.')
    for name, value, lower in (('max_lag', max_lag, 1), ('minimum_repetitions', minimum_repetitions, 3)):
        if isinstance(value, (bool, np.bool_)) or int(value) != value or value < lower:
            raise ValueError(f'{name} must be an integer >= {lower}.')
    thresholds = tuple(float(v) for v in threshold_fractions)
    if not thresholds or len(set(thresholds)) != len(thresholds) or any(
            not np.isfinite(v) or not 0 < v < .5 for v in thresholds):
        raise ValueError('Thresholds must be distinct fractions strictly between zero and 0.5.')
    cycles = len(z) - 1
    limit = min(int(max_lag), cycles // int(minimum_repetitions))
    if limit < 1:
        raise ValueError('The record is too short for minimum_repetitions.')
    lags = np.arange(1, limit + 1)
    half = cycles // 2
    metrics = np.empty((limit, len(ids), 5))
    for k, q in enumerate(lags):
        distances = section_distances(z[q:], z[:-q])
        first = section_distances(z[q:half + 1], z[:half + 1 - q])
        second = section_distances(z[half + q:], z[half:-q])
        metrics[k] = np.stack((np.sqrt(np.mean(distances**2, axis=0)),
                               np.quantile(distances, .95, axis=0), distances.max(axis=0),
                               np.sqrt(np.mean(first**2, axis=0)),
                               np.sqrt(np.mean(second**2, axis=0))), axis=-1)
    # Exclude t=0 for spectra; the two spectral windows have no shared samples.
    spectra = dict(full=_spectrum(z[1:]), first_half=_spectrum(z[1:half + 1]),
                   second_half=_spectrum(z[half + 1:]))
    initial_distances = section_distances(z, np.broadcast_to(z[:1], z.shape))
    summary, accepted, candidates, defects = [], [], [], []
    for j, pid in enumerate(ids):
        best = int(np.argmin(metrics[:, j, 0]))
        q = int(lags[best])
        closest = 1 + int(np.argmin(initial_distances[1:, j]))
        frequency = _dominant_frequency(spectra['full'], j)
        row = dict(particle=pid, best_q=q, rms_over_L=float(metrics[best, j, 0]),
                   p95_over_L=float(metrics[best, j, 1]), max_over_L=float(metrics[best, j, 2]),
                   first_half_rms=float(metrics[best, j, 3]), second_half_rms=float(metrics[best, j, 4]),
                   first_half_best_q=int(lags[np.argmin(metrics[:, j, 3])]),
                   second_half_best_q=int(lags[np.argmin(metrics[:, j, 4])]),
                   overlap_pairs=len(z) - q, record_repetitions=cycles / q,
                   closest_initial_return=closest,
                   closest_initial_distance_over_L=float(initial_distances[closest, j]),
                   peak_frequency=frequency, oscillation_cycles=1 / frequency if frequency else None,
                   first_half_frequency=_dominant_frequency(spectra['first_half'], j),
                   second_half_frequency=_dominant_frequency(spectra['second_half'], j))
        summary.append(row)
        for eps in thresholds:
            passing = lags[metrics[:, j, 2] <= eps]
            accepted.append(dict(particle=pid, threshold_over_L=eps,
                                 smallest_q=int(passing[0]) if len(passing) else None))
        minima = find_peaks(-metrics[:, j, 0])[0]
        ranked = sorted(set([0, best, limit - 1, *minima]), key=lambda k: metrics[k, j, 0])[:10]
        for k in ranked:
            candidates.append(dict(particle=pid, q=int(lags[k]), rms_over_L=float(metrics[k, j, 0]),
                                   max_over_L=float(metrics[k, j, 2])))
        defects.append(section_distances(z[q:, j:j + 1], z[:-q, j:j + 1])[:, 0])
    return PoincarePeriodicity(lags, metrics, spectra, summary, accepted, candidates,
                              defects, initial_distances, thresholds)


def periodicity_interpretation(result: PoincarePeriodicity) -> str:
    """Generate threshold-qualified statements without asserting exact periods."""
    lines = []
    for row in result.summary:
        accepted = [r for r in result.thresholds if r['particle'] == row['particle'] and r['smallest_q'] is not None]
        verdict = ('; '.join(f"smallest q={r['smallest_q']} at {r['threshold_over_L']:g} L" for r in accepted)
                   if accepted else 'No tested lag passes any selected maximum-distance threshold.')
        spectral = (f"Dominant spectral timescale: {row['oscillation_cycles']:.6g} cycles. "
                    if row['oscillation_cycles'] else 'No nonzero-frequency spectral power. ')
        lines.append(f"- **Particle {row['particle']}:** best approximate lag **{row['best_q']} cycles**; "
                     f"RMS {row['rms_over_L']:.6g} L; maximum {row['max_over_L']:.6g} L. "
                     + spectral + verdict)
    return '\n'.join(lines) + (
        '\n\nThese are finite-record diagnostics of saved numerical Poincare returns. '
        'A passing geometric threshold does not certify an exact physical period. '
        'Failure does not prove chaos or exclude a period beyond the searched range. '
        'Full within-cycle repetition and independently audited trajectory accuracy are not tested.')


@dataclass(frozen=True)
class ShortPeriodicity:
    """Short recurrent rhythms and persistence of a fixed integer-cycle pattern."""

    lags: np.ndarray
    correlations: np.ndarray  # (lags including zero, particles), torus observables.
    summary: list[dict[str, Any]]
    peaks: list[dict[str, Any]]
    multiples: list[dict[str, Any]]
    blocks: list[dict[str, Any]]
    revolutions: list[dict[str, Any]]
    thresholds: list[dict[str, Any]]


def _rotation_crossings(points: np.ndarray) -> tuple[np.ndarray, str]:
    """Interpolate complete turns in a localized, monotonically rotating section.

    Use a circular spatial centroid to tolerate a periodic-cell boundary. This
    geometric clock assumes that once-per-cycle sampling resolves angular motion;
    it is not a measurement of the full within-cycle physical orbit.
    """
    centre = np.angle(np.mean(np.exp(2j * np.pi * points), axis=0)) / (2 * np.pi)
    displacement = (points - centre + .5) % 1 - .5
    if (np.max(np.abs(displacement)) >= .45
            or np.min(np.linalg.norm(displacement, axis=1)) < 1e-10):
        return np.empty(0), 'No localized angular clock about the circular centroid.'
    phase = np.unwrap(np.arctan2(displacement[:, 1], displacement[:, 0]))
    direction = np.sign(phase[-1] - phase[0])
    turns = direction * (phase - phase[0]) / (2 * np.pi)
    if direction == 0 or np.any(np.diff(turns) <= 0) or turns[-1] < 3:
        return np.empty(0), 'Angular motion is not resolved as strictly monotone complete turns.'
    levels = np.arange(1, int(np.floor(turns[-1])) + 1)
    return np.interp(levels, turns, np.arange(len(points))), 'Resolved monotone angular clock.'


def analyze_short_periodicity(
    positions: np.ndarray, particle_ids: Sequence[int], *, max_lag: int = 80,
    candidate_max_lag: int = 20, peak_height: float = .5, peak_prominence: float = .2,
    multiple_count: int = 20, block_cycles: int = 500,
    threshold_fractions: Sequence[float] = (1e-3, 5e-3, 1e-2, 2e-2),
) -> ShortPeriodicity:
    """Measure recurring short rhythms without minimizing a long-lag defect.

    The first qualifying positive autocorrelation peak is the short integer
    candidate. Subsequent peaks, defects at its multiples, disjoint temporal
    blocks and a held-out residue template quantify persistence. Correlation is
    the normalized inner product of centered sin/cos spatial observables over
    overlapping samples. Coordinate errors retain the minimum-image convention.
    """
    ids = _particle_ids(particle_ids)
    z = np.asarray(positions, dtype=float)
    if z.ndim != 3 or z.shape[1:] != (len(ids), 2) or len(z) < 21 or not np.isfinite(z).all():
        raise ValueError('Require finite (cycles + 1, particles, 2) data with at least 20 cycles.')
    for name, value, lower in (('max_lag', max_lag, 3), ('candidate_max_lag', candidate_max_lag, 2),
                               ('multiple_count', multiple_count, 2), ('block_cycles', block_cycles, 10)):
        if isinstance(value, (bool, np.bool_)) or int(value) != value or value < lower:
            raise ValueError(f'{name} must be an integer >= {lower}.')
    cycles = len(z) - 1
    max_lag, candidate_max_lag, multiple_count, block_cycles = map(
        int, (max_lag, candidate_max_lag, multiple_count, block_cycles))
    if (not candidate_max_lag < max_lag < cycles or block_cycles > cycles
            or candidate_max_lag > cycles // 4):
        raise ValueError('Require candidate_max_lag < max_lag < cycles, at least four candidate repetitions, '
                         'and block_cycles <= cycles.')
    if not np.isfinite([peak_height, peak_prominence]).all() or not 0 < peak_height < 1 or not 0 < peak_prominence < 2:
        raise ValueError('Peak height must be in (0,1) and prominence in (0,2).')
    thresholds = tuple(float(eps) for eps in threshold_fractions)
    if not thresholds or any(not np.isfinite(eps) or not 0 < eps < .5 for eps in thresholds):
        raise ValueError('Threshold fractions must lie in (0,0.5).')
    obs = np.concatenate((np.sin(2 * np.pi * z), np.cos(2 * np.pi * z)), axis=-1)
    centered = obs - obs[:1]
    centered -= centered.mean(axis=0)
    correlations = np.full((max_lag + 1, len(ids)), np.nan)
    for scan_lag in range(max_lag + 1):
        a, b = (centered[scan_lag:], centered[:-scan_lag]) if scan_lag else (centered, centered)
        norm = np.sqrt(np.sum(a*a, axis=(0, 2)) * np.sum(b*b, axis=(0, 2)))
        np.divide(np.sum(a*b, axis=(0, 2)), norm, out=correlations[scan_lag], where=norm > 0)
    summary, peaks, multiples, blocks, revolutions, accepted = [], [], [], [], [], []
    for j, pid in enumerate(ids):
        locations, _ = find_peaks(correlations[:, j], height=peak_height, prominence=peak_prominence)
        eligible = locations[locations <= candidate_max_lag]
        q = int(eligible[0]) if len(eligible) else None
        for k, lag in enumerate(locations):
            peaks.append(dict(particle=pid, peak_number=k + 1, lag=int(lag),
                              correlation=float(correlations[lag, j]),
                              interval_from_previous=int(lag - locations[k - 1]) if k else None))
        crossings, clock_status = _rotation_crossings(z[:, j])
        periods = np.diff(crossings)
        for k, duration in enumerate(periods):
            revolutions.append(dict(particle=pid, start_cycle=float(crossings[k]),
                                    end_cycle=float(crossings[k + 1]), duration_cycles=float(duration)))
        row: dict[str, Any] = dict(
            particle=pid, short_q=q, correlation=float(correlations[q, j]) if q else None,
            mean_rotation_cycles=float(periods.mean()) if len(periods) else None,
            rotation_q25=float(np.quantile(periods, .25)) if len(periods) else None,
            rotation_q75=float(np.quantile(periods, .75)) if len(periods) else None,
            measured_revolutions=len(periods), angular_clock_status=clock_status,
            rms_over_L=None, max_over_L=None, heldout_template_score=None,
            block_period_min=None, block_period_max=None,
        )
        if q is not None:
            defect = section_distances(z[q:, j:j+1], z[:-q, j:j+1])[:, 0]
            row.update(rms_over_L=float(np.sqrt(np.mean(defect**2))), max_over_L=float(defect.max()))
            for eps in thresholds:
                accepted.append(dict(particle=pid, q=q, threshold_over_L=eps,
                                     fraction_of_pairs_within=float(np.mean(defect <= eps))))
            # Train a fixed q-cycle pattern on the first half; evaluate the second.
            # This tests persistence of the candidate, not an independently selected period.
            split = len(z) // 2
            labels = np.arange(len(z)) % q
            training, testing = obs[:split, j], obs[split:, j]
            template = np.stack([training[labels[:split] == r].mean(axis=0) for r in range(q)])
            baseline = np.sum((testing - training.mean(axis=0))**2)
            if baseline > 0:
                row['heldout_template_score'] = float(1 - np.sum((testing - template[labels[split:]])**2) / baseline)
            for k in range(1, min(multiple_count, cycles // (2*q)) + 1):
                lag = k*q
                d = section_distances(z[lag:, j:j+1], z[:-lag, j:j+1])[:, 0]
                multiples.append(dict(particle=pid, multiple=k, lag=lag,
                                      rms_over_L=float(np.sqrt(np.mean(d*d))), max_over_L=float(d.max())))
        for start in range(0, cycles, block_cycles):
            stop = min(start + block_cycles, cycles)
            # Block positions are disjoint; comparisons never cross a block boundary.
            block = z[start:stop, j:j+1]
            if len(block) <= max_lag:
                continue
            selected = (crossings[:-1] >= start) & (crossings[1:] < stop)
            local = periods[selected]
            if q is not None:
                d = section_distances(block[q:], block[:-q])[:, 0]
                a, b = centered[start+q:stop, j], centered[start:stop-q, j]
                corr = float(np.sum(a*b) / np.sqrt(np.sum(a*a)*np.sum(b*b)))
            else:
                d, corr = np.array([np.nan]), float('nan')
            blocks.append(dict(particle=pid, start_cycle=start, stop_cycle_exclusive=stop,
                               q=q, correlation=corr, rms_over_L=float(np.sqrt(np.mean(d*d))),
                               max_over_L=float(np.max(d)),
                               mean_rotation_cycles=float(local.mean()) if len(local) else None))
        local_periods = [b['mean_rotation_cycles'] for b in blocks
                         if b['particle'] == pid and b['mean_rotation_cycles'] is not None]
        if local_periods:
            row.update(block_period_min=min(local_periods), block_period_max=max(local_periods))
        summary.append(row)
    return ShortPeriodicity(np.arange(max_lag+1), correlations, summary, peaks, multiples,
                            blocks, revolutions, accepted)


def short_periodicity_interpretation(result: ShortPeriodicity) -> str:
    """Describe recurrent short rhythms, distinguishing them from fixed repetition."""
    lines = []
    for row in result.summary:
        if row['short_q'] is None:
            lines.append(f"- **Particle {row['particle']}:** no qualifying short autocorrelation peak.")
            continue
        clock = (f"Mean geometric revolution: **{row['mean_rotation_cycles']:.4g} cycles**, "
                 f"with block means {row['block_period_min']:.4g}–{row['block_period_max']:.4g}. "
                 if row['mean_rotation_cycles'] is not None and row['block_period_min'] is not None else '')
        score = row['heldout_template_score']
        persistence = (f"A fixed {row['short_q']}-cycle template trained on the first half has "
                       f"a second-half score of **{score:.4f}** (1 is perfect; 0 matches a constant baseline). "
                       if score is not None else '')
        lines.append(f"- **Particle {row['particle']}:** recurrent rhythm near **{row['short_q']} cycles**, "
                     f"autocorrelation {row['correlation']:.5f}. " + clock + persistence)
    return '\n'.join(lines) + (
        '\n\nRepeated autocorrelation peaks establish a recurring numerical pattern at short lags. '
        'A stable noninteger rhythm can gradually slip relative to a fixed integer-cycle template. '
        'High template agreement with bounded multiple-lag defects supports a persistent integer-cycle '
        'structure, with residual modulation. These diagnostics do not establish an exact physical orbit period; '
        'the geometric clock uses interpolated once-per-cycle section positions.')
