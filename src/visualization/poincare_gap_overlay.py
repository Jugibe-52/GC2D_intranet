"""Append verified gap probes to viewer copies without changing star archives."""

from typing import Mapping

import numpy as np

from diagnostics.persistence import StoredSolution
from studies.poincare_gap_probes import GapProbeSeeds, validate_saved_gap_probes
from studies.poincare_rho_sweep import folded_rho_positions


# Fixed colors belong to probe order; the original star keeps its own palette.
_PROBE_COLORS = ('#111827', '#7c3aed', '#e6007e', '#92400e',
                 '#0891b2', '#0f766e', '#d97706', '#be123c')


def validate_gap_probe_overlay(
    runs: Mapping[float, StoredSolution], probe_runs: Mapping[float, StoredSolution], *,
    expected_seeds: GapProbeSeeds | None = None,
) -> GapProbeSeeds:
    """Require aligned saved runs and the same frozen probe seeds at every rho."""
    if not probe_runs or set(probe_runs) != set(runs):
        raise ValueError('Gap probes must cover exactly the available background rho values.')
    seeds = expected_seeds
    for rho, background in runs.items():
        current = validate_saved_gap_probes(probe_runs[rho], background=background)
        if seeds is None:
            seeds = current
        if current != seeds:
            raise ValueError('Gap probe positions, IDs, regions and selection rho must remain fixed.')
        if not np.isclose(probe_runs[rho].metadata['config']['rho_hat'], rho, rtol=0, atol=1e-14):
            raise ValueError('The gap probe archive rho does not match its viewer label.')
    assert seeds is not None
    return seeds


def gap_probe_presentation(seeds: GapProbeSeeds) -> dict:
    """Assign stable labels, colors and region groups to the eight added probes."""
    ids = np.asarray(seeds.particle_ids, dtype=int)
    if len(ids) != len(_PROBE_COLORS):
        raise ValueError('The gap overlay requires eight explicit probe particles.')
    groups = [dict(label='All gap probes', particle_ids=ids)]
    labels = {}
    for gap in dict.fromkeys(seeds.gap_names):
        gap_ids = [pid for pid, name in zip(ids, seeds.gap_names) if name == gap]
        groups.append(dict(label=f'Gap: {gap}', particle_ids=gap_ids))
        for index, pid in enumerate(gap_ids, 1):
            labels[int(pid)] = f'#{pid} · {gap} · probe {index}'
    return dict(particle_ids=ids, colors=list(_PROBE_COLORS), labels=labels, groups=groups,
                description=(f'Gap probes #{ids[0]}–#{ids[-1]} use explicit seeds selected at '
                             f'rho = {seeds.selection_rho:.2f} and reused unchanged at every rho '
                             'and for every method. Their saved trajectories are appended for '
                             'display; the original star archives are unchanged. '))


def append_gap_probe_coordinates(
    coordinates: np.ndarray, probes: StoredSolution, *, fold_to_cell: bool,
) -> np.ndarray:
    """Join particle axes of aligned viewer arrays in the requested coordinates."""
    if fold_to_cell:
        _, xy = folded_rho_positions(probes.solution, probes.metadata)
    else:
        x, y = probes.solution.positions()
        xy = np.stack((x.T, y.T), axis=-1)
    return np.concatenate((coordinates, xy), axis=1)


__all__ = ['validate_gap_probe_overlay', 'gap_probe_presentation',
           'append_gap_probe_coordinates']
