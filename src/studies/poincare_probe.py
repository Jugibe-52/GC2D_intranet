"""RK4 probe trajectories inside gaps of an existing Poincare section."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path
import time
from typing import Callable
import zipfile

import numpy as np
import scipy
from scipy.spatial import cKDTree

from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Grid, Potential
from contracts.problem import InitialValueProblem
from methods.classical.rk4 import RK4
from contracts.request import SimulationRequest
from simulation.runner import simulate
from studies.poincare_periodicity import SavedPoincareSection


@dataclass(frozen=True)
class RK4ProbeSettings:
    """One uncoupled GC particle; initial coordinates are fractions of period L."""

    particle_id: int
    initial_xy_over_L: tuple[float, float]
    color: str
    cycles: int
    steps_per_cycle: int
    chunk_steps: int
    rho: float
    t0: float = 0.0
    cycle_duration: float = 1.0

    def __post_init__(self) -> None:
        for name in ('particle_id', 'cycles', 'steps_per_cycle', 'chunk_steps'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        xy = np.asarray(self.initial_xy_over_L, dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all() or np.any((xy < 0) | (xy >= 1)):
            raise ValueError('Initial coordinates must be two finite fractions in [0, 1).')
        if self.chunk_steps % self.steps_per_cycle:
            raise ValueError('Chunk boundaries must coincide with complete forcing cycles.')
        if not np.isfinite([self.rho, self.t0, self.cycle_duration]).all() or self.rho < 0 or self.cycle_duration <= 0:
            raise ValueError('Require finite time controls, rho >= 0 and cycle_duration > 0.')
        if not isinstance(self.color, str) or len(self.color) != 7 or self.color[0] != '#':
            raise ValueError('Use a six-digit hexadecimal particle color.')
        int(self.color[1:], 16)


def load_probe_field(snapshot: Path, *, expected_sha256: str,
                     physical_settings: dict) -> tuple[Potential, dict]:
    """Load only field arrays from a verified snapshot, using the current public API.

    The frozen Python code is never imported. Its identity and the current source
    hashes are recorded separately so equal physical inputs do not imply equal
    software versions.
    """
    content = Path(snapshot).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError('Field snapshot checksum mismatch or unmaterialized Git LFS file.')
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        provenance = json.loads(archive.read('provenance.json'))
        for key, expected in physical_settings.items():
            if provenance.get(key) != expected:
                raise ValueError(f'Field setting differs from the explicit notebook control: {key}')
        with np.load(io.BytesIO(archive.read('field.npz')), allow_pickle=False) as arrays:
            potential = Potential(Grid(**provenance['grid']), mean=arrays['mean'],
                                  modes=arrays['modes'], frequencies=arrays['frequencies'],
                                  interpolation_order=provenance['interpolation_order'])
    return potential, provenance


def probe_contract(settings: RK4ProbeSettings, *, snapshot_sha256: str,
                   provenance: dict, background: SavedPoincareSection) -> dict:
    """Validate context compatibility and identify the reproducible calculation."""
    meta = background.metadata
    if settings.particle_id in background.particle_ids:
        raise ValueError('The probe ID must be absent from the background.')
    if meta['method'] != 'RK4' or meta['snapshot_sha256'] != snapshot_sha256:
        raise ValueError('The context must be RK4 with the same field snapshot.')
    for key in ('rho', 't0', 'cycle_duration', 'steps_per_cycle'):
        if meta[key] != getattr(settings, key):
            raise ValueError(f'Probe and background disagree on {key}.')
    if settings.chunk_steps != meta['checkpoint_steps']:
        raise ValueError('Use the same integration chunk schedule as the background.')
    if settings.cycles > meta['cycles']:
        raise ValueError('The background does not cover the requested cycle count.')
    for key in ('grid', 'source_hdf5_sha256', 'B_tesla', 'characteristic_length_m',
                'source_selection', 'interpolation_order'):
        if provenance[key] != meta['field_provenance'][key]:
            raise ValueError(f'The background field differs on {key}.')
    source = Path(__file__).resolve().parents[1]
    paths = [Path(__file__)]
    for package in ('potential', 'dynamics', 'initial_conditions', 'simulation'):
        paths.extend(sorted((source / package).rglob('*.py')))
    hashes = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in paths}
    return dict(method='RK4', settings=asdict(settings), snapshot_sha256=snapshot_sha256,
                field_provenance=provenance, source_sha256=hashes,
                numpy_version=np.__version__, scipy_version=scipy.__version__,
                background_run_id=meta['run_id'], background_sha256=background.source_hashes,
                accuracy_reference_computed=False)


def integrate_rk4_probe(potential: Potential, settings: RK4ProbeSettings, *,
                        progress: Callable[[int, int], None] | None = None) -> dict:
    """Integrate only the new particle, saving every accepted step in float64.

    Coordinates have shape (steps + 1, 1, 2) in normalized physical units,
    unwrapped. Chunking reproduces the saved background's public-RK4 schedule;
    independent particles do not require reintegrating the 48 context orbits.
    """
    count = settings.cycles * settings.steps_per_cycle
    step = settings.cycle_duration / settings.steps_per_cycle
    times = settings.t0 + np.arange(count + 1) * step
    xy = np.empty((count + 1, 1, 2), dtype=np.float64)
    xy[0, 0] = np.asarray(settings.initial_xy_over_L) * potential.grid.period
    dynamics = GuidingCenterDynamics(potential, rho=settings.rho)
    started = time.perf_counter()
    for start in range(0, count, settings.chunk_steps):
        end = min(count, start + settings.chunk_steps)
        initial = GCInitialConfiguration.from_components(x=xy[start, :, 0], y=xy[start, :, 1])
        request = SimulationRequest.uniform(t_span=(times[start], times[end]),
                                           max_step=step, sample_count=end - start + 1)
        solution = simulate(InitialValueProblem(dynamics, initial), RK4(), request)
        if solution.diagnostics['step_count'] != end - start:
            raise RuntimeError('RK4 did not preserve the requested number of steps.')
        xy[start:end + 1, 0] = solution.states.T
        if progress is not None:
            progress(end, count)
    return dict(times=times, xy=xy, runtime_seconds=time.perf_counter() - started,
                step_count=count)


def probe_section(result: dict, settings: RK4ProbeSettings, period: float) -> np.ndarray:
    """Return normalized, wrapped cycle samples, including the initial state."""
    xy = np.asarray(result['xy'])
    count = settings.cycles * settings.steps_per_cycle + 1
    if xy.shape != (count, 1, 2) or not np.isfinite(xy).all():
        raise ValueError('Probe trajectory shape or values are invalid.')
    expected = settings.t0 + np.arange(count) * settings.cycle_duration / settings.steps_per_cycle
    if not np.allclose(result['times'], expected, atol=1e-10, rtol=0):
        raise ValueError('Probe trajectory times are not on the required grid.')
    if not np.allclose(xy[0, 0] / period, settings.initial_xy_over_L, atol=1e-14, rtol=0):
        raise ValueError('Probe trajectory does not start at the configured position.')
    return (xy[::settings.steps_per_cycle] / period) % 1.0


def probe_clearance(background: SavedPoincareSection, points: np.ndarray) -> np.ndarray:
    """Distance / L to the closest saved context return, using periodic geometry.

    A finite-record clearance is not evidence that the continuous region is
    inaccessible, nor a certified trajectory-error bound.
    """
    tree = cKDTree(background.positions[1:].reshape(-1, 2) % 1, boxsize=1.0)
    return tree.query(np.asarray(points).reshape(-1, 2) % 1)[0]


def probe_comparison_panel(background: SavedPoincareSection, positions: np.ndarray,
                           settings: RK4ProbeSettings) -> dict:
    """Append the new particle to unchanged RK4 context for a single HTML panel."""
    if positions.shape != (settings.cycles + 1, 1, 2):
        raise ValueError('Unexpected probe section shape.')
    if len(background.positions) < len(positions) or settings.particle_id in background.particle_ids:
        raise ValueError('The background must cover all cycles and use distinct IDs.')
    return dict(title=f'RK4 — {len(background.particle_ids)} original particles + particle {settings.particle_id}, '
                      f'{settings.steps_per_cycle} steps/cycle',
                coordinates=np.concatenate((background.positions[1:settings.cycles + 1], positions[1:]), axis=1),
                particle_ids=[*background.particle_ids, settings.particle_id],
                colors=[*background.colors, settings.color])
