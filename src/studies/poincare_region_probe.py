"""Matched RK4 and BM4Midpoint probe batches for sampled Poincare gaps."""

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import time
from typing import Callable

import numpy as np
import scipy

from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from methods.extended.bm4 import BM4Midpoint
from contracts.problem import InitialValueProblem
from methods.classical.rk4 import RK4
from contracts.request import SimulationRequest
from simulation.runner import simulate
from studies.poincare_periodicity import SavedPoincareSection
from studies.poincare_probe import RK4ProbeSettings


@dataclass(frozen=True)
class RegionProbeSettings:
    """Identical geometric seeds and cycle sampling for one numerical method."""

    method: str
    particle_ids: tuple[int, ...]
    initial_xy_over_L: tuple[tuple[float, float], ...]
    colors: tuple[str, ...]
    cycles: int
    steps_per_cycle: int
    chunk_steps: int
    rho: float
    t0: float = 0.0
    cycle_duration: float = 1.0
    coupling_frequency: float | None = None

    def __post_init__(self) -> None:
        if self.method not in ('RK4', 'BM4Midpoint'):
            raise ValueError('Supported probe methods are RK4 and BM4Midpoint.')
        if (not self.particle_ids or len(set(self.particle_ids)) != len(self.particle_ids)
                or len(self.initial_xy_over_L) != len(self.particle_ids)
                or len(self.colors) != len(self.particle_ids)):
            raise ValueError('Require matching, nonempty and distinct particle IDs, positions and colors.')
        for pid, xy, color in zip(self.particle_ids, self.initial_xy_over_L, self.colors):
            RK4ProbeSettings(pid, xy, color, self.cycles, self.steps_per_cycle,
                             self.chunk_steps, self.rho, self.t0, self.cycle_duration)
        if self.method == 'BM4Midpoint':
            if self.coupling_frequency is None or not np.isfinite(self.coupling_frequency) or self.coupling_frequency < 0:
                raise ValueError('BM4Midpoint requires an explicit finite nonnegative coupling frequency.')
        elif self.coupling_frequency is not None:
            raise ValueError('RK4 does not use doubled-copy coupling.')


def region_probe_contract(settings: RegionProbeSettings, *, snapshot_sha256: str,
                          provenance: dict, background: SavedPoincareSection) -> dict:
    """Verify matching physical inputs and identify the reproducible calculation."""
    meta = background.metadata
    if set(settings.particle_ids).intersection(background.particle_ids):
        raise ValueError('Probe IDs must be absent from the background.')
    if meta['method'] != settings.method or meta['snapshot_sha256'] != snapshot_sha256:
        raise ValueError('The context must use the same method and field snapshot.')
    keys = ['rho', 't0', 'cycle_duration', 'steps_per_cycle']
    if settings.method == 'BM4Midpoint':
        keys.append('coupling_frequency')
    for key in keys:
        if meta[key] != getattr(settings, key):
            raise ValueError(f'Probe and background disagree on {key}.')
    if settings.chunk_steps != meta['checkpoint_steps'] or settings.cycles > meta['cycles']:
        raise ValueError('The checkpoint schedule or cycle coverage differs from the background.')
    for key in ('grid', 'source_hdf5_sha256', 'B_tesla', 'characteristic_length_m',
                'source_selection', 'interpolation_order'):
        if provenance[key] != meta['field_provenance'][key]:
            raise ValueError(f'The background field differs on {key}.')
    source = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), source / 'studies/poincare_probe.py']
    for package in ('potential', 'dynamics', 'initial_conditions', 'simulation'):
        paths.extend(sorted((source / package).rglob('*.py')))
    hashes = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in paths}
    clock = ('Global fixed grid t0 + k*h' if settings.method == 'BM4Midpoint'
             else 'Public uniform grids within fixed chunks, matching the saved RK4 run')
    return dict(method=settings.method, settings=asdict(settings), snapshot_sha256=snapshot_sha256,
                field_provenance=provenance, source_sha256=hashes,
                numpy_version=np.__version__, scipy_version=scipy.__version__,
                background_run_id=meta['run_id'], background_sha256=background.source_hashes,
                clock=clock, accuracy_reference_computed=False)


def integrate_region_probes(potential: Potential, settings: RegionProbeSettings, *,
                            progress: Callable[[int, int], None] | None = None) -> dict:
    """Advance uncoupled probes together through the existing public methods.

    Return every accepted physical position in float64 with shape
    (steps + 1, probes, 2), unwrapped. BM4Midpoint keeps the original global clock
    and arithmetic projection after each twelve-stage step; it has no Newton
    solve. RK4 retains its saved context's chunk schedule.
    """
    count = settings.cycles * settings.steps_per_cycle
    size = len(settings.particle_ids)
    step = settings.cycle_duration / settings.steps_per_cycle
    times = settings.t0 + np.arange(count + 1) * step
    xy = np.empty((count + 1, size, 2), dtype=np.float64)
    xy[0] = np.asarray(settings.initial_xy_over_L) * potential.grid.period
    dynamics = GuidingCenterDynamics(potential, rho=settings.rho)
    batch = count if settings.method == 'BM4Midpoint' else settings.chunk_steps
    started = time.perf_counter()
    diagnostics = {}
    for start in range(0, count, batch):
        end = min(count, start + batch)
        initial = GCInitialConfiguration.from_components(x=xy[start, :, 0], y=xy[start, :, 1])
        request = SimulationRequest.uniform(t_span=(times[start], times[end]),
                                           max_step=step, sample_count=end - start + 1)
        if settings.method == 'BM4Midpoint':
            # Progress is observational only; it neither changes the map nor its clock.
            def report(event):
                completed = event.step_index + 1
                if progress is not None and (completed % settings.chunk_steps == 0 or completed == count):
                    progress(completed, count)
            method = BM4Midpoint(coupling_frequency=settings.coupling_frequency,
                                 step_observer=report if progress else None)
        else:
            method = RK4()
        solution = simulate(InitialValueProblem(dynamics, initial), method, request)
        if solution.diagnostics['step_count'] != end - start or solution.diagnostics['output_interpolation_count'] != 0:
            raise RuntimeError('The method did not preserve the requested complete-step grid.')
        xy[start:end + 1] = np.stack(solution.positions(), axis=-1).transpose(1, 0, 2)
        if settings.method == 'RK4' and progress:
            progress(end, count)
        diagnostics = solution.diagnostics
    result = dict(times=times, xy=xy, runtime_seconds=time.perf_counter() - started,
                  step_count=count)
    if settings.method == 'BM4Midpoint':
        expected = dict(composition_stage_count=12, projection_kind='arithmetic_mean',
                        nonlinear_unknown_dimension=0, vector_field_evaluations_per_step=24,
                        state_extension='physical', track_energy=False)
        for key, value in expected.items():
            if diagnostics[key] != value:
                raise RuntimeError(f'Unexpected BM4Midpoint diagnostic: {key}')
        separation = np.asarray(diagnostics['copy_separation_norms'])
        if separation.shape != (count,) or not np.isfinite(separation).all() or np.any(separation < 0):
            raise RuntimeError('Invalid BM4Midpoint copy-separation history.')
        result.update(copy_separation_norms=separation,
                      method_diagnostics={**expected, 'coupling_frequency': settings.coupling_frequency,
                          'copy_separation_scope': 'Infinity norm over all coordinates of this probe batch'})
    return result


def region_probe_section(result: dict, settings: RegionProbeSettings, period: float) -> np.ndarray:
    """Select once-per-cycle wrapped coordinates / L, including initial states."""
    count = settings.cycles * settings.steps_per_cycle + 1
    xy = np.asarray(result['xy'])
    if xy.shape != (count, len(settings.particle_ids), 2) or not np.isfinite(xy).all():
        raise ValueError('Invalid probe trajectory shape or values.')
    expected = settings.t0 + np.arange(count) * settings.cycle_duration / settings.steps_per_cycle
    if np.shape(result['times']) != (count,) or not np.allclose(result['times'], expected, atol=1e-10, rtol=0):
        raise ValueError('Probe times are not on the required cycle grid.')
    if not np.allclose(xy[0] / period, settings.initial_xy_over_L, atol=1e-14, rtol=0):
        raise ValueError('Probe initial positions differ from the configured geometry.')
    return (xy[::settings.steps_per_cycle] / period) % 1.0


def region_probe_panel(background: SavedPoincareSection,
                       batches: list[tuple[RegionProbeSettings, dict]]) -> dict:
    """Merge compatible probe batches and preserve every original context ID."""
    if not batches:
        raise ValueError('At least one probe batch is required.')
    settings = batches[0][0]
    if settings.cycles > background.metadata['cycles']:
        raise ValueError('The background does not cover the probe record.')
    coordinates = [background.positions[1:settings.cycles + 1]]
    ids, colors = list(background.particle_ids), list(background.colors)
    for controls, result in batches:
        for key in ('method', 'cycles', 'steps_per_cycle', 'rho', 't0', 'cycle_duration', 'coupling_frequency'):
            if getattr(controls, key) != getattr(settings, key):
                raise ValueError(f'Probe batches disagree on {key}.')
        if controls.method != background.metadata['method'] or set(ids).intersection(controls.particle_ids):
            raise ValueError('Probe method or IDs conflict with the context.')
        coordinates.append(region_probe_section(result, controls, background.metadata['field_provenance']['grid']['period'])[1:])
        ids.extend(controls.particle_ids)
        colors.extend(controls.colors)
    return dict(title=f'{settings.method} — {len(ids)} particles, {settings.steps_per_cycle} steps/cycle',
                coordinates=np.concatenate(coordinates, axis=1), particle_ids=ids, colors=colors)
