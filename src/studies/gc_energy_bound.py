"""Finite-horizon energy-envelope study for one measured guiding-center orbit."""

from dataclasses import asdict, dataclass, replace
from typing import Any
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from diagnostics import GCGeneralizedEnergyObserver, StoredReferenceTrajectory
from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import BM4Implicit, RK4, InitialValueProblem, NumericalMethod, SimulationRequest, simulate
from ._trajectory_distances import particle_distances
from ._validation import integer_ratio
from .reference_trajectory import potential_fingerprint


METHODS = ("BM4Implicit", "RK4")


@dataclass(frozen=True)
class GCEnergyBoundConfig:
    """Explicit physical, resolution and nonlinear controls; time is normalized."""

    t_span: tuple[float, float] = (0.0, 200.0)
    steps: tuple[float, ...] = (0.1, 0.05, 0.025)
    horizons: tuple[float, ...] = (1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0)
    rho: float = 0.3
    coupling_frequency: float = 0.0
    newton_atol: float = 1e-12
    newton_rtol: float = 1e-11
    newton_max_iterations: int = 40
    jacobian_relative_step: float = float(np.cbrt(np.finfo(float).eps))
    timing_repeats: int = 3
    block_count: int = 10
    plateau_relative_growth: float = 0.10
    progress: bool = True

    def __post_init__(self) -> None:
        """Reject grids that would change the effective step or miss a horizon."""
        start, stop = self.t_span
        if not np.isfinite([start, stop]).all() or stop <= start:
            raise ValueError("t_span must be finite and increasing.")
        if (len(self.steps) < 3 or not np.isfinite(self.steps).all()
                or min(self.steps) <= 0 or np.any(np.diff(self.steps) >= 0)):
            raise ValueError("Use at least three distinct, decreasing positive steps.")
        if (len(self.horizons) < 2 or not np.isfinite(self.horizons).all()
                or min(self.horizons) <= 0 or np.any(np.diff(self.horizons) <= 0)
                or not np.isclose(self.horizons[-1], stop-start)):
            raise ValueError("Horizons must increase and end at the integration duration.")
        for step in self.steps:
            integer_ratio(stop-start, step, "duration / step")
            integer_ratio(step, self.steps[-1], "step / finest step")
            for horizon in self.horizons:
                integer_ratio(horizon, step, "horizon / step")
        for name in ("newton_atol", "newton_rtol", "jacobian_relative_step"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("rho", "coupling_frequency", "plateau_relative_growth"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        for name, minimum in (("timing_repeats", 3), ("block_count", 2), ("newton_max_iterations", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")


@dataclass
class GCEnergyBoundResult:
    """Every accepted node and independently auditable summary records."""

    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]
    summary: list[dict[str, Any]]
    envelopes: list[dict[str, Any]]
    blocks: list[dict[str, Any]]
    orders: list[dict[str, Any]]


def time_rms(values: np.ndarray, times: np.ndarray) -> float:
    """Time-integrated RMS on the actual saved grid, including endpoints."""
    integrate = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(np.sqrt(integrate(np.asarray(values)**2, times) / (times[-1]-times[0])))


def envelope_statistics(
    times: np.ndarray, errors: np.ndarray, horizons: tuple[float, ...],
) -> list[dict[str, float | None]]:
    """Include every accepted node in each prefix; never interpolate a peak."""
    if (times.ndim != 1 or times.size < 2 or errors.shape != times.shape
            or not np.isfinite(times).all() or not np.isfinite(errors).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError("Envelope inputs must be finite, aligned and increasing.")
    maxima = np.maximum.accumulate(np.abs(errors))
    records: list[dict[str, float | None]] = []
    previous: float | None = None
    for horizon in horizons:
        target = times[0]+horizon
        index = int(np.argmin(np.abs(times-target)))
        if not np.isclose(times[index], target, rtol=0, atol=1e-11*max(1, abs(target))):
            raise ValueError("Every requested horizon must be an accepted node.")
        value = float(maxima[index])
        ratio = value/previous if previous is not None and previous > 0 else None
        records.append({"horizon": float(horizon), "maximum": value, "growth_ratio": ratio})
        previous = value
    return records


def matching_reference_nodes(times: np.ndarray, reference_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return coincident method/reference indices without interpolating either orbit."""
    right = np.searchsorted(reference_times, times).clip(0, len(reference_times)-1)
    left = (right-1).clip(0)
    nearest = np.where(np.abs(reference_times[left]-times) < np.abs(reference_times[right]-times), left, right)
    selected = np.flatnonzero(np.isclose(times, reference_times[nearest], rtol=0, atol=1e-12))
    if selected.size < 2 or selected[0] != 0 or selected[-1] != len(times)-1:
        raise ValueError("Method and reference must share both endpoints and at least two nodes.")
    return selected, nearest[selected]


def _validate_reference(
    reference: StoredReferenceTrajectory, dynamics: GuidingCenterDynamics,
    initial: np.ndarray, config: GCEnergyBoundConfig,
) -> None:
    """Reject stale references from another field, orbit, radius or time grid."""
    if initial.shape != (2,):
        raise ValueError("This study requires exactly one guiding-center trajectory.")
    if not np.allclose(reference.times[[0, -1]], config.t_span, rtol=0, atol=1e-12):
        raise ValueError("The reference must cover the requested interval exactly.")
    for step in config.steps:
        count = integer_ratio(config.t_span[1]-config.t_span[0], step, "duration / step")
        times = np.linspace(*config.t_span, count+1)
        indices, _ = matching_reference_nodes(times, reference.times)
        for horizon in config.horizons:
            if not np.any(np.isclose(times[indices], config.t_span[0]+horizon, rtol=0, atol=1e-12)):
                raise ValueError("Every requested horizon must be a shared reference node.")
    if not np.array_equal(reference.initial_state, initial):
        raise ValueError("Reference initial state differs from the requested trajectory.")
    if reference.metadata["dynamics_fingerprint_sha256"] != potential_fingerprint(dynamics.effective_potential):
        raise ValueError("Reference gyroaveraged field fingerprint does not match.")
    if reference.metadata["config"]["rho"] != config.rho:
        raise ValueError("Reference gyro-radius differs from the study radius.")


def run_gc_energy_bound_study(
    potential: Potential, configuration: GCInitialConfiguration, *,
    config: GCEnergyBoundConfig, reference: StoredReferenceTrajectory,
) -> GCEnergyBoundResult:
    """Compare physical accuracy and stage-resolved energy balance at fixed h.

    Timing replays use physical states with no energy observer for either method.
    Energy observers are enabled only in an untimed replay, which is required to
    reproduce the timed trajectory. Reference generation is always excluded.
    """
    dynamics = GuidingCenterDynamics(potential, rho=config.rho)
    problem = InitialValueProblem(dynamics, configuration)
    _validate_reference(reference, dynamics, problem.initial_state, config)
    tref = reference.times
    href = np.asarray(dynamics.hamiltonian(tref, reference.states)).reshape(-1)
    haudit = np.asarray(dynamics.hamiltonian(tref, reference.audit_states)).reshape(-1)
    energy_scale = max(float(np.max(np.abs(href))), float(np.finfo(float).eps))
    arrays = {
        "reference/times": tref, "reference/states": reference.states,
        "reference/audit_states": reference.audit_states,
        "reference/H": href, "reference/audit_H": haudit,
        "reference/energy_discrepancy": href-haudit,
        "reference/periodic_discrepancy": reference.audit_distances[0],
        "initial_state": problem.initial_state,
        "domain": np.array([potential.grid.xmin, potential.grid.ymin, potential.grid.period]),
    }
    summary: list[dict[str, Any]] = []
    envelopes: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    bm4 = BM4Implicit(
        coupling_frequency=config.coupling_frequency,
        newton_absolute_tolerance=config.newton_atol,
        newton_relative_tolerance=config.newton_rtol,
        newton_max_iterations=config.newton_max_iterations,
        newton_jacobian_relative_step=config.jacobian_relative_step,
        newton_jacobian_method="analytic", nonlinear_solver="newton",
    )
    methods: dict[str, NumericalMethod] = {"BM4Implicit": bm4, "RK4": RK4()}

    def log(message: str) -> None:
        if config.progress:
            print(message, flush=True)

    with threadpool_limits(limits=1):
        for level, step in enumerate(config.steps):
            count = integer_ratio(config.t_span[1]-config.t_span[0], step, "duration / step")
            request = SimulationRequest.uniform(t_span=config.t_span, max_step=step, sample_count=count+1)
            times = request.output_times
            method_indices, reference_indices = matching_reference_nodes(times, tref)
            comparison_times = times[method_indices]
            ref_states = reference.states[:, reference_indices]
            ref_H = href[reference_indices]
            arrays[f"h{level}/times"] = times
            arrays[f"h{level}/comparison_times"] = comparison_times
            arrays[f"h{level}/comparison_method_indices"] = method_indices
            arrays[f"h{level}/comparison_reference_indices"] = reference_indices
            timing: dict[str, list[float]] = {name: [] for name in METHODS}
            solutions = {}
            log(f"h={step:g}: {count} steps, {config.timing_repeats} alternating timing repeats.")
            # The initial diagnostic/reference cell has already exercised the field.
            for repeat in range(config.timing_repeats):
                for name in METHODS if repeat % 2 == 0 else METHODS[::-1]:
                    started = perf_counter()
                    solution = simulate(problem, methods[name], request)
                    elapsed = perf_counter()-started
                    assert solution.diagnostics["step_count"] == count
                    assert solution.states.shape == (2, count+1)
                    np.testing.assert_array_equal(solution.t, times)
                    solutions[name] = solution
                    timing[name].append(elapsed)
                    log(f"  repeat {repeat+1}: {name} {elapsed:.2f} s")
            observer = GCGeneralizedEnergyObserver(
                dynamics, initial_time=config.t_span[0], initial_state=problem.initial_state)
            log("  Collecting energy at every accepted step in untimed replays.")
            observed_bm4 = simulate(problem, replace(bm4, step_observer=observer), request)
            observed_rk4 = simulate(problem, RK4(track_energy=True), request)
            np.testing.assert_array_equal(observed_bm4.states, solutions["BM4Implicit"].states)
            np.testing.assert_allclose(observed_rk4.states, solutions["RK4"].states, rtol=1e-13, atol=1e-13)
            assert len(observer.records) == count+1
            np.testing.assert_allclose([r.time for r in observer.records], times, rtol=0, atol=1e-11)
            momenta = {
                "BM4Implicit": np.array([record.kappa for record in observer.records]),
                "RK4": np.asarray(observed_rk4.diagnostics["extended_momentum"]).reshape(-1),
            }
            for name in METHODS:
                solution = solutions[name]
                H = np.asarray(dynamics.hamiltonian(times, solution.states)).reshape(-1)
                K_error = H+momenta[name]-H[0]
                H_error = H[method_indices]-ref_H
                distance = particle_distances(solution.states[:, method_indices], ref_states,
                    distance_convention="periodic", period=potential.grid.period)[0]
                prefix = f"h{level}/{name}"
                for label, values in {
                    "states": solution.states, "H": H, "H_change": H-H[0],
                    "kappa": momenta[name], "K_error": K_error,
                    "H_reference_error": H_error, "distance": distance,
                    "K_envelope": np.maximum.accumulate(np.abs(K_error)),
                    "H_envelope": np.maximum.accumulate(np.abs(H_error)),
                    "runtime_samples": np.asarray(timing[name]),
                }.items():
                    arrays[f"{prefix}/{label}"] = values
                quartiles = np.percentile(timing[name], [25, 50, 75])
                record: dict[str, Any] = {
                    "method": name, "step": step, "steps": count,
                    "comparison_sample_count": len(comparison_times),
                    "trajectory_rms": time_rms(distance, comparison_times),
                    "trajectory_final": float(distance[-1]),
                    "trajectory_max": float(distance.max()),
                    "H_error_rms": time_rms(H_error, comparison_times),
                    "H_error_max": float(np.max(np.abs(H_error))),
                    "K_error_max": float(np.max(np.abs(K_error))),
                    "K_relative_max": float(np.max(np.abs(K_error)))/energy_scale,
                    "runtime_median": float(quartiles[1]),
                    "runtime_q25": float(quartiles[0]), "runtime_q75": float(quartiles[2]),
                    "reference_H_floor": float(np.max(np.abs((href-haudit)[reference_indices]))),
                    "reference_distance_floor": float(np.max(reference.audit_distances[:, reference_indices])),
                }
                if name == "BM4Implicit":
                    diag = solution.diagnostics
                    assert diag["projection_solver_formulation"] == "bm4_implicit_reduced"
                    assert diag["newton_jacobian_method"] == "analytic"
                    assert diag["nonlinear_solver"] == "newton"
                    for key in ("nonlinear_iterations", "residual_evaluations", "nonlinear_residual_norms",
                                "nonlinear_tolerances", "projection_multiplier_norms"):
                        values = np.asarray(diag[key])
                        assert values.shape == (count,)
                        arrays[f"{prefix}/{key}"] = values
                    iterations = arrays[f"{prefix}/nonlinear_iterations"]
                    evaluations = arrays[f"{prefix}/residual_evaluations"]
                    mu = arrays[f"{prefix}/projection_multiplier_norms"]
                    ratios = arrays[f"{prefix}/nonlinear_residual_norms"]/arrays[f"{prefix}/nonlinear_tolerances"]
                    assert np.all(ratios <= 1+1e-12)
                    record.update({
                        "solves_per_step": 1, "newton_mean": float(iterations.mean()),
                        "newton_max": int(iterations.max()), "newton_total": int(iterations.sum()),
                        "residual_evaluations_mean": float(evaluations.mean()),
                        "residual_evaluations_total": int(evaluations.sum()),
                        "residual_tolerance_ratio_max": float(ratios.max()),
                        "mu_mean": float(mu.mean()), "mu_rms": float(np.sqrt(np.mean(mu**2))),
                        "mu_max": float(mu.max()), "mu_final": float(mu[-1]),
                    })
                summary.append(record)
                for metric, errors in (("K", K_error), ("H_reference", H_error)):
                    for envelope in envelope_statistics(times if metric == "K" else comparison_times, errors, config.horizons):
                        envelopes.append({"method": name, "step": step, "metric": metric, **envelope})
                for block, indices in enumerate(np.array_split(np.arange(1, count+1), min(config.block_count, count)), 1):
                    values = K_error[indices]
                    blocks.append({"method": name, "step": step, "block": block,
                        "start": float(times[indices[0]]), "stop": float(times[indices[-1]]),
                        "K_min": float(values.min()), "K_max": float(values.max()),
                        "K_mean": float(values.mean())})
    orders: list[dict[str, Any]] = []
    for name in METHODS:
        for metric in ("K", "H_reference"):
            for horizon in config.horizons:
                selected = [r for r in envelopes if r["method"] == name and r["metric"] == metric and r["horizon"] == horizon]
                hs = np.array([r["step"] for r in selected])[-3:]
                errors = np.array([r["maximum"] for r in selected])[-3:]
                slope = float(np.polyfit(np.log(hs), np.log(errors), 1)[0]) if np.all(errors > 0) else None
                orders.append({"method": name, "metric": metric, "horizon": horizon, "slope": slope})
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError("The study produced a nonfinite diagnostic array.")
    metadata = {
        "study": "one_guiding_center_h5_energy_envelope", "config": asdict(config),
        "reference": dict(reference.metadata), "reference_artifact": str(reference.paths.directory),
        "energy_scale": energy_scale, "time_dependent": bool(potential.frequencies.size),
        "convention": "H=gyroaveraged potential; kappa_dot=-partial_t H; K=H+kappa",
        "BM4_energy": "accepted-stage reconstruction; physical Hairer projection; not a fully extended solver",
        "precision": "float64", "all_accepted_nodes_saved": True,
        "reference_comparison": "Exact shared saved nodes only; no interpolation; energy balance retains every accepted node.",
        "timing": "physical integrations; 1 BLAS thread; alternating order; observers excluded",
    }
    return GCEnergyBoundResult(arrays, metadata, summary, envelopes, blocks, orders)


def energy_bound_conclusions(result: GCEnergyBoundResult) -> str:
    """Report observed growth without claiming an infinite-time bound."""
    config = result.metadata["config"]
    step = config["steps"][0]
    duration = config["horizons"][-1]
    tolerance = config["plateau_relative_growth"]
    lines = [f"At h={step:g}, over {duration:g} normalized time units:"]
    for name in METHODS:
        row = next(r for r in result.summary if r["method"] == name and r["step"] == step)
        last = next(r for r in result.envelopes if r["method"] == name and r["step"] == step
                    and r["metric"] == "K" and r["horizon"] == duration)
        ratio = last["growth_ratio"]
        verdict = ("undefined growth ratio (previous envelope was zero)" if ratio is None else
                   ("little observed envelope growth" if ratio <= 1+tolerance else "the envelope is still growing"))
        lines.append(f"- {name}: max |K-K0|={row['K_error_max']:.6e}; "
                     f"max |H-Href|={row['H_error_max']:.6e}; "
                     f"last-interval growth ratio={ratio}; {verdict}.")
        floor = row["reference_H_floor"]
        if floor > 0:
            separation = row["H_error_max"]/floor
            lines.append(f"  Physical-energy error / measured reference discrepancy = {separation:.4g}. "
                         + ("This diagnostic is not comfortably resolved above the reference discrepancy."
                            if separation < 10 else "The error exceeds that measured discrepancy by at least a factor of ten."))
        else:
            lines.append("  The measured physical-energy reference discrepancy is zero; this is not a proof of zero reference error.")
    candidates = [r for r in result.summary if r["step"] == step]
    for key, label in (("runtime_median", "Fastest median physical integration"),
                       ("trajectory_rms", "Smallest time-RMS trajectory error"),
                       ("H_error_rms", "Smallest time-RMS physical-energy error")):
        best = min(candidates, key=lambda row: row[key])
        lines.append(f"- {label}: {best['method']} ({best[key]:.6e}).")
    lines.append("A finite-window plateau is evidence, not a proof of a time-uniform bound. "
                 "Check the reference floor, step refinement, spline regularity and Newton tolerance. "
                 "Small physical-energy error alone does not validate the trajectory.")
    return "\n".join(lines)


def audit_initial_step_geometry(
    potential: Potential, configuration: GCInitialConfiguration, *, config: GCEnergyBoundConfig,
) -> list[dict[str, Any]]:
    """Measure the physical one-step symplectic defect at two FD resolutions.

    This local diagnostic does not certify an autonomous extended map or an
    energy bound. Refining the finite-difference increment exposes its noise floor.
    """
    from diagnostics.jacobians import central_difference_jacobian
    from diagnostics.symplecticity.observer import gc_physical_symplectic_form
    from simulation import IntegrationStep, ImplicitBM4IntegrationStep

    dynamics = GuidingCenterDynamics(potential, rho=config.rho)
    problem = InitialValueProblem(dynamics, configuration)
    if problem.initial_state.shape != (2,):
        raise ValueError("The local geometry audit requires one planar trajectory.")
    h = config.steps[0]
    request = SimulationRequest.uniform(t_span=(config.t_span[0], config.t_span[0]+h),
                                        max_step=h, sample_count=2)
    rows: list[dict[str, Any]] = []
    form = gc_physical_symplectic_form(1)
    for name in METHODS:
        events: list[IntegrationStep] = []
        method = (BM4Implicit(coupling_frequency=config.coupling_frequency,
            newton_absolute_tolerance=config.newton_atol, newton_relative_tolerance=config.newton_rtol,
            newton_max_iterations=config.newton_max_iterations, newton_jacobian_method="analytic",
            newton_jacobian_relative_step=config.jacobian_relative_step, step_observer=events.append)
            if name == "BM4Implicit" else RK4(step_observer=events.append))
        simulate(problem, method, request)
        assert len(events) == 1
        event = events[0]
        if name == "BM4Implicit":
            assert isinstance(event, ImplicitBM4IntegrationStep)
            assert len(event.base_stages) == 12
        for scale in (config.jacobian_relative_step, config.jacobian_relative_step/2):
            jac = central_difference_jacobian(event.map_state, event.state_before, relative_step=scale)
            rows.append({"method": name, "step": h, "fd_relative_step": scale,
                "determinant": float(np.linalg.det(jac)),
                "symplectic_defect": float(np.linalg.norm(jac.T@form@jac-form, ord="fro"))})
    return rows
