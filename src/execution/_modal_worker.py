"""Versioned NumPy/SciPy and JAX CPU integration payloads for Modal."""

from __future__ import annotations

from hashlib import sha256
import pickle
from time import perf_counter, time

from contracts.execution_options import ExecutionOptions
from contracts.problem import InitialValueProblem
from contracts.request import SimulationRequest
from contracts.result import IntegrationData
from execution.execution import Execution
from methods.base import NumericalMethod


PAYLOAD_VERSION = 1


def validate_job(
    problem: InitialValueProblem, method: NumericalMethod, request: SimulationRequest,
    options: ExecutionOptions | None,
) -> ExecutionOptions:
    """Reject unsupported work before submission and again in the worker."""
    if not isinstance(problem, InitialValueProblem):
        raise TypeError("`problem` must be an InitialValueProblem.")
    if not isinstance(method, NumericalMethod):
        raise TypeError("`method` must implement NumericalMethod.")
    if not isinstance(request, SimulationRequest):
        raise TypeError("`request` must be a SimulationRequest.")
    if options is not None and not isinstance(options, ExecutionOptions):
        raise TypeError("`options` must be an ExecutionOptions instance or None.")
    choice = ExecutionOptions() if options is None else options
    if choice.device != "cpu":
        raise NotImplementedError("Modal execution currently supports CPU devices only.")
    if getattr(method, "step_observer", None) is not None or getattr(method, "progress", False):
        raise NotImplementedError("Modal execution requires step_observer=None and progress=False.")
    if getattr(method, "_status", "configuration") != "configuration":
        raise ValueError("Submit a configured method, not an initialized or completed run.")
    return choice


def encode_job(
    problem: InitialValueProblem, method: NumericalMethod, request: SimulationRequest,
    options: ExecutionOptions | None,
) -> bytes:
    """Snapshot inputs using importable project types, excluding device caches.

    Potential's pickle contract transfers field samples and recreates its
    interpolation resources in the worker. No source HDF5 path is required.
    """
    choice = validate_job(problem, method, request, options)
    return pickle.dumps((PAYLOAD_VERSION, problem, method, request, choice), protocol=5)


def decode_job(payload: bytes) -> tuple[InitialValueProblem, NumericalMethod, SimulationRequest, ExecutionOptions]:
    """Restore and validate a trusted job once for one or several integrations."""
    version, problem, method, request, options = pickle.loads(payload)
    if version != PAYLOAD_VERSION:
        raise ValueError("Unsupported Modal integration payload version.")
    choice = validate_job(problem, method, request, options)
    return problem, method, request, choice


def execute_payload(payload: bytes) -> tuple[int, str, IntegrationData]:
    """Integrate an authenticated job and return data to the submitting machine.

    This is a trusted Python-pickle boundary, used only by the authenticated
    Modal Function. It must never be exposed as an unauthenticated endpoint.
    """
    problem, method, request, choice = decode_job(payload)
    data = Execution().run(problem, method, request, options=choice)
    if not isinstance(data, IntegrationData):
        raise TypeError("The remote method must return IntegrationData.")
    return PAYLOAD_VERSION, sha256(payload).hexdigest(), data


def execute_cycle_payload(payload: bytes) -> tuple[int, str, IntegrationData]:
    """Return cycle observations and scalar diagnostics with worker timing.

    Absolute Unix intervals allow the submitting batch to measure integration
    overlap rather than infer it from the configured container ceiling.
    """
    import numpy as np

    started_unix, started = time(), perf_counter()
    version, digest, data = execute_payload(payload)
    diagnostics = {key: value for key, value in data.diagnostics.items()
                   if not isinstance(value, np.ndarray)}
    diagnostics.update(worker_wall_seconds=perf_counter() - started,
                       worker_started_unix_seconds=started_unix,
                       worker_finished_unix_seconds=time())
    return version, digest, IntegrationData(data.t, data.states, diagnostics)


__all__: list[str] = []
