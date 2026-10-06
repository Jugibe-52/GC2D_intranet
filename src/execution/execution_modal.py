"""Submit complete integrations to a deployed Modal Function and recover them."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from types import ModuleType
from typing import Any
from uuid import uuid4

from contracts.execution_options import ExecutionOptions
from contracts.problem import InitialValueProblem
from contracts.request import SimulationRequest
from contracts.result import IntegrationData
from execution._modal_worker import PAYLOAD_VERSION, encode_job
from execution.execution import Execution
from methods.base import NumericalMethod


def _modal_sdk() -> ModuleType:
    """Load the optional SDK only when a remote operation is requested."""
    try:
        import modal
    except ImportError as exc:
        raise ImportError("Modal execution requires the optional 'modal' extra: pip install -e '.[modal]'.") from exc
    return modal


def _write_record(path: Path, record: dict[str, Any]) -> None:
    """Atomically persist a receipt before waiting for any remote output."""
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".modal-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(record, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _validate_endpoint_options(
	app_name: str, function_name: str, environment_name: str | None,
	resume_call_id: str | None, wait_timeout: float | None,
) -> None:
	"""Check endpoint names and local waiting controls without contacting Modal."""
	for name, value in (
		("app_name", app_name), ("function_name", function_name),
		("environment_name", environment_name), ("resume_call_id", resume_call_id),
	):
		if value is None and name in ("environment_name", "resume_call_id"):
			continue
		if not isinstance(value, str) or not value.strip():
			raise ValueError(f"`{name}` must be a non-empty string.")
	if wait_timeout is not None:
		if isinstance(wait_timeout, bool) or not math.isfinite(wait_timeout) or wait_timeout < 0:
			raise ValueError("`wait_timeout` must be finite and non-negative, or None.")


def _validated_remote_result(result: object, digest: str) -> IntegrationData:
	"""Require a versioned result from the exact serialized job being recovered."""
	if (not isinstance(result, tuple) or len(result) != 3
			or result[0] != PAYLOAD_VERSION or not isinstance(result[2], IntegrationData)):
		raise TypeError("Modal returned an incompatible integration result.")
	if result[1] != digest:
		raise ValueError("The Modal result belongs to a different serialized job; use the original inputs.")
	data = result[2]
	return data


class Execution_Modal(Execution):
    """Run a complete job remotely, returning IntegrationData locally.

    The deployed Function owns its image, CPU, memory, duration and retry
    settings. This adapter only selects an endpoint and controls local waiting.
    Receipts contain identifiers and an input digest, never result arrays or
    cloud credentials. Use one instance per concurrent caller.
    """

    def __init__(
        self, app_name: str = "gc2d-execution", function_name: str = "integrate",
        *, environment_name: str | None = None,
        record_directory: str | Path = "logs/execution_modal",
        wait_timeout: float | None = None,
        resume_call_id: str | None = None,
    ) -> None:
        """Select a deployed endpoint; construction performs no cloud calls.

        wait_timeout is measured in seconds and bounds local result waiting,
        not remote integration time. None waits indefinitely; zero polls once.
        resume_call_id makes run retrieve an existing call without submitting.
        """
        _validate_endpoint_options(
            app_name, function_name, environment_name, resume_call_id, wait_timeout,
        )
        self.app_name = app_name
        self.function_name = function_name
        self.environment_name = environment_name
        self.record_directory = Path(record_directory)
        self.wait_timeout = wait_timeout
        self.resume_call_id = resume_call_id
        self.last_call_id: str | None = None
        self.last_record_path: Path | None = None

    @classmethod
    def from_record(cls, path: str | Path, *, wait_timeout: float | None = None) -> Execution_Modal:
        """Recover a submitted call after restarting the local process."""
        record_path = Path(path)
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("version") != PAYLOAD_VERSION or not record.get("call_id"):
            raise ValueError("The receipt has no confirmed Modal call ID; inspect Modal before resubmitting.")
        return cls(record["app_name"], record["function_name"],
                   environment_name=record["environment_name"],
                   record_directory=record_path.parent, wait_timeout=wait_timeout,
                   resume_call_id=record["call_id"])

    def run(
        self, problem: InitialValueProblem, method: NumericalMethod, request: SimulationRequest,
        *, options: ExecutionOptions | None = None,
    ) -> IntegrationData:
        """Submit once, save its ID, then fetch the result without local fallback.

        Resumption verifies the digest of the exact serialized job before
        accepting its result. Caller and worker must use matching project and
        dependency versions. Modal exceptions retain their type and traceback;
        local exception notes identify the call and its receipt.
        """
        self.last_call_id = None
        self.last_record_path = None
        payload = encode_job(problem, method, request, options)
        digest = sha256(payload).hexdigest()
        modal = _modal_sdk()
        self.record_directory.mkdir(parents=True, exist_ok=True)
        self.last_record_path = self.record_directory / f"{uuid4().hex}.json"
        record = {
            "version": PAYLOAD_VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
            "app_name": self.app_name, "function_name": self.function_name,
            "environment_name": self.environment_name, "input_sha256": digest,
            "call_id": self.resume_call_id,
            "submission": "confirmed" if self.resume_call_id is not None else "unconfirmed",
        }
        # A failed preflight disk write must prevent submission of an unrecorded job.
        _write_record(self.last_record_path, record)
        try:
            if self.resume_call_id is None:
                function = modal.Function.from_name(
                    self.app_name, self.function_name, environment_name=self.environment_name,
                )
                call = function.spawn(payload)
                self.last_call_id = call.object_id
                record.update(call_id=self.last_call_id, submission="confirmed")
                _write_record(self.last_record_path, record)
            else:
                self.last_call_id = self.resume_call_id
                call = modal.FunctionCall.from_id(self.resume_call_id)
            result = call.get(timeout=self.wait_timeout)
            data = _validated_remote_result(result, digest)
            return IntegrationData(data.t, data.states, {
                **data.diagnostics, "execution_executor": "modal", "execution_call_id": self.last_call_id,
            })
        except BaseException as exc:
            # Preserve numerical errors and SDK error classes, including the
            # distinction between result-wait expiry and FunctionTimeoutError.
            if self.last_call_id is not None:
                exc.add_note(f"Modal call ID: {self.last_call_id}. Receipt: {self.last_record_path}. "
                             "Recover this call instead of submitting the integration again.")
            else:
                exc.add_note(f"Modal submission is unconfirmed. Receipt: {self.last_record_path}. "
                             "It may have reached the server; inspect Modal before resubmitting.")
            raise


__all__ = ["Execution_Modal"]
