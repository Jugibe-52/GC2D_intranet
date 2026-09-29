"""Offline contracts for remote integration, receipts and safe recovery."""

import importlib.util
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from contracts.execution_options import ExecutionOptions
from contracts.problem import InitialValueProblem
from contracts.request import SimulationRequest
from contracts.result import IntegrationData
from diagnostics.persistence import load_solution, save_solution
from dynamics.gc import GuidingCenterDynamics
from execution import Execution, Execution_Modal
from execution._modal_worker import encode_job, execute_payload
from initial_conditions.gc import GCInitialConfiguration
from methods import BM4Implicit, DOP853, RK4, Radau
from potential.potential import Potential
from simulation.runner import simulate


def make_problem():
    """A small batched physical problem with reconstructible field data."""
    potential = Potential.random(A=.1, M=3, nx=16, ny=16, seed=27)
    return InitialValueProblem(
        GuidingCenterDynamics(potential, rho=.3),
        GCInitialConfiguration.from_components(x=np.array([.2, .4]), y=np.array([.3, .5])),
    )


class ModalExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.records = Path(self.temporary.name) / "records"
        self.problem = make_problem()
        self.method = RK4()
        self.request = SimulationRequest.uniform(t_span=(0., .02), max_step=.01, sample_count=3)
        self.calls = {}
        self.function = SimpleNamespace(spawn=Mock(side_effect=self.spawn))
        self.sdk = SimpleNamespace(
            Function=SimpleNamespace(from_name=Mock(return_value=self.function)),
            FunctionCall=SimpleNamespace(from_id=Mock(side_effect=self.calls.__getitem__)),
        )
        self.loader = patch("execution.execution_modal._modal_sdk", return_value=self.sdk).start()
        self.addCleanup(patch.stopall)

    def spawn(self, payload):
        call_id = f"fc-{len(self.calls) + 1}"

        def get(*, timeout):
            # Model both serialization boundaries while executing the actual
            # numerical worker. No Modal cloud calls or bucket uploads occur.
            return pickle.loads(pickle.dumps(execute_payload(payload), protocol=5))

        call = SimpleNamespace(object_id=call_id, get=Mock(side_effect=get))
        self.calls[call_id] = call
        return call

    def executor(self, **kwargs):
        return Execution_Modal(record_directory=self.records, **kwargs)

    def test_complete_integrations_match_local_and_can_be_saved_locally(self):
        for method in (RK4(), BM4Implicit(), DOP853(), Radau()):
            with self.subTest(method=type(method).__name__):
                reference = simulate(self.problem, method, self.request)
                executor = self.executor()
                result = simulate(self.problem, method, self.request, execution=executor)
                np.testing.assert_allclose(result.states, reference.states, rtol=1e-13, atol=1e-14)
                np.testing.assert_array_equal(result.t, reference.t)
                self.assertFalse(result.states.flags.writeable)
                self.assertEqual(result.diagnostics["execution_call_id"], executor.last_call_id)
                destination = Path(self.temporary.name) / type(method).__name__
                save_solution(result, destination, metadata={"test": "modal_roundtrip"})
                np.testing.assert_array_equal(load_solution(destination).solution.states, result.states)
        self.assertEqual(self.function.spawn.call_count, 4)

    def test_receipt_is_durable_before_waiting_for_result(self):
        original_spawn = self.spawn

        def inspect_wait(payload):
            call = original_spawn(payload)
            original_get = call.get.side_effect

            def get(*, timeout):
                records = list(self.records.glob("*.json"))
                self.assertEqual(len(records), 1)
                receipt = json.loads(records[0].read_text())
                self.assertEqual(receipt["call_id"], call.object_id)
                self.assertEqual(receipt["submission"], "confirmed")
                self.assertNotIn("payload", receipt)
                return original_get(timeout=timeout)

            call.get.side_effect = get
            return call

        self.function.spawn.side_effect = inspect_wait
        executor = self.executor(wait_timeout=10.)
        self.assertIsInstance(executor, Execution)
        executor.run(self.problem, self.method, self.request)
        self.calls[executor.last_call_id].get.assert_called_once_with(timeout=10.)

    def test_lost_connection_recovers_the_same_call_in_a_new_executor(self):
        original_spawn = self.spawn

        def disconnect(payload):
            call = original_spawn(payload)
            result = execute_payload(payload)
            call.get.side_effect = [ConnectionError("Lost connection"), result]
            return call

        self.function.spawn.side_effect = disconnect
        executor = self.executor()
        with self.assertRaises(ConnectionError) as caught:
            simulate(self.problem, self.method, self.request, execution=executor)
        self.assertIn(executor.last_call_id, caught.exception.__notes__[0])
        recovered = Execution_Modal.from_record(executor.last_record_path)
        # Reconstructing the same physical inputs works after a local restart.
        result = simulate(make_problem(), RK4(), self.request, execution=recovered)
        reference = simulate(self.problem, self.method, self.request)
        np.testing.assert_allclose(result.states, reference.states, rtol=1e-13, atol=1e-14)
        self.assertEqual(self.function.spawn.call_count, 1)
        self.sdk.FunctionCall.from_id.assert_called_once_with(executor.last_call_id)

    def test_resume_rejects_changed_method_or_physics_or_sampling(self):
        executor = self.executor()
        executor.run(self.problem, self.method, self.request)
        other_problem = InitialValueProblem(GuidingCenterDynamics(self.problem.dynamics.potential, rho=.1),
                                           self.problem.initial_configuration)
        other_request = SimulationRequest.uniform(t_span=(0., .02), max_step=.005, sample_count=3)
        for problem, method, request in ((other_problem, self.method, self.request),
                                         (self.problem, BM4Implicit(), self.request),
                                         (self.problem, self.method, other_request)):
            resumed = self.executor(resume_call_id=executor.last_call_id)
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, "different serialized job"):
                resumed.run(problem, method, request)
        self.assertEqual(self.function.spawn.call_count, 1)

    def test_wait_expiry_numerical_failure_and_interrupt_keep_original_error(self):
        for error in (TimeoutError("Still running"), RuntimeError("Newton did not converge"),
                      KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                def fail(payload):
                    call = self.spawn(payload)
                    call.get.side_effect = error
                    return call

                self.function.spawn.side_effect = fail
                executor = self.executor(wait_timeout=0.)
                with self.assertRaises(type(error)) as caught:
                    executor.run(self.problem, self.method, self.request)
                self.assertIs(caught.exception, error)
                self.assertIn(executor.last_call_id, error.__notes__[0])
        self.assertEqual(self.function.spawn.call_count, 3)

    def test_failed_submission_retains_an_unconfirmed_receipt_without_retry(self):
        self.function.spawn.side_effect = ConnectionError("Reply lost during submission")
        executor = self.executor()
        with self.assertRaises(ConnectionError) as caught:
            executor.run(self.problem, self.method, self.request)
        record = json.loads(executor.last_record_path.read_text())
        self.assertIsNone(record["call_id"])
        self.assertEqual(record["submission"], "unconfirmed")
        self.assertIn("unconfirmed", caught.exception.__notes__[0])
        with self.assertRaisesRegex(ValueError, "no confirmed"):
            Execution_Modal.from_record(executor.last_record_path)
        self.function.spawn.assert_called_once()

    def test_failed_preflight_record_prevents_submission(self):
        executor = self.executor()
        with patch("execution.execution_modal._write_record", side_effect=OSError("Disk full")):
            with self.assertRaises(OSError):
                executor.run(self.problem, self.method, self.request)
        self.function.spawn.assert_not_called()

    def test_failed_record_after_submission_exposes_id_for_recovery(self):
        from execution.execution_modal import _write_record
        count = 0

        def fail_after_submit(path, record):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError("Disk full after submission")
            _write_record(path, record)

        executor = self.executor()
        with patch("execution.execution_modal._write_record", side_effect=fail_after_submit):
            with self.assertRaises(OSError) as caught:
                executor.run(self.problem, self.method, self.request)
        self.assertIn(executor.last_call_id, caught.exception.__notes__[0])
        self.calls[executor.last_call_id].get.assert_not_called()
        resumed = self.executor(resume_call_id=executor.last_call_id)
        self.assertIsInstance(resumed.run(self.problem, self.method, self.request), IntegrationData)
        self.function.spawn.assert_called_once()

    def test_unsupported_jobs_fail_before_sdk_or_submission(self):
        for method, options in ((RK4(), ExecutionOptions(backend="jax", device="gpu")),
                                (RK4(progress=True), None),
                                (RK4(step_observer=lambda event: None), None)):
            with self.subTest(method=method), self.assertRaises(NotImplementedError):
                self.executor().run(self.problem, method, self.request, options=options)
        with self.assertRaises(TypeError):
            self.executor().run(self.problem, self.method, self.request, options="jax")
        self.loader.assert_not_called()
        self.assertFalse(self.records.exists())

    def test_unserializable_job_fails_before_sdk(self):
        class LocalMethod:
            def integrate(self, problem, request):
                raise AssertionError("Not called")

        with self.assertRaises((AttributeError, pickle.PicklingError)):
            self.executor().run(self.problem, LocalMethod(), self.request)
        self.loader.assert_not_called()

    def test_reused_executor_clears_the_previous_id_before_failed_submission(self):
        executor = self.executor()
        executor.run(self.problem, self.method, self.request)
        previous_path = executor.last_record_path
        self.function.spawn.side_effect = ConnectionError("Unknown submission")
        with self.assertRaises(ConnectionError):
            executor.run(self.problem, self.method, self.request)
        self.assertIsNone(executor.last_call_id)
        self.assertNotEqual(executor.last_record_path, previous_path)

    def test_bad_remote_results_are_rejected_and_keep_call_context(self):
        for output in (None, (99, "digest", None), (1, "digest", "not data")):
            def bad_result(payload):
                call = self.spawn(payload)
                call.get.side_effect = None
                call.get.return_value = output
                return call

            self.function.spawn.side_effect = bad_result
            with self.subTest(output=output), self.assertRaisesRegex(TypeError, "incompatible integration"):
                self.executor().run(self.problem, self.method, self.request)

    def test_simulation_still_validates_remote_physical_output(self):
        def wrong_initial(payload):
            call = self.spawn(payload)
            version, digest, data = execute_payload(payload)
            data.states[0, 0] += 1.
            call.get.side_effect = None
            call.get.return_value = (version, digest, data)
            return call

        self.function.spawn.side_effect = wrong_initial
        with self.assertRaisesRegex(ValueError, "initial state"):
            simulate(self.problem, self.method, self.request, execution=self.executor())

    def test_configuration_validation_is_local(self):
        for kwargs in ({"app_name": ""}, {"function_name": ""}, {"resume_call_id": ""},
                       {"wait_timeout": -1}, {"wait_timeout": float("nan")}, {"wait_timeout": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.executor(**kwargs)
        self.loader.assert_not_called()


class ModalWorkerTests(unittest.TestCase):
    def test_worker_in_a_separate_process_and_serialized_result(self):
        problem = make_problem()
        request = SimulationRequest.uniform(t_span=(0., .02), max_step=.01, sample_count=3)
        payload = encode_job(problem, BM4Implicit(), request, None)
        script = (
            "import pickle, sys; from execution._modal_worker import execute_payload; "
            "sys.stdout.buffer.write(pickle.dumps(execute_payload(sys.stdin.buffer.read()), protocol=5))"
        )
        result = subprocess.run([sys.executable, "-c", script], input=payload, capture_output=True, check=True)
        _, _, data = pickle.loads(result.stdout)
        reference = BM4Implicit().integrate(problem, request)
        np.testing.assert_allclose(data.states, reference.states, rtol=1e-13, atol=1e-14)

    def test_worker_rejects_incompatible_payload_version(self):
        with self.assertRaisesRegex(ValueError, "version"):
            execute_payload(pickle.dumps((999, None, None, None, None)))

    def test_default_import_and_local_execution_do_not_require_modal(self):
        script = '''
import sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname == "modal" or fullname.startswith("modal."):
            raise ImportError("Modal deliberately unavailable")
sys.meta_path.insert(0, Block())
from execution import Execution, Execution_Modal
from execution.execution_modal import _modal_sdk
from test_jax_rk4 import problem as make_problem
from contracts.request import SimulationRequest
from methods import RK4
from simulation.runner import simulate
simulate(make_problem(), RK4(), SimulationRequest.uniform(t_span=(0., .01), max_step=.01))
assert "modal" not in sys.modules
Execution_Modal()
try:
    _modal_sdk()
except ImportError as error:
    assert "optional" in str(error)
else:
    raise AssertionError("Missing optional dependency was ignored")
'''
        result = subprocess.run([sys.executable, "-c", script], cwd="tests", capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(importlib.util.find_spec("modal"), "Optional Modal SDK is not installed")
    def test_deployment_module_registers_without_contacting_modal(self):
        import runpy
        namespace = runpy.run_path("examples/modal_execution_app.py")
        self.assertEqual(namespace["app"].name, "gc2d-execution")
        payload = encode_job(make_problem(), RK4(), SimulationRequest.uniform(
            t_span=(0., .01), max_step=.01), None)
        self.assertIsInstance(namespace["integrate"].local(payload)[2], IntegrationData)


if __name__ == "__main__":
    unittest.main()
