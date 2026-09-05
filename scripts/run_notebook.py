#!/usr/bin/env python3
"""Run the configured notebook locally or on the reusable AWS worker."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import tomllib
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "conf" / "notebook_execution.toml"


class RunnerError(RuntimeError):
    """Report an actionable execution or cloud-orchestration failure."""


def load_config(path: Path) -> dict[str, Any]:
    """Load and minimally validate the execution configuration."""
    with path.open("rb") as stream:
        config = tomllib.load(stream)

    target = config.get("execution_target")
    if target not in {"local", "aws"}:
        raise RunnerError("execution_target must be either 'local' or 'aws'.")
    for section in ("notebook", "aws"):
        if not isinstance(config.get(section), dict):
            raise RunnerError(f"Missing [{section}] configuration section.")
    return config


def display_plan(config: dict[str, Any], target: str) -> None:
    """Print the resolved action without changing local or AWS state."""
    notebook = config["notebook"]
    print(f"Execution target: {target}")
    print(f"Notebook: {notebook['path']}")
    if target == "local":
        print(f"Result: {notebook['local_result']}")
    else:
        aws = config["aws"]
        print(f"AWS region: {aws['region']}")
        print(f"Reusable instance tag: Name={aws['instance_name']}")
        print(f"Result: {notebook['aws_result']}")
        print(
            "Safety: the worker stops after the job or after "
            f"{notebook['hard_stop_minutes']} minutes."
        )


def run_local(config: dict[str, Any]) -> None:
    """Execute a copy of the configured notebook in the project environment."""
    notebook = config["notebook"]
    source = PROJECT_ROOT / notebook["path"]
    result = PROJECT_ROOT / notebook["local_result"]
    jupyter = PROJECT_ROOT / ".venv" / "bin" / "jupyter"

    if not source.is_file():
        raise RunnerError(f"Notebook not found: {source}")
    if not jupyter.is_file():
        raise RunnerError(
            "Project Jupyter executable not found. Install the notebook extra first."
        )

    result.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, result)
    command = [
        str(jupyter),
        "execute",
        "--inplace",
        f"--timeout={int(notebook['cell_timeout_seconds'])}",
        str(result),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    print(f"Executed notebook written to {result}")


def aws_command(
    aws_executable: str,
    arguments: Sequence[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run one AWS CLI command and retain its output for validation."""
    completed = subprocess.run(
        [aws_executable, *arguments],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RunnerError(f"AWS CLI command failed: {detail}")
    return completed


def find_worker(
    aws_executable: str,
    region: str,
    instance_name: str,
) -> dict[str, Any]:
    """Resolve exactly one active or stopped reusable worker by its Name tag."""
    response = aws_command(
        aws_executable,
        [
            "ec2",
            "describe-instances",
            "--region",
            region,
            "--filters",
            f"Name=tag:Name,Values={instance_name}",
            "Name=instance-state-name,Values=pending,running,stopping,stopped",
            "--output",
            "json",
        ],
    )
    document = json.loads(response.stdout)
    instances = [
        instance
        for reservation in document.get("Reservations", [])
        for instance in reservation.get("Instances", [])
    ]
    if not instances:
        raise RunnerError(
            "Reusable AWS worker not found. Provision the CloudFormation stack first."
        )
    if len(instances) != 1:
        raise RunnerError(
            f"Expected one reusable AWS worker, found {len(instances)}."
        )
    return instances[0]


def instance_state(
    aws_executable: str,
    region: str,
    instance_id: str,
) -> str:
    """Return the current EC2 state for one worker."""
    response = aws_command(
        aws_executable,
        [
            "ec2",
            "describe-instances",
            "--region",
            region,
            "--instance-ids",
            instance_id,
            "--query",
            "Reservations[0].Instances[0].State.Name",
            "--output",
            "text",
        ],
    )
    return response.stdout.strip()


def wait_for_ssm(
    aws_executable: str,
    region: str,
    instance_id: str,
    deadline: float,
) -> None:
    """Wait until Systems Manager can accept a command for the worker."""
    while time.monotonic() < deadline:
        response = aws_command(
            aws_executable,
            [
                "ssm",
                "describe-instance-information",
                "--region",
                region,
                "--filters",
                f"Key=InstanceIds,Values={instance_id}",
                "--query",
                "InstanceInformationList[0].PingStatus",
                "--output",
                "text",
            ],
        )
        if response.stdout.strip() == "Online":
            return
        time.sleep(10)
    raise RunnerError("The AWS worker did not become available in Systems Manager.")


def s3_object_exists(
    aws_executable: str,
    region: str,
    bucket: str,
    key: str,
) -> bool:
    """Check whether a result object exists without downloading it."""
    response = aws_command(
        aws_executable,
        [
            "s3api",
            "head-object",
            "--region",
            region,
            "--bucket",
            bucket,
            "--key",
            key,
        ],
        check=False,
    )
    return response.returncode == 0


def stop_worker(
    aws_executable: str,
    region: str,
    instance_id: str,
) -> None:
    """Request a safe stop when the worker is still consuming compute."""
    state = instance_state(aws_executable, region, instance_id)
    if state in {"pending", "running"}:
        aws_command(
            aws_executable,
            ["ec2", "stop-instances", "--region", region, "--instance-ids", instance_id],
        )


def run_aws(config: dict[str, Any]) -> None:
    """Start, dispatch to, monitor, and stop the reusable AWS worker."""
    aws_executable = shutil.which("aws")
    if aws_executable is None:
        raise RunnerError(
            "AWS CLI v2 is not installed or is not on PATH. Complete the one-time "
            "AWS CLI setup before selecting execution_target='aws'."
        )

    notebook = config["notebook"]
    aws = config["aws"]
    region = str(aws["region"])
    instance = find_worker(aws_executable, region, str(aws["instance_name"]))
    instance_id = str(instance["InstanceId"])
    state = str(instance["State"]["Name"])
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    result_root = f"{str(aws['result_prefix']).rstrip('/')}/{run_id}"
    status_key = f"{result_root}/status.json"
    started = False

    try:
        if state == "stopping":
            aws_command(
                aws_executable,
                [
                    "ec2",
                    "wait",
                    "instance-stopped",
                    "--region",
                    region,
                    "--instance-ids",
                    instance_id,
                ],
            )
            state = "stopped"
        if state == "stopped":
            aws_command(
                aws_executable,
                ["ec2", "start-instances", "--region", region, "--instance-ids", instance_id],
            )
            started = True
        elif state == "running":
            raise RunnerError(
                "The reusable worker is already running. Wait for it to stop or inspect "
                "the active job before dispatching another run."
            )
        else:
            raise RunnerError(f"Worker cannot be started from EC2 state {state!r}.")

        startup_deadline = time.monotonic() + 15 * 60
        wait_for_ssm(aws_executable, region, instance_id, startup_deadline)
        parameters = json.dumps(
            {"commands": [f"sudo /usr/local/bin/gc2d-run-notebook {run_id}"]}
        )
        response = aws_command(
            aws_executable,
            [
                "ssm",
                "send-command",
                "--region",
                region,
                "--instance-ids",
                instance_id,
                "--document-name",
                "AWS-RunShellScript",
                "--parameters",
                parameters,
                "--query",
                "Command.CommandId",
                "--output",
                "text",
            ],
        )
        print(f"Remote run {run_id} dispatched as SSM command {response.stdout.strip()}.")

        deadline = time.monotonic() + (int(notebook["hard_stop_minutes"]) + 15) * 60
        while time.monotonic() < deadline:
            if s3_object_exists(
                aws_executable, region, str(aws["bucket"]), status_key
            ):
                break
            current_state = instance_state(aws_executable, region, instance_id)
            if current_state == "stopped":
                raise RunnerError(
                    "The worker stopped before publishing its status. Inspect its system log."
                )
            time.sleep(15)
        else:
            raise RunnerError("Timed out while waiting for the remote notebook result.")

        status_uri = f"s3://{aws['bucket']}/{status_key}"
        status_response = aws_command(
            aws_executable, ["s3", "cp", "--region", region, status_uri, "-"]
        )
        status = json.loads(status_response.stdout)

        cloud_output_dir = PROJECT_ROOT / "outputs" / "cloud" / run_id
        cloud_output_dir.mkdir(parents=True, exist_ok=True)
        log_uri = f"s3://{aws['bucket']}/{result_root}/execution.log"
        aws_command(
            aws_executable,
            ["s3", "cp", "--region", region, log_uri, str(cloud_output_dir / "execution.log")],
        )
        if status.get("status") != "success":
            raise RunnerError(
                f"Remote notebook failed with exit code {status.get('exit_code')}. "
                f"Log: {cloud_output_dir / 'execution.log'}"
            )

        notebook_uri = f"s3://{aws['bucket']}/{result_root}/executed.ipynb"
        result = PROJECT_ROOT / notebook["aws_result"]
        aws_command(
            aws_executable,
            ["s3", "cp", "--region", region, notebook_uri, str(result)],
        )
        print(f"Executed notebook written to {result}")
    finally:
        if started:
            stop_worker(aws_executable, region, instance_id)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line overrides while keeping the TOML target authoritative."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--target", choices=("local", "aws"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run the selected execution route and return a shell-friendly status."""
    arguments = parse_arguments()
    try:
        config = load_config(arguments.config.resolve())
        target = arguments.target or str(config["execution_target"])
        display_plan(config, target)
        if arguments.dry_run:
            return 0
        if target == "local":
            run_local(config)
        else:
            run_aws(config)
    except (OSError, subprocess.SubprocessError, ValueError, RunnerError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
