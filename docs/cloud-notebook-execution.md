# Local and AWS notebook execution

The execution location is selected by one attribute in
`conf/notebook_execution.toml`:

```toml
execution_target = "local"
```

or:

```toml
execution_target = "aws"
```

Preview the resolved route without executing anything:

```bash
.venv/bin/python scripts/run_notebook.py --dry-run
```

Run the selected route:

```bash
.venv/bin/python scripts/run_notebook.py
```

The local route executes a copy of the source notebook and writes the result
beside it. The AWS route starts the existing `gc2d-worker` instance, waits for
Systems Manager, dispatches the notebook, downloads the executed notebook and
log, and stops the instance in a `finally` block. The worker also has an
independent 60-minute stop guard that starts on every boot.

## One-time AWS setup

The reusable worker is defined by
`infra/aws/gc2d-reusable-worker.yaml`. Provisioning it is a separate,
cost-incurring action and is intentionally not performed by the notebook
launcher. The instance prepares the project and environment during its first
boot and then stops itself. Its encrypted EBS volume persists so later runs do
not reinstall dependencies or download the input data again.

The local machine needs AWS CLI v2 version 2.32.0 or later. Use the browser-based
`aws login` flow with temporary console credentials; do not create or store
long-lived AWS access keys in this repository or in the notebook. The signed-in
identity must be allowed to describe, start, and stop the tagged EC2 instance,
send and inspect Systems Manager commands, and read the configured S3 result
prefix.
