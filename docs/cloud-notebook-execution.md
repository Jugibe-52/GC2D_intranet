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

## Existing AWS workers

AWS is no longer used for this project, and its provisioning template has been
removed. The AWS route remains in `scripts/run_notebook.py` and requires an
existing `gc2d-worker` instance with Systems Manager available and
`/usr/local/bin/gc2d-run-notebook` installed. The launcher does not create the
worker. Removing the template from the repository does not stop or delete
existing AWS resources.

The local machine needs AWS CLI v2 version 2.32.0 or later. Use the browser-based
`aws login` flow with temporary console credentials; do not create or store
long-lived AWS access keys in this repository or in the notebook. The signed-in
identity must be allowed to describe, start, and stop the tagged EC2 instance,
send and inspect Systems Manager commands, and read the configured S3 result
prefix.
