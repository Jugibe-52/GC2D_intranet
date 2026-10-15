# Synchronizing Spanish reading copies

The root of `gc2d_es_docs:gcd-esp-docs` corresponds to the root of
`GC2D_codigo/`. For example, `docs/dynamics/protocols.es.md` keeps that exact
relative path in the bucket. Only untracked Spanish Markdown reading copies
named `*.es.md` or `*.es.<number>.md` are selected, together with numbered
`.conflict` copies preserved by bisync. Git tracking rules remain unchanged.

The procedure is implemented in [sync_spanish_docs.py](../scripts/sync_spanish_docs.py).
It excludes dependency, cache, Git, and build directories and checks selected
local and remote paths against the Git index before transferring. It stops if
any selected path is tracked. The connection is stored in the owner's shared
rclone configuration; credentials must never be copied into this repository.

## Run on an initialized computer

From the repository root, preview the changes and inspect the reported log:

```bash
python3 scripts/sync_spanish_docs.py --dry-run
```

Then perform the synchronization:

```bash
python3 scripts/sync_spanish_docs.py
```

The synchronization is bidirectional: additions, modifications, and deletions
propagate between the two selected trees. It compares size, modification time,
and SHA-1, then runs an independent `rclone check`. Conflicts preserve both
versions with numbered suffixes for manual reconciliation. Local files replaced
or deleted are backed up under the current run's local state directory. B2 uses
ordinary version-preserving deletion (`hard_delete=false`). A change deleting
more than 10 percent of either selected tree stops for review.

No periodic job is installed. The existing notebook synchronization also covers
the Spanish documents under `notebooks/developements/`; their dedicated copies
here are an additional destination. Run the workflows sequentially and review
conflicts if the same document was edited independently.

## First run on another computer

Install the shared rclone connection and locate a real checkout of the project.
Each computer must create its own state; never copy another computer's state.
Use the following commands only for a pair that has never been initialized:

```bash
python3 scripts/sync_spanish_docs.py --initialize --dry-run
python3 scripts/sync_spanish_docs.py --initialize
```

Initialization refuses existing bisync listings and refuses different contents
at the same relative path. Preserve and reconcile both versions before trying
again. Files present on only one side are included in the initial union. Later
runs must omit `--initialize`. A missing or damaged baseline on an existing
installation requires investigation and recovery, not automatic initialization.

## Local state and recovery

Filters, listings, locks, logs, and local backups live in
`~/.local/state/rclone/gc2d-es-docs/`, outside Git. Each run reports its log path.
The script refuses changed filter definitions until their state migration has
been reviewed. Do not edit filters or discard state to bypass a failed run.

To undo the setup, stop invoking the script and remove the `gc2d_es_docs`
connection from the shared rclone configuration after checking its users.
Keep the local documents, bucket contents, and recovery state. Remove the
procedure and its documentation only if they are no longer needed. This does
not delete cloud data or change Git tracking.

See the official [bisync manual](https://rclone.org/bisync/) and
[filtering reference](https://rclone.org/filtering/).
