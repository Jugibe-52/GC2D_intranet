#!/usr/bin/env python3
"""Synchronize untracked Spanish Markdown copies with their dedicated B2 bucket."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REMOTE = "gc2d_es_docs:gcd-esp-docs"
STATE = Path.home() / ".local/state/rclone/gc2d-es-docs"
FILTERS = r"""# Spanish reading copies and preserved bisync conflict copies only.
- .git/**
- .venv/**
- node_modules/**
- __pycache__/**
- .ipynb_checkpoints/**
- .mypy_cache/**
- .pytest_cache/**
- .ruff_cache/**
- .wrangler/**
- /build/**
- /dist/**
+ {{[^/]+\.es(\.[0-9]+)?\.md(\.conflict[0-9]+)*}}
- **
"""


def capture(*args: str) -> str:
	"""Read command output without invoking a shell."""
	return subprocess.check_output(args, text=True, cwd=ROOT)


def inventory(path: str | Path, filters: Path) -> dict[str, dict[str, Any]]:
	"""Return selected relative paths and SHA-1 hashes on either side."""
	items = json.loads(capture(
		"rclone", "lsjson", str(path), "--recursive", "--files-only",
		"--hash", "--hash-type", "SHA-1", "--filter-from", str(filters),
	))
	return {item["Path"]: item for item in items}


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--dry-run", action="store_true", help="Preview transfers only.")
	parser.add_argument(
		"--initialize", action="store_true",
		help="Create this computer's first bisync baseline after checking both sides.",
	)
	args = parser.parse_args()
	if Path(capture("git", "rev-parse", "--show-toplevel").strip()).resolve() != ROOT:
		parser.error("The script must remain in the GC2D_codigo repository.")
	tracked = set(capture("git", "ls-files", "-z").split("\0"))
	STATE.mkdir(parents=True, exist_ok=True, mode=0o700)
	filters = STATE / "spanish-docs.filters"
	if filters.exists() and filters.read_text() != FILTERS:
		parser.error("Filters changed: review existing bisync state before migrating them.")
	if not filters.exists():
		filters.write_text(FILTERS)
	work = STATE / "bisync"
	if args.initialize and (
		any(work.glob("*.lst")) or any(work.glob("*.lst-err"))
	):
		parser.error("Bisync state already exists; omit --initialize.")
	local = inventory(ROOT, filters)
	remote = inventory(REMOTE, filters)
	collisions = tracked.intersection(local.keys() | remote.keys())
	if collisions:
		parser.error("Selected paths are tracked by Git: " + ", ".join(sorted(collisions)))
	if not local and not remote:
		parser.error("Both selected trees are empty; nothing to initialize or synchronize.")
	if args.initialize:
		# Initial resync overwrites conflicts, so permit it only when overlap is equal.
		conflicts = []
		for path in local.keys() & remote.keys():
			left = local[path].get("Hashes", {}).get("sha1")
			right = remote[path].get("Hashes", {}).get("sha1")
			if not left or not right or left != right:
				conflicts.append(path)
		if conflicts:
			parser.error("Preserve and reconcile these versions first: " + ", ".join(sorted(conflicts)))
	stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
	logs = STATE / "logs"
	logs.mkdir(exist_ok=True)
	log = logs / (stamp + ("-dry-run" if args.dry_run else "") + ".log")
	command = [
		"rclone", "bisync", str(ROOT), REMOTE,
		"--filters-file", str(filters), "--workdir", str(work),
		"--compare", "size,modtime,checksum", "--conflict-resolve", "none",
		"--conflict-loser", "num", "--max-delete", "10",
		"--backup-dir1", str(STATE / "backups" / stamp),
		"--log-file", str(log), "--log-level", "INFO", "--stats", "0",
	]
	if args.initialize:
		command.append("--resync")
	if args.dry_run:
		command.append("--dry-run")
	print(f"Selected files: local={len(local)}, remote={len(remote)}", flush=True)
	print(f"Log: {log}", flush=True)
	result = subprocess.run(command, cwd=ROOT)
	if log.exists():
		print("\n".join(log.read_text().splitlines()[-20:]), flush=True)
	if result.returncode:
		raise SystemExit(result.returncode)
	if not args.dry_run:
		subprocess.run([
			"rclone", "check", str(ROOT), REMOTE, "--filter-from", str(filters),
		], cwd=ROOT, check=True)


if __name__ == "__main__":
	main()
