"""Local and rclone transports for immutable simulation artifacts.

Remote locations use ``remote:bucket/prefix``. Credentials remain in rclone's
configuration; they never become part of a simulation artifact or command line.
"""

from __future__ import annotations

from pathlib import Path
import json
import re
import shutil
import subprocess


class StorageError(RuntimeError):
    """An artifact could not be transferred to or from its destination."""


class ArtifactStore:
    """Transfer individual files below a local directory or rclone prefix."""

    def __init__(self, location: str | Path) -> None:
        self.location = str(location)
        self.remote = isinstance(location, str) and bool(
            re.match(r"^[A-Za-z0-9_-]+:", location)
        )
        if self.remote:
            _, prefix = self.location.split(":", 1)
            if not prefix.strip("/") or ".." in prefix.split("/"):
                raise ValueError("A remote destination requires a bucket and safe prefix.")
            self.location = self.location.rstrip("/")
            executable = shutil.which("rclone")
            user_executable = Path.home() / ".local/bin/rclone"
            if executable is None and user_executable.is_file():
                executable = str(user_executable)
            if executable is None:
                raise StorageError("Install rclone and configure the requested remote first.")
            self.executable = executable
        else:
            if isinstance(location, str) and "://" in location:
                raise ValueError("Use a local path or rclone 'remote:bucket/prefix' location.")
            self.location = str(Path(location).expanduser().resolve())
            self.executable = ""

    def _run(self, *arguments: str, missing_ok: bool = False) -> str | None:
        """Use bounded transfers and omit process output that might contain secrets."""
        try:
            result = subprocess.run(
                [self.executable, *arguments, "--retries", "2", "--low-level-retries", "3",
                 "--contimeout", "15s", "--timeout", "60s", "--stats", "0"],
                capture_output=True, text=True, timeout=300, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise StorageError("Could not complete the rclone operation.") from exc
        if missing_ok and result.returncode in (3, 4):
            return None
        if result.returncode:
            raise StorageError(
                f"rclone {arguments[0]} failed (exit {result.returncode}). "
                "Check the remote name, connectivity and bucket permissions."
            )
        return result.stdout

    def exists(self, name: str) -> bool:
        """Check presence without treating authentication failures as missing data."""
        if self.remote:
            # B2 can report a nonexistent object as a synthetic directory when
            # using --stat. Listing actual files in the run avoids false hits.
            listing = self._run("lsjson", self.location, "--files-only",
                                "--max-depth", "1", missing_ok=True)
            if listing is None:
                return False
            return any(item["Name"] == name for item in json.loads(listing))
        return (Path(self.location) / name).exists()

    def put(self, source: Path, name: str) -> None:
        """Publish one file, refusing to replace different remote content."""
        if self.remote:
            self._run("copyto", str(source), f"{self.location}/{name}",
                      "--immutable", "--checksum")
        else:
            with source.open("rb") as reader, (Path(self.location) / name).open("xb") as writer:
                shutil.copyfileobj(reader, writer)

    def get(self, name: str, destination: Path) -> None:
        """Fetch one object into a private local staging directory."""
        if self.remote:
            self._run("copyto", f"{self.location}/{name}", str(destination))
        else:
            shutil.copyfile(Path(self.location) / name, destination)


__all__ = ["ArtifactStore", "StorageError"]
