"""Archive a completed midpoint-geometry study and verify every downloaded byte."""
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from typing import Any

from diagnostics.io.storage import ArtifactStore


def archive_study(directory: Path) -> dict[str, Any]:
    """Store immutable scientific artifacts, then verify a fresh read-back.

    Only the named result directory is archived. Credentials remain entirely
    inside rclone. A receipt is written only after all scientific files match.
    """
    directory = directory.resolve(strict=True)
    required: tuple[str, ...] = (
        "local.npz", "calibration.npz", "implicit_controls.npz",
        "accumulation.npz", "summary.json", "manifest.json", "source.tar.gz",
    )
    for name in required:
        if not (directory / name).is_file():
            raise FileNotFoundError(f"Missing completed study artifact: {name}")
    summary: dict[str, Any] = json.loads((directory / "summary.json").read_text())
    scientific_manifest: dict[str, Any] = json.loads((directory / "manifest.json").read_text())
    if summary.get("status") != "complete" or scientific_manifest.get("status") != "complete":
        raise ValueError("Only a complete study can be archived.")
    for name, entry in scientific_manifest["files"].items():
        if Path(name).name != name or sha256((directory / name).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Scientific manifest mismatch: {name}")
    bucket: str = f"gc2d_data:gc2d-notebooks-data/studies/midpoint_geometry/{directory.name}"
    files: list[Path] = sorted(p for p in directory.rglob("*")
        if p.is_file() and not p.is_symlink()
        and p.suffix in (".json", ".npz", ".gz", ".pdf", ".png", ".py", ".ipynb")
        and p.name not in ("archive-manifest.json", "publication.json"))
    manifest: dict[str, Any] = {
        "schema_version": 1, "bucket": bucket,
        "files": {p.relative_to(directory).as_posix(): {
            "bytes": p.stat().st_size, "sha256": sha256(p.read_bytes()).hexdigest()
        } for p in files},
    }
    manifest_path: Path = directory / "archive-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    store: ArtifactStore = ArtifactStore(bucket)
    for path in files + [manifest_path]:
        name: str = path.relative_to(directory).as_posix()
        store.put(path, name)
        print(f"Stored {name}", flush=True)
    with tempfile.TemporaryDirectory(prefix="midpoint-geometry-readback-") as temporary:
        recovery: Path = Path(temporary)
        for path in files + [manifest_path]:
            name = path.relative_to(directory).as_posix()
            downloaded: Path = recovery / name
            downloaded.parent.mkdir(parents=True, exist_ok=True)
            store.get(name, downloaded)
            if downloaded.read_bytes() != path.read_bytes():
                raise ValueError(f"Read-back mismatch: {name}")
            print(f"Verified {name}", flush=True)
    receipt: dict[str, Any] = {
        "status": "verified", "bucket": bucket,
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "verified_files": len(files) + 1,
        "archive_manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
    }
    receipt_path: Path = directory / "publication.json"
    # Preserve an existing matching immutable receipt on a repeated verification.
    if receipt_path.exists():
        previous: dict[str, Any] = json.loads(receipt_path.read_text())
        if (previous.get("status") == "verified" and previous.get("bucket") == bucket
                and previous.get("archive_manifest_sha256") == receipt["archive_manifest_sha256"]):
            receipt = previous
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    store.put(receipt_path, "publication.json")
    return receipt


def main() -> None:
    parser: ArgumentParser = ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args: Any = parser.parse_args()
    print(json.dumps(archive_study(args.directory), indent=2))


if __name__ == "__main__":
    main()
