"""Prepare the small public timing dataset without uploading or integrating."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path


def prepare_bundle(archive: Path, destination: Path) -> str:
	"""Copy only the two verified JSON inputs and create a deterministic manifest."""
	original_bytes = (archive / "manifest.json").read_bytes()
	original = json.loads(original_bytes)
	publication = json.loads((archive / "publication.json").read_text(encoding="utf-8"))
	original_digest = sha256(original_bytes).hexdigest()
	if publication.get("status") != "verified" or publication.get("manifest_sha256") != original_digest:
		raise ValueError("The original publication receipt does not verify this archive manifest.")
	objects = {}
	files = {}
	for name in ("study.json", "timings.json"):
		data = (archive / name).read_bytes()
		entry = dict(bytes=len(data), sha256=sha256(data).hexdigest())
		if entry != original["files"][name]:
			raise ValueError(f"The original archive manifest does not match {name}.")
		objects[name] = data
		files[name] = entry
	study = json.loads(objects["study.json"])
	if (study.get("run_id") != "jax-20261118-rho0015-c500"
			or study.get("study_prefix") != "studies/method_timings"):
		raise ValueError("This publication script requires the selected section-4.2 study.")
	manifest = dict(schema_version=1, study_prefix=study["study_prefix"], run_id=study["run_id"],
		files=files, source_manifest_sha256=original_digest,
		source_archive_verification=f"Original publication receipt records verification of {publication['verified_files']} files; "
			"this public bundle contains only the study parameters and timing measurements.")
	objects["manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
	# Immutable version directories must not silently overwrite a prior publication.
	if destination.exists():
		if set(p.name for p in destination.iterdir()) != set(objects):
			raise ValueError("The destination contains unexpected files; choose a new version directory.")
		if any((destination / name).read_bytes() != data for name, data in objects.items()):
			raise ValueError("The destination differs from this bundle; choose a new version directory.")
	else:
		destination.mkdir(parents=True)
		for name, data in objects.items():
			(destination / name).write_bytes(data)
	return sha256(objects["manifest.json"]).hexdigest()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("destination", type=Path, help="A new version directory, or an identical existing bundle.")
	parser.add_argument("--archive", type=Path,
		default=Path(__file__).resolve().parents[2] / "outputs/studies/method_timings/jax-20261118-rho0015-c500")
	arguments = parser.parse_args()
	digest = prepare_bundle(arguments.archive, arguments.destination)
	print("Prepared bundle:", arguments.destination.resolve())
	print("Manifest SHA-256:", digest)
