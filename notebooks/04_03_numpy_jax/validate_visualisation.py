"""Execute the portable notebook using its public HTTPS data or an offline bundle."""
from __future__ import annotations

import argparse
import base64
from contextlib import redirect_stdout
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")


def validate(bundle: Path | None, save_outputs: bool) -> None:
	"""Use unchanged HTTPS transport when bundle is None; label the actual mode."""
	path = Path(__file__).resolve().with_name("visualisation.ipynb")
	notebook = json.loads(path.read_text(encoding="utf-8"))
	namespace = {"__name__": "__main__"}
	execution_count = 0
	for cell in notebook["cells"]:
		if cell["cell_type"] != "code":
			continue
		execution_count += 1
		previous_figures = {name: namespace.get(name) for name in ("figure", "agreement_figure")}
		with redirect_stdout(io.StringIO()) as captured:
			exec(compile("".join(cell["source"]), f"notebook-cell-{execution_count}", "exec"), namespace)
		if execution_count == 1 and bundle is not None:
			# Replace transport only: the pinned manifest and content checks still run.
			namespace["download_bytes"] = lambda name: (bundle / name).read_bytes()
		outputs = []
		if captured.getvalue():
			outputs.append(dict(output_type="stream", name="stdout", text=captured.getvalue().splitlines(keepends=True)))
		for name in ("figure", "agreement_figure"):
			figure = namespace.get(name)
			if figure is not None and figure is not previous_figures[name]:
				buffer = io.BytesIO()
				figure.savefig(buffer, format="png", dpi=180)
				outputs.append(dict(output_type="display_data", metadata={}, data={
					"image/png": base64.b64encode(buffer.getvalue()).decode(),
					"text/plain": ["<Figure: verified section-4.3 backend comparison>"],
				}))
		cell["outputs"] = outputs
		cell["execution_count"] = execution_count
	metadata = notebook["metadata"]["gc2d"]
	if bundle is None:
		metadata["saved_outputs_source"] = (
			"Local Python execution using the notebook's unchanged HTTPS downloader against public B2; "
			"manifest, data integrity and timing protocol verified. This is not a Google Colab runtime test."
		)
		metadata["execution_mode"] = "public_https"
		metadata["public_data_base_url"] = namespace["PUBLIC_DATA_BASE_URL"]
	else:
		metadata["saved_outputs_source"] = (
			"Offline execution against the prepared publication bundle; "
			+ ("public B2 URL pending." if not namespace["PUBLIC_DATA_BASE_URL"] else "live public access not verified by this run.")
		)
		metadata["execution_mode"] = "offline"
		metadata.pop("public_data_base_url", None)
	metadata["manifest_sha256"] = namespace["EXPECTED_MANIFEST_SHA256"]
	metadata["executed_utc"] = datetime.now(timezone.utc).isoformat()
	if save_outputs:
		path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
	mode = "using public HTTPS data" if bundle is None else "offline"
	print(f"Executed {execution_count} code cells {mode}; verified 72 timing records and 18 paired trajectories.")
	print("Saved outputs updated." if save_outputs else "Notebook file unchanged.")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("bundle", type=Path, nargs="?", help="Offline directory containing the pinned public manifest and its four data files.")
	parser.add_argument("--live", action="store_true", help="Use the notebook's real public HTTPS downloader, without an offline bundle.")
	parser.add_argument("--save-outputs", action="store_true", help="Save outputs with their actual execution mode and provenance in the notebook.")
	arguments = parser.parse_args()
	if arguments.live == (arguments.bundle is not None):
		parser.error("Choose either an offline bundle directory or --live, not both.")
	validate(arguments.bundle, arguments.save_outputs)
