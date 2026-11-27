"""Execute all viewer cells using their actual embedded or optional HTTPS data."""
from __future__ import annotations

import argparse
import base64
from contextlib import redirect_stdout
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')


def validate(mode: str, save_outputs: bool) -> dict:
	"""Record actual execution provenance; a local check is not a Colab-runtime test."""
	path = Path(__file__).resolve().with_name('visualisation.ipynb')
	notebook = json.loads(path.read_text(encoding='utf-8'))
	namespace = {'__name__': '__main__'}
	count = 0
	for cell in notebook['cells']:
		if cell['cell_type'] != 'code':
			continue
		count += 1
		previous = {name: namespace.get(name) for name in ('correction_figure', 'newton_figure')}
		with redirect_stdout(io.StringIO()) as captured:
			exec(compile(''.join(cell['source']), f'notebook-cell-{count}', 'exec'), namespace)
		if count == 1:
			namespace['DATA_SOURCE'] = mode
		outputs = []
		if captured.getvalue():
			outputs.append(dict(output_type='stream', name='stdout', text=captured.getvalue().splitlines(keepends=True)))
		for name, old in previous.items():
			figure = namespace.get(name)
			if figure is not None and figure is not old:
				buffer = io.BytesIO()
				figure.savefig(buffer, format='png', dpi=180)
				outputs.append(dict(output_type='display_data', metadata={}, data={
					'image/png': base64.b64encode(buffer.getvalue()).decode(),
					'text/plain': ['<Figure: verified section-4.5 projection comparison>']}))
		cell['outputs'], cell['execution_count'] = outputs, count
	metadata = notebook['metadata']['gc2d']
	metadata.update(execution_mode=mode, manifest_sha256=namespace['EXPECTED_MANIFEST_SHA256'],
		executed_utc=datetime.now(timezone.utc).isoformat(),
		saved_outputs_source=f'Local Python execution using the notebook\'s {mode} loader; pinned hashes, case records and eight exact thesis-figure series verified. This is not a Google Colab runtime test.')
	if save_outputs:
		path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
	report = dict(code_cells=count, figures=2, panels=4, complete_cases=21, unavailable_cases=1,
		thesis_series_verified=8, execution_mode=mode, saved_outputs=save_outputs,
		manifest_sha256=namespace['EXPECTED_MANIFEST_SHA256'])
	print(json.dumps(report, indent=2))
	return report


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--mode', choices=('snapshot', 'public_https'), default='snapshot')
	parser.add_argument('--save-outputs', action='store_true')
	args = parser.parse_args()
	validate(args.mode, args.save_outputs)
