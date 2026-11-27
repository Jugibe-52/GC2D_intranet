"""Build the portable notebook from its reviewable percent-format source."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def build_notebook(directory: Path) -> Path:
	"""Preserve verified outputs only when every executable cell is unchanged."""
	path = directory / 'visualisation.ipynb'
	previous = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
	source_text = (directory / 'visualisation.py').read_text(encoding='utf-8')
	cells = []
	for block in source_text.split('# %%')[1:]:
		header, source = block.split('\n', 1)
		markdown = header.strip() == '[markdown]'
		if markdown:
			source = '\n'.join(line[2:] if line.startswith('# ') else line[1:] if line.startswith('#') else line
				for line in source.splitlines())
		source = source.strip() + '\n'
		cell = dict(cell_type='markdown' if markdown else 'code', metadata={},
			id=hashlib.sha256(source.encode()).hexdigest()[:12], source=source.splitlines(keepends=True))
		if not markdown:
			cell.update(execution_count=None, outputs=[])
		cells.append(cell)
	previous_code = [cell for cell in previous.get('cells', []) if cell['cell_type'] == 'code']
	current_code = [cell for cell in cells if cell['cell_type'] == 'code']
	unchanged = [cell['source'] for cell in previous_code] == [cell['source'] for cell in current_code]
	if unchanged:
		for cell, old in zip(current_code, previous_code):
			cell.update(execution_count=old.get('execution_count'), outputs=old.get('outputs', []))
	notebook = dict(cells=cells, nbformat=4, nbformat_minor=5, metadata={
		'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
		'language_info': {'name': 'python', 'version': '3.12'},
		'colab': {'provenance': []},
		'gc2d': {**(previous.get('metadata', {}).get('gc2d', {}) if unchanged else {}),
			'source_sha256': hashlib.sha256(source_text.encode()).hexdigest()},
	})
	path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
	return path


if __name__ == '__main__':
	print(build_notebook(Path(__file__).resolve().parent))
