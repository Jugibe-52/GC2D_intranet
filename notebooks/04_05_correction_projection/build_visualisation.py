"""Build a portable notebook with its verified, compressed reference snapshot."""
from __future__ import annotations

import argparse
import base64
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import zlib


def build_notebook(directory: Path, bundle: Path | None = None, calculation_url: str | None = None) -> Path:
	"""Preserve executed outputs only when every executable cell is unchanged."""
	repository = directory.parents[1]
	bundle = bundle or repository / 'outputs/colab-public/04_05_correction_projection/viewer/v1'
	path = directory / 'visualisation.ipynb'
	previous = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
	backup = repository / 'outputs/colab-backups/04_05_correction_projection/visualisation-before-portable.ipynb'
	if path.exists() and not backup.exists():
		backup.parent.mkdir(parents=True, exist_ok=True)
		shutil.copy2(path, backup)
	source_path = directory / 'visualisation.py'
	source_text = source_path.read_text(encoding='utf-8')
	if bundle.exists():
		files = {name: (bundle / name).read_bytes() for name in ('manifest.json', 'summary.json', 'provenance.json')}
		manifest = json.loads(files['manifest.json'])
		for name, proof in manifest['files'].items():
			if len(files[name]) != proof['bytes'] or sha256(files[name]).hexdigest() != proof['sha256']:
				raise ValueError(f'Publication bundle integrity failure: {name}.')
		payload = json.dumps({name: base64.b64encode(data).decode() for name, data in files.items()}, separators=(',', ':')).encode()
		embedded = base64.b64encode(zlib.compress(payload, 9)).decode()
		source_text = re.sub(r"EXPECTED_MANIFEST_SHA256 = '[^']*'", f"EXPECTED_MANIFEST_SHA256 = '{sha256(files['manifest.json']).hexdigest()}'", source_text)
		source_text = re.sub(r"EMBEDDED_SNAPSHOT_B64 = '[^']*'", f"EMBEDDED_SNAPSHOT_B64 = '{embedded}'", source_text)
	if 'SNAPSHOT_PLACEHOLDER' in source_text or 'MANIFEST_SHA256_PLACEHOLDER' in source_text:
		raise ValueError('Prepare the verified reference bundle before building the notebook.')
	if calculation_url:
		if not calculation_url.startswith('https://colab.research.google.com/drive/'):
			raise ValueError('The paired notebook link must be a Google Colab Drive URL.')
		source_text = re.sub(r'(?<=\[calculation notebook\]\()[^)]*', calculation_url, source_text)
	if source_path.read_text(encoding='utf-8') != source_text:
		source_path.write_text(source_text, encoding='utf-8')
	cells = []
	for block in source_text.split('# %%')[1:]:
		header, source = block.split('\n', 1)
		markdown = header.strip() == '[markdown]'
		if markdown:
			source = source.replace('[calculation notebook](CALCULATION_COLAB_URL)', 'calculation notebook')
			source = '\n'.join(line[2:] if line.startswith('# ') else line[1:] if line.startswith('#') else line for line in source.splitlines())
		source = source.strip() + '\n'
		cell = dict(cell_type='markdown' if markdown else 'code', metadata={}, id=sha256(source.encode()).hexdigest()[:12], source=source.splitlines(keepends=True))
		if not markdown:
			cell.update(execution_count=None, outputs=[])
		cells.append(cell)
	old_code = [cell for cell in previous.get('cells', []) if cell['cell_type'] == 'code']
	new_code = [cell for cell in cells if cell['cell_type'] == 'code']
	unchanged = [cell['source'] for cell in old_code] == [cell['source'] for cell in new_code]
	if unchanged:
		for cell, old in zip(new_code, old_code):
			cell.update(execution_count=old.get('execution_count'), outputs=old.get('outputs', []))
	notebook = dict(cells=cells, nbformat=4, nbformat_minor=5, metadata={
		'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
		'language_info': {'name': 'python', 'version': '3.12'}, 'colab': {'provenance': []},
		'gc2d': {**(previous.get('metadata', {}).get('gc2d', {}) if unchanged else {}), 'source_sha256': sha256(source_text.encode()).hexdigest()},
	})
	path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
	return path


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--bundle', type=Path)
	parser.add_argument('--calculation-url')
	args = parser.parse_args()
	print(build_notebook(Path(__file__).resolve().parent, args.bundle, args.calculation_url))
