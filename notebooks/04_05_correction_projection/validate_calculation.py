"""Exercise reduced real integrations and the notebook's checked ZIP path locally."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import calculation_runtime as runtime

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT.parents[1] / 'outputs/colab-public/04_05_correction_projection/calculation/v1'
BASE = ROOT.parents[1] / 'outputs/colab-public/04_04_evolution_energetique/calculation/v1'


def validate(workspace):
    workspace.mkdir(parents=True, exist_ok=True)
    kit = workspace / 'kit'
    manifest = json.loads((BASE / 'manifest.json').read_text())
    if not (kit / 'inputs.json').exists():
        runtime.prepare_kit((BASE / 'calculation-kit.tar.xz').read_bytes(), manifest['sha256'], kit)
    protocol_bytes = (PUBLIC / 'protocol.json').read_bytes()
    (kit / 'projection-protocol.json').write_bytes(protocol_bytes)
    protocol_sha = sha256(protocol_bytes).hexdigest()
    protocol, inputs = runtime.verify_protocol(kit, protocol_sha)
    settings = runtime.check_settings(1, 1, 'validation-one-cycle', [0, 6])
    output = workspace / settings['run_id']
    output.mkdir(exist_ok=True)
    environment = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
        JAX_ENABLE_X64='true', JAX_PLATFORMS='cpu')
    cases = runtime.selected_cases(protocol, settings)
    for case in cases:
        request = output / (case['case_key'] + '.request.json')
        runtime.write_json(request, dict(case=case, settings=settings, protocol_sha256=protocol_sha))
        destination = output / (case['case_key'] + '.zip')
        subprocess.run([sys.executable, str(ROOT / 'calculation_runtime.py'), '--integrate',
            str(kit), str(request), str(destination)], env=environment, check=True, timeout=300)
        result = runtime.verify_case_result(destination.read_bytes(), case, settings, protocol, protocol_sha)
        print(case['case_key'], result['status'], result.get('summary', result.get('reason')), flush=True)
    destination, summary = runtime.combine_results(output, settings, protocol, protocol_sha)
    if len(summary['cases']) != 4 or not any(c['status'] == 'complete' for c in summary['cases']):
        raise AssertionError('The reduced run did not produce meaningful complete results.')
    print('Four actual 84-particle, one-cycle integrations verified, plus combined ZIP:', destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace', type=Path)
    validate(parser.parse_args().workspace.resolve())
