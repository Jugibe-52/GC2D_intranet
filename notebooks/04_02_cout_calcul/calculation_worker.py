"""Importable Modal worker: numerical dependencies exist only in its container."""
from hashlib import sha256
import io
import json
from pathlib import Path
import tempfile
import zipfile

import modal

ROOT = Path(__file__).resolve().parent
image = (modal.Image.debian_slim(python_version='3.12')
    .pip_install('numpy==1.26.4', 'scipy==1.17.1', 'h5py==3.16.0',
        'threadpoolctl==3.6.0', 'jax==0.7.1', 'jaxlib==0.7.1')
    .env({'PYTHONPATH': '/opt/gc2d/source/src', 'JAX_ENABLE_X64': 'true',
        'JAX_PLATFORMS': 'cpu', 'JAX_ENABLE_COMPILATION_CACHE': 'false',
        'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'})
    .add_local_dir(ROOT / 'kit', '/opt/gc2d')
    .add_local_python_source('calculation_runtime'))
app = modal.App('gc2d-colab-method-timings')


@app.function(image=image, cpu=(2., 2.), memory=(4096, 8192), timeout=14400,
    retries=0, max_containers=1, scaledown_window=60)
def calculate(cycles=500, repetitions=3, run_id='reader-run'):
    """Run the frozen benchmark, then return portable result files as ZIP bytes."""
    from calculation_runtime import check_settings, verify_inputs
    from diagnostics.io.persistence import save_solution, load_solution
    from execution._modal_worker import decode_job, encode_job
    from request import SimulationRequest
    from studies.method_timings import benchmark_methods, METHODS
    import numpy as np

    settings = check_settings(cycles, repetitions, run_id)
    kit = Path('/opt/gc2d')
    inputs = verify_inputs(kit)
    payloads = {name: (kit / (name + '.payload')).read_bytes() for name in METHODS}
    if cycles != 500:
        # Short checks reuse exactly the physical model, methods and fixed step.
        schedule = SimulationRequest.uniform(t_span=(0., float(cycles)), sample_count=cycles * 10 + 1)
        for name, payload in payloads.items():
            problem, method, options, observers = decode_job(payload)
            payloads[name] = encode_job(problem.with_request(schedule), method, options, observers)
    result = benchmark_methods(payloads, repetitions)
    solutions = result.pop('solutions')
    record = json.loads((kit / 'study.json').read_text())
    record.pop('bucket', None)
    record.update(settings, reference_run_id=inputs['reference_run_id'],
        storage='ZIP download to the reader computer', reference_configuration=cycles == 500 and repetitions == 3,
        original_input_files=inputs['files'])
    with tempfile.TemporaryDirectory(prefix='gc2d-results-') as directory:
        output = Path(directory)
        for name, solution in solutions.items():
            if solution.states.shape != (cycles * 10 + 1, 80) or solution.diagnostics['step_count'] != cycles * 40:
                raise ValueError('Incomplete trajectory.')
            metadata = json.loads((kit / (name + '.json')).read_text())
            metadata['config']['cycles'] = cycles
            metadata.update(repetitions=repetitions, run_id=run_id, payload_sha256=sha256(payloads[name]).hexdigest(),
                environment=result['environment'], original_payload_sha256=inputs['files'][name + '.payload']['sha256'])
            save_solution(solution, output / 'runs' / name, metadata=metadata)
            restored = load_solution(output / 'runs' / name)
            np.testing.assert_array_equal(restored.solution.states, solution.states)
        for filename, value in [('study.json', record), ('timings.json', result), ('inputs.json', inputs)]:
            (output / filename).write_text(json.dumps(value, indent=2) + '\n')
        (output / 'source.tar.gz').write_bytes((kit / 'source.tar.gz').read_bytes())
        files = {p.relative_to(output).as_posix(): p.read_bytes() for p in sorted(output.rglob('*')) if p.is_file()}
        manifest = {'schema_version': 1, 'files': {name: dict(bytes=len(data), sha256=sha256(data).hexdigest())
            for name, data in files.items()}}
        files['manifest.json'] = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name, data in files.items():
                archive.writestr(name, data)
        return buffer.getvalue()
