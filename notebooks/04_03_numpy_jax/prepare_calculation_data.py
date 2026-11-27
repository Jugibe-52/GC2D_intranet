"""Build a public, immutable kit from the saved section 4.3 inputs."""
from hashlib import sha256
import io
import gzip
import json
from pathlib import Path, PurePosixPath
import tarfile
import subprocess
import sys
import tempfile

METHODS = ('BM4Implicit', 'GaussLegendre4', 'RK2')
CYCLES = (10, 30, 90)
REFERENCE = 'methods-20261119-union16-rho0015-c10-30-90'

SANITIZE_SCRIPT = r'''
import dataclasses, hashlib, json, pickle, sys
from collections.abc import Mapping
from pathlib import Path
import numpy as np
root = Path(sys.argv[1])
sys.path.insert(0, str(root / 'source/src'))
from execution._modal_worker import decode_job, encode_job

def snapshot(value, active=None):
    # Compare every numerical array and object field after source-path removal.
    if active is None: active = set()
    if isinstance(value, np.ndarray):
        return ('array', value.dtype.str, value.shape, hashlib.sha256(value.tobytes()).hexdigest())
    if isinstance(value, np.generic): return snapshot(value.item(), active)
    if isinstance(value, Path): return ('path', '/data/potential/V1/PHI_2.h5')
    if value is None or isinstance(value, (str, bytes, int, float, complex, bool)): return value
    if isinstance(value, type): return ('type', value.__module__, value.__qualname__)
    if id(value) in active: raise ValueError('Unexpected input-object cycle.')
    active = active | {id(value)}
    if isinstance(value, Mapping): return {k: snapshot(v, active) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return (type(value).__name__, tuple(snapshot(v, active) for v in value))
    if dataclasses.is_dataclass(value): fields = {f.name: getattr(value, f.name, ('unset-dataclass-field',)) for f in dataclasses.fields(value)}
    elif hasattr(value, '__dict__'): fields = vars(value)
    elif hasattr(value, '__slots__'):
        fields = {}
        for cls in type(value).__mro__:
            slots = getattr(cls, '__slots__', ())
            for name in ((slots,) if isinstance(slots, str) else slots):
                if name not in ('__weakref__', '__dict__') and hasattr(value, name):
                    fields[name] = getattr(value, name)
    else: raise TypeError(f'Unsupported comparison type: {type(value)}')
    return (type(value).__module__, type(value).__qualname__, snapshot(fields, active))

for path in sorted(root.glob('*.payload')):
    original = path.read_bytes()
    job = decode_job(original)
    before = snapshot(job)
    original_path = job[0].dynamics.potential.metadata.source_path
    baseline = encode_job(*job)
    object.__setattr__(job[0].dynamics.potential.metadata, 'source_path', Path('/data/potential/V1/PHI_2.h5'))
    public = encode_job(*job)
    assert snapshot(decode_job(public)) == before, 'A numerical input changed during normalization.'
    object.__setattr__(job[0].dynamics.potential.metadata, 'source_path', original_path)
    assert encode_job(*job) == baseline, 'Normalization modified another serialized input.'
    assert b'juagil@cartif.local' not in public and b'/home/' not in public
    path.write_bytes(public)
'''


def prepare(archive: Path, destination: Path) -> dict:
    """Cross-check original receipts, paired results and the frozen source digest."""
    study_bytes = (archive / 'study.json').read_bytes()
    study = json.loads(study_bytes)
    publication = json.loads((archive / 'publication.json').read_text())
    cases = [f'{method}-{cycles}' for cycles in CYCLES for method in METHODS]
    if (study['run_id'] != REFERENCE or study['methods'] != list(METHODS)
            or study['cycles'] != list(CYCLES) or study['particle_count'] != 16
            or study['rho'] != .015 or study['repetitions'] != 3
            or publication['status'] != 'verified'
            or set(publication['verified_cases']) != {f'{case}/{b}' for case in cases for b in ('scipy', 'jax')}):
        raise ValueError('Unexpected or incomplete reference study.')
    objects = {'study.json': study_bytes, 'source.tar.gz': (archive / 'source.tar.gz').read_bytes()}
    proofs = {}
    for case in cases:
        payload = (archive / f'{case}.payload').read_bytes()
        metadata_bytes = (archive / f'{case}.json').read_bytes()
        metadata = json.loads(metadata_bytes)
        receipt_bytes = (archive / 'receipts' / f'{case}.json').read_bytes()
        pair_bytes = (archive / 'runs' / case / 'pair.json').read_bytes()
        receipt, pair = json.loads(receipt_bytes), json.loads(pair_bytes)
        digest = sha256(payload).hexdigest()
        if (receipt['submission'] != 'confirmed' or digest != receipt['input_sha256']
                or digest != pair['payload_sha256'] or metadata['code_sha256'] != study['code_sha256']
                or metadata['particle_count'] != 16 or metadata['rho'] != .015
                or f"{metadata['method']}-{metadata['config']['cycles']}" != case):
            raise ValueError(f'Original input provenance mismatch: {case}')
        objects[f'{case}.payload'] = payload
        objects[f'{case}.json'] = metadata_bytes
        # Retain hashes of local proofs without publishing personal Modal call IDs.
        proofs[case] = dict(payload_sha256=digest, receipt_sha256=sha256(receipt_bytes).hexdigest(),
            pair_sha256=sha256(pair_bytes).hexdigest())
    extras = {'examples/modal_bm4_numpy_jax_app.py', 'constraints.txt', 'pyproject.toml',
        'data/particles/chaotic_candidates.json', 'data/particles/periodic_candidates_8.json',
        'data/particles/expanded_84.json', 'campana_referencia.json'}
    original_source_sha256 = sha256(objects['source.tar.gz']).hexdigest()
    source_objects = {}
    with tarfile.open(fileobj=io.BytesIO(objects['source.tar.gz'])) as source:
        members = source.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)):
            raise ValueError('Duplicated frozen source member.')
        for member in members:
            path = PurePosixPath(member.name)
            if (not member.isfile() or path.is_absolute() or '..' in path.parts or '\\' in member.name
                    or not (member.name.startswith('src/') and member.name.endswith('.py')
                        or member.name in extras)):
                raise ValueError(f'Unexpected source member: {member.name}')
        if not extras.issubset(names):
            raise ValueError('Incomplete source snapshot.')
        digest = sha256()
        code_names = sorted(n for n in names if n.startswith('src/'))
        code_names += ['examples/modal_bm4_numpy_jax_app.py', 'constraints.txt', 'pyproject.toml']
        for name in code_names:
            digest.update(name.encode())
            digest.update(source.extractfile(name).read())
        if digest.hexdigest() != study['code_sha256']:
            raise ValueError('The source snapshot differs from the original code fingerprint.')
        for selection in study['particle_selection']['source_sets']:
            data = source.extractfile('data/particles/' + selection['source_file']).read()
            if sha256(data).hexdigest() != selection['sha256']:
                raise ValueError('Particle selection fingerprint mismatch.')
        source_objects = {name: source.extractfile(name).read() for name in names}
    # Keep every scientific/source byte while dropping author account metadata
    # from tar headers and the variable timestamp in the gzip envelope.
    sanitized_source = io.BytesIO()
    with gzip.GzipFile(fileobj=sanitized_source, mode='wb', mtime=0, filename='') as compressed:
        with tarfile.open(fileobj=compressed, mode='w', format=tarfile.USTAR_FORMAT) as source:
            for name, data in sorted(source_objects.items()):
                entry = tarfile.TarInfo(name)
                entry.size, entry.mode = len(data), 0o644
                source.addfile(entry, io.BytesIO(data))
    objects['source.tar.gz'] = sanitized_source.getvalue()
    # The trusted pickle carries only one local path, in potential provenance.
    # Normalize it in an isolated interpreter using the original frozen classes.
    with tempfile.TemporaryDirectory(prefix='gc2d-public-inputs-') as directory:
        root = Path(directory)
        for name, data in source_objects.items():
            target = root / 'source' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        for name, data in objects.items():
            if name.endswith('.payload'):
                (root / name).write_bytes(data)
        subprocess.run([sys.executable, '-I', '-c', SANITIZE_SCRIPT, str(root)], check=True, timeout=180)
        for name in list(objects):
            if name.endswith('.payload'):
                objects[name] = (root / name).read_bytes()
                proofs[name.removesuffix('.payload')]['public_payload_sha256'] = sha256(objects[name]).hexdigest()
    manifest = dict(schema_version=1, reference_run_id=REFERENCE, provenance=proofs,
        normalization='Only potential.metadata.source_path becomes /data/potential/V1/PHI_2.h5; all object fields and numerical array bytes compared after decode; source member bytes unchanged',
        source_code_sha256=study['code_sha256'],
        original_source_tar_sha256=original_source_sha256,
        source_members={name: dict(bytes=len(data), sha256=sha256(data).hexdigest())
            for name, data in sorted(source_objects.items())},
        files={name: dict(bytes=len(data), sha256=sha256(data).hexdigest()) for name, data in objects.items()})
    objects['inputs.json'] = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
    packed = io.BytesIO()
    with tarfile.open(fileobj=packed, mode='w:xz', preset=6) as kit:
        for name, data in sorted(objects.items()):
            entry = tarfile.TarInfo(name)
            entry.size, entry.mode = len(data), 0o644
            kit.addfile(entry, io.BytesIO(data))
    data = packed.getvalue()
    record = dict(schema_version=1, file='calculation-kit.tar.xz', bytes=len(data),
        sha256=sha256(data).hexdigest(), reference_run_id=REFERENCE,
        source_code_sha256=study['code_sha256'], source_tar_sha256=manifest['files']['source.tar.gz']['sha256'])
    outputs = {'calculation-kit.tar.xz': data,
        'manifest.json': (json.dumps(record, indent=2, sort_keys=True) + '\n').encode()}
    destination.mkdir(parents=True, exist_ok=True)
    for name, value in outputs.items():
        target = destination / name
        if target.exists() and target.read_bytes() != value:
            raise FileExistsError('Choose a new version; existing public data differs.')
        target.write_bytes(value)
    return record


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--archive', type=Path, default=Path(__file__).resolve().parents[2] /
        f'outputs/studies/bm4_numpy_jax/{REFERENCE}')
    args = parser.parse_args()
    print(json.dumps(prepare(args.archive, args.destination), indent=2))
