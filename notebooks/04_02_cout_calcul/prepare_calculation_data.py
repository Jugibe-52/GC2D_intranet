"""Build an immutable public reproduction kit from the verified timing archive."""
from hashlib import sha256
import io
import json
from pathlib import Path
import tarfile


def prepare(archive: Path, destination: Path) -> dict:
    original = (archive / 'manifest.json').read_bytes()
    manifest = json.loads(original)
    publication = json.loads((archive / 'publication.json').read_text())
    if publication['status'] != 'verified' or publication['manifest_sha256'] != sha256(original).hexdigest():
        raise ValueError('The original archive is not verified.')
    study = json.loads((archive / 'study.json').read_text())
    receipt = json.loads((archive / 'receipt.json').read_text())
    if study['run_id'] != 'jax-20261118-rho0015-c500' or receipt['submission'] != 'confirmed':
        raise ValueError('Unexpected reference study.')
    names = ['source.tar.gz', 'study.json']
    names += [name + suffix for name in study['methods'] for suffix in ('.json', '.payload')]
    objects = {}
    for name in names:
        data = (archive / name).read_bytes()
        fingerprint = {'bytes': len(data), 'sha256': sha256(data).hexdigest()}
        if name.endswith('.payload'):
            method = name.removesuffix('.payload')
            metadata = json.loads((archive / (method + '.json')).read_text())
            if fingerprint['sha256'] != receipt['payload_sha256'][method] or fingerprint['sha256'] != metadata['payload_sha256']:
                raise ValueError(f'Input fingerprint mismatch: {name}')
        elif fingerprint != manifest['files'][name]:
            raise ValueError(f'Archive fingerprint mismatch: {name}')
        objects[name] = data
    # The frozen source has only code, dependency declarations and scientific JSON.
    with tarfile.open(fileobj=io.BytesIO(objects['source.tar.gz'])) as source:
        for member in source.getmembers():
            if not member.isfile() or not (
                member.name.startswith('src/') and member.name.endswith('.py')
                or member.name in ('examples/modal_method_timings_app.py', 'constraints.txt',
                    'pyproject.toml', 'data/particles/original_40.json', 'campana_referencia.json')
            ):
                raise ValueError(f'Unexpected source member: {member.name}')
    kit_manifest = dict(schema_version=1, reference_run_id=study['run_id'],
        source_manifest_sha256=sha256(original).hexdigest(),
        files={name: dict(bytes=len(data), sha256=sha256(data).hexdigest()) for name, data in objects.items()})
    objects['inputs.json'] = (json.dumps(kit_manifest, indent=2, sort_keys=True) + '\n').encode()
    packed = io.BytesIO()
    # XZ retains repeated field arrays across the eight inputs without changing them.
    with tarfile.open(fileobj=packed, mode='w:xz', preset=6) as kit:
        for name, data in sorted(objects.items()):
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            entry.mode = 0o644
            kit.addfile(entry, io.BytesIO(data))
    bundle = packed.getvalue()
    record = dict(schema_version=1, file='calculation-kit.tar.xz', bytes=len(bundle),
        sha256=sha256(bundle).hexdigest(), reference_run_id=study['run_id'],
        source_manifest_sha256=sha256(original).hexdigest())
    outputs = {'calculation-kit.tar.xz': bundle,
        'manifest.json': (json.dumps(record, indent=2, sort_keys=True) + '\n').encode()}
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in outputs.items():
        target = destination / name
        if target.exists() and target.read_bytes() != data:
            raise FileExistsError('Choose a new version; existing public data differs.')
        target.write_bytes(data)
    return record


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--archive', type=Path, default=Path(__file__).resolve().parents[2] /
        'outputs/studies/method_timings/jax-20261118-rho0015-c500')
    args = parser.parse_args()
    print(json.dumps(prepare(args.archive, args.destination), indent=2))
