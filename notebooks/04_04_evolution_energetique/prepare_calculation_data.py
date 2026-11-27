"""Build an immutable section 4.4 kit without changing historical numerical code."""
from hashlib import file_digest, sha256
import copy
import io
import json
from pathlib import Path
import tarfile

RUN_ID = 'main-20261117-rho-div20-online-balance-20261120'
ROOT = Path(__file__).resolve().parents[2]
THESIS = ROOT.parent / 'TFM_redaccion'
ORIGINAL = THESIS / 'notebooks/04_04_02_balance_energetico_en_linea'
SCRIPT = 'TFM_redaccion/notebooks/04_04_02_balance_energetico_en_linea/calculo.py'
REFERENCE = 'TFM_redaccion/datos/campana_referencia.json'


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def normalized_metadata(value):
    """Remove the author's local source filename, leaving all other values intact."""
    if isinstance(value, dict):
        return {key: ('/data/potential/V1/PHI_2.h5' if key == 'source_path' else normalized_metadata(item))
            for key, item in value.items()}
    if isinstance(value, list):
        return [normalized_metadata(item) for item in value]
    return value


def prepare(destination):
    reference = json.loads((THESIS / 'datos/campana_referencia.json').read_text())
    campaign = ROOT / reference['campaign_path']
    original_script = ORIGINAL / 'calculo.py'
    archive = campaign / 'source.tar.gz'
    potential = ROOT / 'data/potential/V1/PHI_2.h5'
    if digest(potential) != reference['source_sha256']:
        raise ValueError('Physical potential differs from the reference.')
    source_sha = digest(archive)
    # The historical source archive already has anonymous deterministic headers.
    with tarfile.open(archive) as source:
        members = source.getmembers()
        if any(not m.isfile() or m.name.startswith('/') or '..' in Path(m.name).parts
                or m.uname or m.gname or m.uid or m.gid for m in members):
            raise ValueError('Unexpected or personal frozen source archive metadata.')
    files = {SCRIPT: original_script, REFERENCE: THESIS / 'datos/campana_referencia.json',
        'GC2D_codigo/' + reference['campaign_path'] + '/source.tar.gz': archive,
        'GC2D_codigo/data/potential/V1/PHI_2.h5': potential}
    objects, proofs = {}, {}
    for method in reference['methods']:
        input_record = ORIGINAL / 'inputs' / (method + '.inputs.json')
        payload = ORIGINAL / 'inputs' / (method + '.inputs.npz')
        receipt_path = ORIGINAL / 'recovery' / RUN_ID / (method + '.json')
        receipt = json.loads(receipt_path.read_text())
        record = json.loads(input_record.read_text())
        for key, value in {'script_sha256': digest(original_script), 'reference_sha256': digest(files[REFERENCE]),
                'source_archive_sha256': source_sha, 'input_bundle_sha256': digest(payload),
                'input_bundle_receipt_sha256': digest(input_record), 'run_id': RUN_ID, 'cycles': 5000}.items():
            if receipt[key] != value:
                raise ValueError(f'Historical receipt mismatch: {method}/{key}')
        if (record['source_archive_sha256'] != source_sha
                or record['input_bundle_sha256'] != digest(payload)
                or record['reference_sha256'] != digest(files[REFERENCE])):
            raise ValueError('Input bundle provenance mismatch.')
        public = normalized_metadata(record)
        restored = copy.deepcopy(public)
        # A recursive normalization is auditable by structural equality, including
        # every numerical value and all archived temporal-plan fingerprints.
        def restore_source_paths(public_value, original_value):
            if isinstance(original_value, dict):
                for key, item in original_value.items():
                    if key == 'source_path': public_value[key] = item
                    else: restore_source_paths(public_value[key], item)
            elif isinstance(original_value, list):
                for p, o in zip(public_value, original_value): restore_source_paths(p, o)
        restore_source_paths(restored, record)
        if restored != record:
            raise ValueError('Metadata normalization changed a scientific value.')
        public_bytes = (json.dumps(public, sort_keys=True, indent=2) + '\n').encode()
        if b'/home/' in public_bytes or b'juagil' in public_bytes:
            raise ValueError('Personal path remains in public metadata.')
        objects['inputs/' + method + '.inputs.json'] = public_bytes
        files['inputs/' + method + '.inputs.npz'] = payload
        proofs[method] = {'original_input_receipt_sha256': digest(input_record),
            'public_input_receipt_sha256': sha256(public_bytes).hexdigest(),
            'historical_result_receipt_sha256': digest(receipt_path),
            'input_bundle_sha256': digest(payload), 'temporal_plan_sha256': receipt['temporal_plan_sha256']}
    manifest = dict(schema_version=1, reference_run_id=RUN_ID, methods=reference['methods'],
        script=SCRIPT, reference=REFERENCE, source='GC2D_codigo/' + reference['campaign_path'] + '/source.tar.gz',
        script_sha256=digest(original_script), source_code_sha256=reference['code_sha256'],
        source_tar_sha256=source_sha, source_sha256=reference['source_sha256'],
        normalization='Only source_path strings in input JSON become /data/potential/V1/PHI_2.h5; original source code, reference, physical HDF5 and NPZ arrays are byte-identical',
        provenance=proofs,
        files={name: dict(bytes=path.stat().st_size, sha256=digest(path)) for name, path in files.items()})
    manifest['files'].update({name: dict(bytes=len(data), sha256=sha256(data).hexdigest()) for name, data in objects.items()})
    objects['inputs.json'] = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / 'calculation-kit.tmp.tar.xz'
    with tarfile.open(temporary, mode='w:xz', preset=1, format=tarfile.USTAR_FORMAT) as packed:
        for name in sorted(set(files) | set(objects)):
            entry = tarfile.TarInfo(name)
            entry.mode = 0o644
            if name in files:
                entry.size = files[name].stat().st_size
                with files[name].open('rb') as stream: packed.addfile(entry, stream)
            else:
                entry.size = len(objects[name])
                packed.addfile(entry, io.BytesIO(objects[name]))
    record = dict(schema_version=1, file='calculation-kit.tar.xz', bytes=temporary.stat().st_size,
        sha256=digest(temporary), reference_run_id=RUN_ID, script_sha256=manifest['script_sha256'],
        source_code_sha256=reference['code_sha256'])
    target = destination / record['file']
    if target.exists() and digest(target) != record['sha256']:
        raise FileExistsError('Choose a new version; the existing kit differs.')
    temporary.replace(target)
    record['parts'] = split_kit(target)
    manifest_path = destination / 'manifest.json'
    public_manifest = (json.dumps(record, sort_keys=True, indent=2) + '\n').encode()
    if manifest_path.exists() and manifest_path.read_bytes() != public_manifest:
        raise FileExistsError('Choose a new version; the existing manifest differs.')
    manifest_path.write_bytes(public_manifest)
    return record


def split_kit(path):
    """Keep each upload below 500 MB while preserving the complete tar bytes."""
    parts = []
    with path.open('rb') as source:
        while data := source.read(200 * 1024 * 1024):
            target = path.with_name(path.name + f'.part{len(parts) + 1:03d}')
            if target.exists() and target.read_bytes() != data:
                raise FileExistsError('An existing public kit part differs.')
            target.write_bytes(data)
            parts.append(dict(file=target.name, bytes=len(data), sha256=sha256(data).hexdigest()))
    return parts


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    print(json.dumps(prepare(parser.parse_args().destination), indent=2))
