"""Execute both notebooks, validate outputs and export a private S3 archive."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import traceback

import boto3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--bucket', required=True)
    parser.add_argument('--region', default='eu-central-1')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from study_io import atomic_json, digest, validate_run_id

    run = validate_run_id(args.run_id)
    client = boto3.client('s3', region_name=args.region)
    prefix = f'runs/{run}'
    status = {'run_id': run, 'status': 'running',
              'started_utc': datetime.now(timezone.utc).isoformat()}
    status_path = root / 'aws' / 'cloud_status.json'

    def publish_status():
        atomic_json(status_path, status)
        client.upload_file(str(status_path), args.bucket, f'{prefix}/cloud_status.json')

    publish_status()
    try:
        for stage in ('calculate', 'visualize'):
            status['stage'] = stage
            publish_status()
            subprocess.run([str(root / '.venv/bin/python'), '-u', 'run.py', stage,
                            '--run-id', run, '--timeout', '18000'], cwd=root, check=True)
        numeric = root / 'resultados' / run
        figures = root / 'figuras' / run
        for folder in (numeric, figures):
            assert json.loads((folder / 'execution_status.json').read_text())['status'] == 'success'
        manifest = json.loads((numeric / 'COMPLETE.json').read_text())
        for name, expected in manifest['sha256'].items():
            assert digest(numeric / name) == expected
        metadata = json.loads((numeric / 'metadata.json').read_text())
        assert (metadata['particle_count'], metadata['cycles'], metadata['steps_per_cycle'],
                metadata['process_count']) == (35, 5000, 20, 8)
        assert len({w['pid'] for w in metadata['workers']}) == 8
        assert metadata['simultaneous_integration_seconds'] > 0
        archive = root.parent / f'{run}_results.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            for path in (numeric, figures, root / 'resultados/latest_success.json'):
                tar.add(path, arcname=str(path.relative_to(root)))
        checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
        sidecar = archive.with_name(archive.name + '.sha256')
        sidecar.write_text(checksum + '  ' + archive.name + '\n')
        for path in (archive, sidecar):
            client.upload_file(str(path), args.bucket, f'{prefix}/{path.name}')
        status.update(status='success', stage='uploaded', archive=archive.name,
                      archive_bytes=archive.stat().st_size, archive_sha256=checksum,
                      result_s3_uri=f's3://{args.bucket}/{prefix}/{archive.name}',
                      runtime_seconds=metadata['runtime_seconds'], workers=metadata['workers'],
                      simultaneous_integration_seconds=metadata['simultaneous_integration_seconds'],
                      finished_utc=datetime.now(timezone.utc).isoformat())
        publish_status()
        print(json.dumps(status, indent=2), flush=True)
    except BaseException as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}',
                      finished_utc=datetime.now(timezone.utc).isoformat())
        traceback.print_exc()
        publish_status()
        raise


if __name__ == '__main__':
    main()
