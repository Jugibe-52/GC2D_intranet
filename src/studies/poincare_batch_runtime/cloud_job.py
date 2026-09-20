"""Run both notebooks and synchronize immutable checkpoints to private S3."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import threading
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from study_io import atomic_json, digest, utc_now, validate_run_id


class CheckpointStore:
    """Publish data before commit markers; restore only verified committed pairs."""
    def __init__(self, client, bucket, prefix, directory):
        self.client, self.bucket = client, bucket
        self.prefix = prefix.rstrip('/') + '/checkpoints/'
        self.directory = Path(directory)
        self.uploaded = set()

    def restore(self):
        pages = self.client.get_paginator('list_objects_v2').paginate(Bucket=self.bucket, Prefix=self.prefix)
        keys = [obj['Key'] for page in pages for obj in page.get('Contents', [])]
        for key in sorted(keys):
            relative = key[len(self.prefix):]
            if not re.fullmatch(r'worker_\d{2}/chunk_\d{8}\.json', relative):
                continue
            marker = self.directory / relative
            marker.parent.mkdir(parents=True, exist_ok=True)
            temporary = marker.with_suffix('.json.restore')
            self.client.download_file(self.bucket, key, str(temporary))
            record = json.loads(temporary.read_text())
            file = marker.with_suffix('.npz')
            assert record['file'] == file.name
            if not file.exists() or digest(file) != record['sha256']:
                data_tmp = file.with_suffix('.npz.restore')
                self.client.download_file(self.bucket, key[:-5] + '.npz', str(data_tmp))
                assert digest(data_tmp) == record['sha256'], 'Remote checkpoint checksum mismatch.'
                data_tmp.replace(file)
            if marker.exists():
                assert json.loads(marker.read_text()) == record, 'Local and remote checkpoint conflict.'
            temporary.replace(marker)
            self.uploaded.add(relative)

    def sync(self):
        completed = [0] * 16
        for marker in sorted(self.directory.glob('worker_*/chunk_*.json')):
            relative = marker.relative_to(self.directory).as_posix()
            assert re.fullmatch(r'worker_\d{2}/chunk_\d{8}\.json', relative)
            record = json.loads(marker.read_text())
            worker = int(marker.parent.name.split('_')[1])
            assert 1 <= worker <= 16
            if relative not in self.uploaded:
                file = marker.with_suffix('.npz')
                assert file.name == record['file'] and digest(file) == record['sha256']
                self.client.upload_file(str(file), self.bucket, self.prefix + relative[:-5] + '.npz')
                self.client.upload_file(str(marker), self.bucket, self.prefix + relative)
                self.uploaded.add(relative)
            completed[worker - 1] = max(completed[worker - 1], record['end_step'])
        return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--bucket', required=True)
    parser.add_argument('--region', default='eu-central-1')
    args = parser.parse_args()
    import boto3
    from botocore.config import Config
    run = validate_run_id(args.run_id)
    client = boto3.client('s3', region_name=args.region,
                         config=Config(retries={'max_attempts':5,'mode':'standard'},
                                       connect_timeout=10, read_timeout=30))
    prefix = f'runs/{run}'
    numeric, figures = ROOT/'resultados'/run, ROOT/'figuras'/run
    status_path = ROOT/'aws/cloud_status.json'
    cfg = json.loads((ROOT/'experiment.json').read_text())
    nsteps = cfg['cycles'] * cfg['steps_per_cycle']
    status = {'run_id':run, 'status':'running', 'started_utc':utc_now(),
              'particles':len(cfg['particle_ids']), 'workers':16, 'cycles':cfg['cycles'], 'steps_per_cycle':cfg['steps_per_cycle'], 'method':cfg['method']}
    store = CheckpointStore(client, args.bucket, prefix, numeric/'checkpoints')
    stopped = threading.Event()
    sync_thread = None

    def publish_status():
        atomic_json(status_path, status)
        client.upload_file(str(status_path), args.bucket, prefix+'/cloud_status.json')

    def sync_progress():
        completed = store.sync()
        progress = {'run_id':run, 'updated_utc':utc_now(), 'worker_completed_steps':completed,
                    'total_steps_per_worker':nsteps, 'minimum_completed_cycles':min(completed)/cfg['steps_per_cycle'],
                    'maximum_completed_cycles':max(completed)/cfg['steps_per_cycle'],
                    'average_percent':100*sum(completed)/16/nsteps,
                    'committed_remote_chunks':len(store.uploaded)}
        path = ROOT/'aws/cloud_progress.json'
        atomic_json(path, progress)
        client.upload_file(str(path), args.bucket, prefix+'/cloud_progress.json')
        log = numeric/'execution.log'
        if log.exists():
            client.upload_file(str(log), args.bucket, prefix+'/calculation.log')

    def background_sync():
        while not stopped.wait(30):
            try:
                sync_progress()
            except Exception:
                traceback.print_exc()  # Retry on the next interval; local checkpoints remain intact.

    try:
        # Complete local runs are verified by run.py; no reintegration occurs.
        publish_status()
        store.restore()
        sync_progress()
        sync_thread = threading.Thread(target=background_sync, daemon=True)
        sync_thread.start()
        for stage in ('calculate', 'visualize'):
            status['stage'] = stage
            publish_status()
            command = [str(ROOT/'.venv/bin/python'), '-u', 'run.py', stage,
                       '--run-id',run,'--timeout',str(cfg['deadline_hours']*3600)]
            if stage == 'calculate': command.append('--resume')
            subprocess.run(command, cwd=ROOT, check=True)
        stopped.set()
        sync_thread.join()
        sync_progress()
        for directory in (numeric, figures):
            assert json.loads((directory/'execution_status.json').read_text())['status'] == 'success'
        manifest = json.loads((numeric/'COMPLETE.json').read_text())
        for name, expected in manifest['sha256'].items():
            assert digest(numeric/name) == expected
        meta = json.loads((numeric/'metadata.json').read_text())
        assert (meta['particle_count'],meta['cycles'],meta['steps_per_cycle'],meta['process_count']) == (len(cfg['particle_ids']),cfg['cycles'],cfg['steps_per_cycle'],16)
        assert meta['method'] == cfg['method'] and meta['particle_ids'] == cfg['particle_ids']
        assert meta['newton_history_recorded'] == (cfg['method'] == 'BM4Implicit')
        assert not meta['reference_computed']
        assert len(store.uploaded) == 16 * ((nsteps + cfg['checkpoint_steps'] - 1)//cfg['checkpoint_steps'])
        archive = ROOT.parent/f'{run}_results.tar.gz'
        with tarfile.open(archive,'w:gz') as tar:
            for path in (numeric, figures, ROOT/'resultados/latest_success.json'):
                tar.add(path, arcname=str(path.relative_to(ROOT)),
                        filter=lambda member: None if 'checkpoints' in Path(member.name).parts else member)
        checksum = digest(archive)
        sidecar = archive.with_name(archive.name+'.sha256')
        sidecar.write_text(checksum+'  '+archive.name+'\n')
        for path in (archive,sidecar):
            client.upload_file(str(path),args.bucket,prefix+'/'+path.name)
        status.update(status='success',stage='uploaded',archive=archive.name,
                      archive_bytes=archive.stat().st_size,archive_sha256=checksum,
                      result_s3_uri=f's3://{args.bucket}/{prefix}/{archive.name}',
                      runtime_seconds=meta['runtime_seconds'],workers=meta['workers'],finished_utc=utc_now())
        publish_status()
        print(json.dumps(status,indent=2),flush=True)
    except BaseException as error:
        stopped.set()
        if sync_thread is not None: sync_thread.join()
        status.update(status='failed',error=f'{type(error).__name__}: {error}',finished_utc=utc_now())
        traceback.print_exc()
        try: sync_progress()
        finally: publish_status()
        raise


if __name__ == '__main__':
    main()
