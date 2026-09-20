"""Offline verification of S3 commit ordering and restoration after disk loss."""
import json
from pathlib import Path
import sys
import tempfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from aws.cloud_job import CheckpointStore
from study_io import digest, atomic_json, utc_now

class FakeS3:
    def __init__(self): self.objects = {}; self.calls = []
    def upload_file(self,file,bucket,key):
        if key.endswith('.json'):
            assert key[:-5]+'.npz' in self.objects
        self.objects[key] = Path(file).read_bytes(); self.calls.append(key)
    def download_file(self,bucket,key,file): Path(file).write_bytes(self.objects[key])
    def get_paginator(self,name): return self
    def paginate(self,**kwargs):
        return [{'Contents':[{'Key':key} for key in self.objects if key.startswith(kwargs['Prefix'])]}]

client=FakeS3()
source=ROOT/'resultados/checkpoint_resume_test/checkpoints'
meta=json.loads((source.parent/'metadata.json').read_text())
steps=meta['complete_steps']
pairs=len(list(source.glob('worker_*/chunk_*.json')))
store=CheckpointStore(client,'test','runs/test',source)
assert store.sync()==[steps]*16
assert len(client.calls)==2*pairs
store.sync()
assert len(client.calls)==2*pairs
with tempfile.TemporaryDirectory() as directory:
    restored=CheckpointStore(client,'test','runs/test',directory)
    restored.restore()
    for marker in source.glob('worker_*/chunk_*.json'):
        target=Path(directory)/marker.relative_to(source)
        assert target.read_bytes()==marker.read_bytes()
        assert digest(target.with_suffix('.npz'))==digest(marker.with_suffix('.npz'))
    assert restored.sync()==[steps]*16 and len(client.calls)==2*pairs
    # Corrupt remote data must never become a committed local checkpoint.
    key=next(k for k in client.objects if k.endswith('.npz'))
    client.objects[key]=b'corrupted payload'
    with tempfile.TemporaryDirectory() as damaged:
        try: CheckpointStore(client,'test','runs/test',damaged).restore()
        except AssertionError: pass
        else: raise AssertionError('Corrupt checkpoint accepted')
        assert not (Path(damaged)/key.split('/checkpoints/')[1]).with_suffix('.json').exists()
atomic_json(ROOT/'aws/validation_cloud_store.json',{
    'verified_utc':utc_now(),'committed_pairs':pairs,'upload_data_before_marker':True,
    'restore_after_local_disk_loss_bitwise_equal':True,'duplicate_sync_skips_immutable_chunks':True,
    'corrupt_remote_payload_rejected_before_commit':True,'backend':'in-memory S3 test double'})
print(f'{pairs} checkpoint pairs: ordering, restored hashes, deduplication and corruption checks passed.')
