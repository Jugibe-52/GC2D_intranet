"""Create the immutable AWS input bundle and render its hash-pinned user data."""
from pathlib import Path
import sys
import tarfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from study_io import digest, atomic_json, utc_now
import json
cfg=json.loads((ROOT/'experiment.json').read_text())
run=cfg['run_id']
bucket=cfg['bucket']
files=['calculo.ipynb','visualizacion.ipynb','run.py','study_io.py','parallel_calculation.py',
       'checkpoint_calculation.py','newton_diagnostics.py','experiment.py','experiment.json',
       'poincare_batch.py','poincare_selector.py','_poincare_selector.html',
       'requirements.txt','README.md','validate_restart.py','assets',
       'aws/cloud_job.py','aws/validate_checkpoint_store.py',
       'aws/validation_restart.json','aws/validation_cloud_store.json']
archive=ROOT/'aws'/f'{ROOT.name}_aws.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for name in files: tar.add(ROOT/name,arcname=f'{ROOT.name}/{name}')
sha=digest(archive)
atomic_json(ROOT/'aws/input_manifest.json',{'created_utc':utc_now(),'run_id':run,'archive':archive.name,
    'sha256':sha,'bytes':archive.stat().st_size,'bucket':bucket,'key':'inputs/'+archive.name,
    'instance_type':'c8a.4xlarge','market':'spot','region':'eu-central-1','processes':16,
    'configuration':cfg,
    'source_sha256':{name:digest(ROOT/name) for name in files if (ROOT/name).is_file()}})
script=r'''#!/usr/bin/env bash
set -euo pipefail
exec > >(tee -a /var/log/poincare-bootstrap.log) 2>&1
trap '/usr/sbin/shutdown -h now' ERR
# One global configured deadline. OnCalendar/Persistent preserves it across reboots.
TASK_DEADLINE=$(date -u -d '+__HOURS__ hours' '+%Y-%m-%d %H:%M:%S UTC')
cat > /etc/systemd/system/poincare-stop.service <<'EOF'
[Unit]
Description=Stop Poincare compute at the global deadline
[Service]
Type=oneshot
ExecStart=/usr/sbin/shutdown -h now
EOF
cat > /etc/systemd/system/poincare-stop.timer <<EOF
[Unit]
Description=Global configured Poincare deadline
[Timer]
OnCalendar=$TASK_DEADLINE
Persistent=true
AccuracySec=1s
Unit=poincare-stop.service
[Install]
WantedBy=timers.target
EOF
systemctl daemon-reload
systemctl enable --now poincare-stop.timer
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y python3-venv python3-pip
python3 -c 'import sys; assert sys.version_info[:2] == (3,12), sys.version'
install -d -o ubuntu -g ubuntu /home/ubuntu/poincare
sudo -u ubuntu python3 -m venv /home/ubuntu/poincare/.transfer
sudo -u ubuntu /home/ubuntu/poincare/.transfer/bin/python -m pip install boto3==1.43.98
sudo -u ubuntu /home/ubuntu/poincare/.transfer/bin/python - <<'PYDOWNLOAD'
from pathlib import Path,PurePosixPath
import boto3,hashlib,tarfile
archive=Path('/home/ubuntu/poincare/__ARCHIVE__')
boto3.client('s3',region_name='eu-central-1').download_file('__BUCKET__','inputs/__ARCHIVE__',str(archive))
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='__SHA__'
with tarfile.open(archive) as tar:
    for item in tar.getmembers():
        p=PurePosixPath(item.name)
        assert not p.is_absolute() and '..' not in p.parts
        assert p.parts[0]=='__FOLDER__' and (item.isfile() or item.isdir())
    tar.extractall(archive.parent,filter='data')
print('Input package SHA256 verified and extracted.',flush=True)
PYDOWNLOAD
sudo -u ubuntu python3 -m venv /home/ubuntu/poincare/__FOLDER__/.venv
sudo -u ubuntu /home/ubuntu/poincare/__FOLDER__/.venv/bin/python -m pip install -r /home/ubuntu/poincare/__FOLDER__/requirements.txt
cat > /usr/local/sbin/poincare-job <<'EOF'
#!/usr/bin/env bash
set -uo pipefail
cd /home/ubuntu/poincare/__FOLDER__
sudo -u ubuntu /home/ubuntu/poincare/.transfer/bin/python -u aws/cloud_job.py --run-id __RUN__ --bucket __BUCKET__
TASK_RESULT=$?
# User-initiated shutdown keeps a persistent Spot request stopped (no replacement).
/usr/sbin/shutdown -h now
exit "$TASK_RESULT"
EOF
chmod 755 /usr/local/sbin/poincare-job
cat > /etc/systemd/system/__SERVICE__ <<'EOF'
[Unit]
Description=__METHOD__ Poincare study with restart checkpoints
Wants=network-online.target
After=network-online.target
[Service]
Type=simple
ExecStart=/usr/local/sbin/poincare-job
Restart=no
TimeoutStopSec=90
Environment=OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now __SERVICE__
'''
for old,new in {'__ARCHIVE__':archive.name,'__BUCKET__':bucket,'__SHA__':sha,
                '__FOLDER__':ROOT.name,'__RUN__':run,'__HOURS__':str(cfg['deadline_hours']), '__SERVICE__':cfg['service'],'__METHOD__':cfg['method']}.items():script=script.replace(old,new)
(ROOT/'aws/launch_user_data.sh').write_text(script)
print(archive)
print(sha)
print(f'{archive.stat().st_size} bytes')
