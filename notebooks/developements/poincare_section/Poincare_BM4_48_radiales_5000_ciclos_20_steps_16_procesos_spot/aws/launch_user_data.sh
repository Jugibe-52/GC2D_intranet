#!/usr/bin/env bash
set -euo pipefail
exec > >(tee -a /var/log/poincare-bootstrap.log) 2>&1
# One global six-hour deadline. OnCalendar/Persistent preserves it across reboots.
TASK_DEADLINE=$(date -u -d '+6 hours' '+%Y-%m-%d %H:%M:%S UTC')
cat > /etc/systemd/system/poincare-stop.service <<'EOF'
[Unit]
Description=Stop Poincare compute at the global deadline
[Service]
Type=oneshot
ExecStart=/usr/sbin/shutdown -h now
EOF
cat > /etc/systemd/system/poincare-stop.timer <<EOF
[Unit]
Description=Global six-hour Poincare deadline
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
archive=Path('/home/ubuntu/poincare/Poincare_BM4_48_radiales_5000_ciclos_20_steps_16_procesos_spot_aws.tar.gz')
boto3.client('s3',region_name='eu-central-1').download_file('gc2d-frankfurt-20260904-j7m3q9','inputs/Poincare_BM4_48_radiales_5000_ciclos_20_steps_16_procesos_spot_aws.tar.gz',str(archive))
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='316fce464861285f32f11283c0208959acd55bf30214f521069d5de2729a74eb'
with tarfile.open(archive) as tar:
    for item in tar.getmembers():
        p=PurePosixPath(item.name)
        assert not p.is_absolute() and '..' not in p.parts
        assert p.parts[0]=='Poincare_BM4_48_radiales_5000_ciclos_20_steps_16_procesos_spot' and (item.isfile() or item.isdir())
    tar.extractall(archive.parent,filter='data')
print('Input package SHA256 verified and extracted.',flush=True)
PYDOWNLOAD
sudo -u ubuntu python3 -m venv /home/ubuntu/poincare/Poincare_BM4_48_radiales_5000_ciclos_20_steps_16_procesos_spot/.venv
sudo -u ubuntu /home/ubuntu/poincare/Poincare_BM4_48_radiales_5000_ciclos_20_steps_16_procesos_spot/.venv/bin/python -m pip install -r /home/ubuntu/poincare/Poincare_BM4_48_radiales_5000_ciclos_20_steps_16_procesos_spot/requirements.txt
cat > /usr/local/sbin/poincare-job <<'EOF'
#!/usr/bin/env bash
set -uo pipefail
cd /home/ubuntu/poincare/Poincare_BM4_48_radiales_5000_ciclos_20_steps_16_procesos_spot
sudo -u ubuntu /home/ubuntu/poincare/.transfer/bin/python -u aws/cloud_job.py --run-id aws_48p_5000c_20s_16proc_spot_20260920 --bucket gc2d-frankfurt-20260904-j7m3q9
TASK_RESULT=$?
# User-initiated shutdown keeps a persistent Spot request stopped (no replacement).
/usr/sbin/shutdown -h now
exit "$TASK_RESULT"
EOF
chmod 755 /usr/local/sbin/poincare-job
cat > /etc/systemd/system/poincare-48p-5000cycles.service <<'EOF'
[Unit]
Description=BM4 48 particles 5000 cycles with restart checkpoints
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
systemctl enable --now poincare-48p-5000cycles.service
