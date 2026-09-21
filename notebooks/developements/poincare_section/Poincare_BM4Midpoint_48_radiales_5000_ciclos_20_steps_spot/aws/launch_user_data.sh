#!/usr/bin/env bash
set -euo pipefail
exec > >(tee -a /var/log/poincare-bootstrap.log) 2>&1
trap '/usr/sbin/shutdown -h now' ERR
# One global configured deadline. OnCalendar/Persistent preserves it across reboots.
TASK_DEADLINE=$(date -u -d '+3 hours' '+%Y-%m-%d %H:%M:%S UTC')
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
archive=Path('/home/ubuntu/poincare/Poincare_BM4Midpoint_48_radiales_5000_ciclos_20_steps_spot_aws.tar.gz')
boto3.client('s3',region_name='eu-central-1').download_file('gc2d-frankfurt-20260904-j7m3q9','inputs/Poincare_BM4Midpoint_48_radiales_5000_ciclos_20_steps_spot_aws.tar.gz',str(archive))
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='66536e42f1714847d93cb30b3d6beb58720a5fb1159b5883428e9720384b29ad'
with tarfile.open(archive) as tar:
    for item in tar.getmembers():
        p=PurePosixPath(item.name)
        assert not p.is_absolute() and '..' not in p.parts
        assert p.parts[0]=='Poincare_BM4Midpoint_48_radiales_5000_ciclos_20_steps_spot' and (item.isfile() or item.isdir())
    tar.extractall(archive.parent,filter='data')
print('Input package SHA256 verified and extracted.',flush=True)
PYDOWNLOAD
sudo -u ubuntu python3 -m venv /home/ubuntu/poincare/Poincare_BM4Midpoint_48_radiales_5000_ciclos_20_steps_spot/.venv
sudo -u ubuntu /home/ubuntu/poincare/Poincare_BM4Midpoint_48_radiales_5000_ciclos_20_steps_spot/.venv/bin/python -m pip install -r /home/ubuntu/poincare/Poincare_BM4Midpoint_48_radiales_5000_ciclos_20_steps_spot/requirements.txt
cat > /usr/local/sbin/poincare-job <<'EOF'
#!/usr/bin/env bash
set -uo pipefail
cd /home/ubuntu/poincare/Poincare_BM4Midpoint_48_radiales_5000_ciclos_20_steps_spot
sudo -u ubuntu /home/ubuntu/poincare/.transfer/bin/python -u aws/cloud_job.py --run-id aws_bm4midpoint_48p_5000c_20s_20260920 --bucket gc2d-frankfurt-20260904-j7m3q9
TASK_RESULT=$?
# User-initiated shutdown keeps a persistent Spot request stopped (no replacement).
/usr/sbin/shutdown -h now
exit "$TASK_RESULT"
EOF
chmod 755 /usr/local/sbin/poincare-job
cat > /etc/systemd/system/poincare-midpoint.service <<'EOF'
[Unit]
Description=BM4Midpoint Poincare study with restart checkpoints
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
systemctl enable --now poincare-midpoint.service
