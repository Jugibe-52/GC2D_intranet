"""Run and verify the authorized local study, with an exclusive process lock."""
import fcntl,json,subprocess,sys,os,traceback
from study_io import ROOT,atomic_json,utc_now
from validate_products import validate

def main():
    cfg=json.loads((ROOT/'experiment.json').read_text());run=cfg['run_id']
    with (ROOT/'.local_job.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status=dict(run_id=run,status='running',pid=os.getpid(),started_utc=utc_now(),processes=4)
        atomic_json(ROOT/'local_status.json',status)
        try:
            for stage in ('calculate','visualize'):
                status['stage']=stage;atomic_json(ROOT/'local_status.json',status)
                command=[sys.executable,'-u',str(ROOT/'run.py'),stage,'--run-id',run,'--timeout','7200']
                if stage=='calculate':command.append('--resume')
                subprocess.run(command,cwd=ROOT,check=True)
            validate(run);status.update(status='success',stage='verified',finished_utc=utc_now())
        except BaseException as error:
            status.update(status='failed',error=str(error),finished_utc=utc_now());traceback.print_exc();raise
        finally:atomic_json(ROOT/'local_status.json',status)
if __name__=='__main__':main()
