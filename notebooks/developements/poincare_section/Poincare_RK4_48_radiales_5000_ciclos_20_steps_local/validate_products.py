"""Validate local RK4 products without integrating trajectories."""
import argparse,json
import numpy as np
from study_io import ROOT,load_calculation,atomic_json,utc_now,digest

def validate(run):
    folder,meta,a,positions,initial=load_calculation(run)
    cfg=json.loads((ROOT/'experiment.json').read_text());cycles=2 if meta['validation_run'] else cfg['cycles'];n=len(cfg['particle_ids']);steps=cycles*cfg['steps_per_cycle']
    for key,value in dict(method='RK4',particle_count=n,process_count=4,cycles=cycles,steps_per_cycle=20,complete_steps=steps,newton_history_recorded=False,reference_computed=False).items():assert meta[key]==value,key
    eq=np.testing.assert_array_equal;original=json.loads((ROOT/'assets/original_particles.json').read_text())['particles']
    eq(a['particle_ids'],cfg['particle_ids']);eq(a['initial_positions'],[[p['x'],p['y']] for p in original]);eq(a['colors_rgba'],[p['rgba'] for p in original]);eq(a['colors_hex'],[p['color'] for p in original])
    assert a['states'].shape==(96,steps+1) and np.isfinite(a['states']).all()
    eq(a['times'],np.arange(steps+1)*.05);eq(a['cycle_times'],np.arange(cycles+1))
    eq(a['cycle_positions'][:,:,0],a['states'][:48,::20].T);eq(a['cycle_positions'][:,:,1],a['states'][48:,::20].T)
    assert len(positions)==48*cycles and len(initial)==48
    eq(positions.particle,np.tile(a['particle_ids'],cycles));eq(positions.color,np.tile(a['colors_hex'],cycles));eq(positions[['x_unwrapped','y_unwrapped']],a['cycle_positions'][1:].reshape(-1,2))
    assert not any('newton' in k or 'multiplier' in k or 'copy_separation' in k for k in a)
    assert len(meta['workers'])==4 and len({w['pid'] for w in meta['workers']})==4
    for i,w in enumerate(meta['workers']):
        assert w['particle_ids']==cfg['particle_ids'][i::4]
        assert w['resumed_steps']+w['new_steps']==steps
    assert meta['rk4_checkpoint_driver_sha256']==digest(ROOT/'rk4_checkpoints.py')
    for category,notebook in [('resultados','calculo_executed.ipynb'),('figuras','visualizacion_executed.ipynb')]:
        directory=ROOT/category/run
        assert json.loads((directory/'execution_status.json').read_text())['status']=='success'
        for cell in json.loads((directory/notebook).read_text())['cells']:
            if cell['cell_type']=='code' and ''.join(cell['source']).strip():assert cell['execution_count'] is not None
            assert not any(o.get('output_type')=='error' for o in cell.get('outputs',[]))
    report=dict(verified_utc=utc_now(),run_id=run,method='RK4',particles=48,cycles=cycles,steps_per_cycle=20,processes=4,return_position_rows=len(positions),numeric_manifest_verified=True,original_ids_positions_colors_preserved=True,executed_notebooks_verified=True)
    atomic_json(ROOT/'validation'/f'{run}_products.json',report)
    print(json.dumps(report,indent=2));return report
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);validate(parser.parse_args().run_id)
