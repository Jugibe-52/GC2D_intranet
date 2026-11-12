"""Package verified old and new Poincare returns as a lazy, grouped static site."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np

from diagnostics.io.persistence import load_solution
from studies.poincare_parallel import METHOD_SLUGS, load_returns
from visualization.poincare_transport import attach_horizontal_transport
from run_poincare_parallel import BASE, OUTPUT, ROOT
from run_horizontal_transport import write_json


def decoded(panel: dict) -> np.ndarray:
    return np.frombuffer(base64.b64decode(panel['data']), dtype='<f4').reshape(panel['shape'])


def extended_panel(panel: dict, new: np.ndarray, colors: list[str]) -> None:
    """Append IDs while checking that every original float32 sample survives."""
    old = decoded(panel)
    if panel['ids'] != list(range(1,49)) or old.shape[1:] != (48,2):
        raise ValueError('Unexpected original particle ordering.')
    if (new.shape != (old.shape[0],36,2) or not np.isfinite(new).all()
            or np.any((new<0)|(new>1))):
        raise ValueError('Incomplete or nonfinite new trajectories.')
    joined = np.concatenate((old,new.astype('<f4')),axis=1).astype('<f4')
    if not np.array_equal(joined[:,:48],old):
        raise ValueError('An original sample changed.')
    panel.update(data=base64.b64encode(joined.tobytes()).decode(),shape=list(joined.shape),
                 ids=list(range(1,85)),colors=colors)
    if not np.array_equal(decoded(panel),joined):
        raise ValueError('Viewer encoding changed trajectory values.')


def main() -> None:
    """Export only fully audited results; never submit work from publication."""
    audit=json.loads((OUTPUT/'audit.json').read_text())
    if audit['status']!='passed' or len(audit['jobs'])!=44:
        raise ValueError('Publication requires a successful 44-job audit.')
    baseline=json.loads((OUTPUT/'baseline.json').read_text())
    source=(BASE/'four_methods_poincare.html').read_bytes()
    if sha256(source).hexdigest()!=baseline['original_html_sha256']:
        raise ValueError('The original viewer fingerprint changed.')
    old_results=(BASE/'horizontal_transport_results.json').read_bytes()
    if sha256(old_results).hexdigest()!=baseline['original_results_sha256']:
        raise ValueError('The original metric fingerprint changed.')
    match=re.search(r'^const cfg=(.*);$',source.decode(),re.MULTILINE)
    if match is None:raise ValueError('Missing original viewer configuration.')
    config=json.loads(match.group(1));del source,match
    seeds=json.loads((OUTPUT/'seeds.json').read_text())
    summary=json.loads(old_results)
    summary['created_utc']=datetime.now(timezone.utc).isoformat()
    summary['provenance']=baseline
    cases={(round(c['rho'],2),c['method']):c for c in summary['cases']}
    colors=config['panels'][0]['colors'][:]
    # The reference keeps its original color; neighbours use ordered light/dark variants.
    for family in seeds['families']:
        parent_color=colors[family['parent']-1].lstrip('#')
        rgb=np.array([int(parent_color[i:i+2],16) for i in (0,2,4)])
        for target,amount in ((255,.25),(255,.50),(0,.15),(0,.35)):
            c=np.rint(rgb*(1-amount)+target*amount).astype(int)
            colors.append('#'+''.join(f'{v:02x}' for v in c))
    initial=np.asarray([p['cell_fraction'] for p in seeds['particles']])
    checked=0
    for k,dataset in enumerate(config['datasets']):
        if dataset['key']!=f'{k/20:.2f}':raise ValueError('Unexpected rho order.')
        for panel in dataset['panels']:
            if panel.get('static'):
                extended_panel(panel,initial[None,:,:],colors)
                panel['title']='Potentiel effectif et vitesse GC'
                continue
            method=panel['title'];record=load_solution(OUTPUT/f'runs/{METHOD_SLUGS[method]}_{k:02d}')
            if record.metadata['seed_manifest']!=seeds:
                raise ValueError('The run uses different initial seeds.')
            diagnostics=record.solution.diagnostics
            xy=np.asarray(diagnostics['cycle_positions_cell_fraction'])
            np.testing.assert_allclose(xy[0],initial,rtol=0,atol=2e-15)
            extended_panel(panel,xy,colors)
            case=cases[(k/20,method)]
            if case['particle_ids']!=list(range(1,49)):raise ValueError('Unexpected metric IDs.')
            case['particle_ids']+=list(range(49,85))
            if record.metadata.get('solver_recovery'):
                case['solver_recovery']=record.metadata['solver_recovery']
                dataset['solverRecovery']=record.metadata['solver_recovery']
            if record.metadata.get('grouping_recovery'):
                case['grouping_recovery']=record.metadata['grouping_recovery']
                dataset['groupingRecovery']=record.metadata['grouping_recovery']
            for public,key in [('q_x','horizontal_transport_rate'),('travel_x','horizontal_travel'),
                               ('width_x','horizontal_width'),('x_min','horizontal_min'),('x_max','horizontal_max')]:
                case[public]+=np.asarray(diagnostics[key]).tolist()
            checked+=1
        dataset['note']='84 particles · 48 original + 36 neighbours · 5,000 cycles · 50 steps/cycle'
        if dataset.get('solverRecovery'):
            dataset['note']+=' · New BM4-I particles: damped Newton, unchanged step and tolerances'
        if dataset.get('groupingRecovery'):
            dataset['note']+=' · New BM4-I particles integrated independently; original step and solver controls'
    config['panels']=config['datasets'][6]['panels']
    config=attach_horizontal_transport(config,summary)
    config['plotLanguage']='fr'
    config['title']='Poincare trajectory families · 84 particles'
    config['description']='Nine reference curves at ρ = 0.30, each with two initial positions inside and two outside. The same 36 new seeds are used for every ρ. Select, isolate or zoom a family below. Initial normal offsets are 2% and 4% of its projected curve width; nearby trajectories need not remain parallel. Drag to zoom; Shift-drag to pan.'
    config.pop('particleGroups',None)
    config['particleFamilies']=[]
    for family in seeds['families']:
        parent=family['parent']
        group='star' if parent<=40 else 'probes'
        metadata,returns=load_returns(BASE/'horizontal_transport/runs','GaussLegendre4',6,group)
        points=returns[:,metadata['particle_id'].index(parent)]
        minimum=points.min(axis=0);maximum=points.max(axis=0)
        span=float(max(maximum-minimum)*1.25)
        center=(minimum+maximum)/2
        view=dict(x=float(np.clip(center[0]-span/2,0,1-span)),y=float(np.clip(center[1]-span/2,0,1-span)),span=span)
        near=next(p['distance_m'] for p in seeds['particles'] if p['parent']==parent)
        note=f'ρref = 0.30 · offsets ±{near*1000:.4f} / ±{near*2000:.4f} mm'
        if parent==2:note+=' · normal to the initial island (7-island chain)'
        config['particleFamilies'].append(dict(parent=parent,ids=family['ids'],label=f'Family #{parent}',note=note,view=view))
    for p in seeds['particles']:
        side='inside' if p['signed_width_fraction']<0 else 'outside'
        config['particleLabels'][str(p['id'])]=f"#{p['id']} · {side} {abs(p['signed_width_fraction'])*100:.0f}% · {p['distance_m']*1000:.4f} mm"
    site=OUTPUT/'site';assets=site/'assets';assets.mkdir(parents=True,exist_ok=True)
    files=[]
    for dataset in config['datasets']:
        payload={key:dataset[key] for key in ('key','panels','horizontalTransport')}
        raw=json.dumps(payload,separators=(',',':'),allow_nan=False).encode()
        digest=sha256(raw).hexdigest()
        name=f"rho-{dataset['key']}-{digest[:12]}.json"
        if len(raw)>=25*1024**2:raise ValueError('A Pages asset exceeds 25 MiB.')
        (assets/name).write_bytes(raw)
        if sha256((assets/name).read_bytes()).hexdigest()!=digest:raise ValueError('Written asset differs.')
        files.append(dict(path='assets/'+name,bytes=len(raw),sha256=digest))
        dataset['asset']='assets/'+name
        del dataset['panels'],dataset['horizontalTransport']
    del config['panels']
    template=(ROOT/'src/visualization/_poincare_comparison.html').read_text()
    template=template.replace('</section>\n<script>', '</section>\n<p>Study data: <a href="seeds.json">Initial positions</a> · <a href="horizontal_transport_results.json">All Qx values</a> · <a href="provenance.json">Provenance</a> · <a href="https://gc2d-poincare-four-methods.pages.dev/">Original viewer</a></p>\n<script>')
    html=template.replace('__CONFIG__',json.dumps(config,separators=(',',':'),allow_nan=False).replace('<','\\u003c'))
    (site/'index.html').write_text(html)
    (site/'404.html').write_text('<!doctype html><meta charset="utf-8"><title>Not found</title><p>Page not found. <a href="/">Return to the simulation</a></p>')
    (site/'_headers').write_text('/assets/*\n  Cache-Control: public, max-age=31536000, immutable\n/\n  Cache-Control: no-cache\n')
    write_json(site/'seeds.json',seeds)
    write_json(site/'horizontal_transport_results.json',summary)
    report=dict(status='passed',created_utc=datetime.now(timezone.utc).isoformat(),datasets=11,
        particles=84,old_particles_unchanged=48,new_particles=36,methods=list(METHOD_SLUGS),
        validated_cases=checked,trajectories=3696,metric_values=3696,returns_per_trajectory=5001,
        reference_rho=.3,baseline=baseline,assets=files,html_sha256=sha256(html.encode()).hexdigest())
    write_json(OUTPUT/'export_audit.json',report)
    write_json(site/'provenance.json',report)
    print(json.dumps(dict(site=str(site),assets=len(files),max_asset_mb=max(f['bytes'] for f in files)/1024**2,
                         html_kb=len(html)/1024,trajectories=3696)),flush=True)


if __name__=='__main__':main()
