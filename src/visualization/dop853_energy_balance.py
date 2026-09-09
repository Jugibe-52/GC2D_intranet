"""Global/particle energy figures and a standalone selectable trajectory player."""
import json
import uuid
import numpy as np
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
from IPython.display import HTML
from studies.dop853_energy_balance import METHODS

COLORS = dict(zip(METHODS, ('black', 'tab:green', 'tab:orange', 'tab:purple')))


def summary_rows(arrays, particle=None):
    """Report time-weighted RMS, peak and final errors without cancellation."""
    t = arrays['times']
    rows = []
    for name in METHODS:
        row = {'method': name}
        for field in ('balance', 'distance'):
            values = arrays[name+'.'+field]
            if particle is not None:
                values = values[particle:particle+1]
            row[field+'_rms'] = float(np.sqrt(trapezoid(np.mean(values**2, axis=0), t)/(t[-1]-t[0])))
            row[field+'_max'] = float(np.max(np.abs(values)))
            row[field+'_final_rms'] = float(np.sqrt(np.mean(values[:, -1]**2)))
        rows.append(row)
    return rows


def plot_analysis(arrays, particle=None):
    """Show physical energy, signed/absolute balance, and periodic distance."""
    t = arrays['times']
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    scope = 'Global: particle RMS' if particle is None else f'Particle {particle+1}'
    for name in METHODS:
        h = arrays[name+'.H']
        e = arrays[name+'.balance']
        d = arrays[name+'.distance']
        aggregate = lambda a: np.sqrt(np.mean(a*a, axis=0)) if particle is None else a[particle]
        axes[0, 0].plot(t, aggregate(h), color=COLORS[name], label=name)
        axes[0, 1].plot(t, aggregate(e), color=COLORS[name], label=name)
        axes[1, 0].plot(t, np.maximum.accumulate(np.max(np.abs(e), axis=0) if particle is None else np.abs(e[particle])), color=COLORS[name])
        if name != 'DOP853':
            axes[1, 1].plot(t, aggregate(d), color=COLORS[name], label=name)
    titles = ('Physical Hamiltonian H', 'Energy-balance defect K(t) - K(0)',
              'Running maximum absolute balance defect', 'Minimum-image distance to DOP853')
    for ax, title in zip(axes.flat, titles):
        ax.set(title=title, xlabel='Normalized cycles')
        ax.grid(True, alpha=.3)
    axes[0, 1].set_yscale('symlog', linthresh=1e-12)
    axes[1, 0].set_yscale('symlog', linthresh=1e-12)
    axes[1, 1].set_yscale('symlog', linthresh=1e-12)
    axes[0, 0].legend()
    axes[1, 1].legend()
    fig.suptitle(scope)
    return fig


def plot_mu(arrays):
    """Display endpoint and within-save-block peak norms at ten saves/cycle."""
    t = arrays['times'][1:]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
    for index, ax in enumerate(axes.flat):
        for kind, label in [('endpoint', 'Endpoint infinity norm'), ('max', 'Peak within output block')]:
            values = arrays['BM4Implicit.mu_'+kind][:, 1:]
            y = values.max(axis=0) if index == 0 else values[index-1]
            ax.plot(t, y, label=label)
        ax.set(title='Global infinity norm' if index == 0 else f'Particle {index}', xlabel='Normalized cycles', ylabel='BM4 multiplier norm')
        ax.set_yscale('symlog', linthresh=1e-16)
        ax.grid(True, alpha=.3)
        ax.legend()
    return fig


def mu_summary_rows(arrays):
    """Summarize every accepted step using equal-size block sufficient statistics."""
    rows = []
    for p in range(3):
        rows.append(dict(particle=f'Particle {p+1}',
                         mean=float(np.mean(arrays['BM4Implicit.mu_mean'][p, 1:])),
                         rms=float(np.sqrt(np.mean(arrays['BM4Implicit.mu_rms'][p, 1:]**2))),
                         maximum=float(np.max(arrays['BM4Implicit.mu_max'][p, 1:])),
                         final=float(arrays['BM4Implicit.mu_endpoint'][p, -1])))
    endpoint, peak, rms, mean = arrays['BM4Implicit.mu_global_statistics'][:, 1:]
    rows.insert(0, dict(particle='Global', mean=float(mean.mean()),
                        rms=float(np.sqrt(np.mean(rms**2))), maximum=float(peak.max()),
                        final=float(endpoint[-1])))
    return rows


def plot_paths(arrays, particle=None):
    """Plot all saved paths and break lines at periodic boundary crossings."""
    xmin, ymin, period = arrays['domain']
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    for name in METHODS:
        states = arrays[name+'.states']
        for p in range(3) if particle is None else [particle]:
            x = (states[p]-xmin) % period + xmin
            y = (states[3+p]-ymin) % period + ymin
            jumps = np.r_[False, (np.abs(np.diff(x)) > period/2) | (np.abs(np.diff(y)) > period/2)]
            x = np.where(jumps, np.nan, x)
            y = np.where(jumps, np.nan, y)
            ax.plot(x, y, color=COLORS[name], alpha=.65, label=name if p == (0 if particle is None else particle) else None)
    initial = arrays['initial_state']
    ax.scatter(initial[:3], initial[3:], color='gold', edgecolor='black', zorder=5)
    for p in range(3):
        ax.annotate(f'P{p+1}', (initial[p], initial[p+3]))
    ax.set(xlim=(xmin, xmin+period), ylim=(ymin, ymin+period), xlabel='x', ylabel='y', title='Saved trajectories including DOP853', aspect='equal')
    ax.legend()
    return fig


def trajectory_player(arrays):
    """Self-contained HTML animation with method toggles, seek, trail and speed."""
    ident = 'player_' + uuid.uuid4().hex
    payload = json.dumps(dict(times=arrays['times'].tolist(), domain=arrays['domain'].tolist(),
                              states={name: arrays[name+'.states'].tolist() for name in METHODS}))
    html = '''<div id="ID" style="max-width:820px;font-family:sans-serif">
<div class="checks"></div><button class="play">Play</button>
<label>Frame <input class="seek" type="range" min="0" value="0" style="width:40%"></label>
<label>FPS <input class="fps" type="number" min="1" max="60" value="10" style="width:50px"></label>
<label>Trail samples <input class="trail" type="number" min="0" value="100" style="width:65px"></label>
<span class="time"></span><canvas width="760" height="650" style="width:100%;background:#fafafa"></canvas>
<p>Color identifies method; circle, square and triangle identify particles 1, 2 and 3. All saved frames are available; periodic crossings break trails.</p>
<script>(()=>{const root=document.getElementById('ID'), data=DATA;
const names=Object.keys(data.states), colors=['#111111','#2ca02c','#ff7f0e','#9467bd'];
const enabled=new Set(names), checks=root.querySelector('.checks');
names.forEach((name,i)=>{let label=document.createElement('label'), box=document.createElement('input');box.type='checkbox';box.checked=true;box.onchange=()=>{box.checked?enabled.add(name):enabled.delete(name);draw();};label.append(box,document.createTextNode(name+'  '));label.style.color=colors[i];checks.append(label);});
const canvas=root.querySelector('canvas'),ctx=canvas.getContext('2d'),seek=root.querySelector('.seek');seek.max=data.times.length-1;
let frame=0,timer=null;const [xmin,ymin,L]=data.domain, size=560,left=85,top=35;
const wrap=(v,min)=>((v-min)%L+L)%L;
function pos(s,p,k){return [wrap(s[p][k],xmin),wrap(s[p+3][k],ymin)];}
function pixel(v){return [left+v[0]/L*size,top+size-v[1]/L*size];}
function draw(){ctx.clearRect(0,0,760,650);ctx.strokeStyle='#888';ctx.strokeRect(left,top,size,size);ctx.fillStyle='#333';ctx.font='14px sans-serif';ctx.fillText('x',left+size/2,top+size+35);ctx.fillText('y',left-35,top+size/2);
for(let tick=0;tick<=4;tick++){const f=tick/4;ctx.fillText((xmin+L*f).toFixed(2),left+size*f-15,top+size+18);ctx.fillText((ymin+L*f).toFixed(2),left-55,top+size*(1-f)+4);}
names.forEach((name,i)=>{if(!enabled.has(name))return;const s=data.states[name];ctx.strokeStyle=colors[i];ctx.fillStyle=colors[i];ctx.lineWidth=name==='DOP853'?2.3:1.4;
for(let p=0;p<3;p++){ctx.beginPath();let prev=null;for(let k=Math.max(0,frame-Number(root.querySelector('.trail').value));k<=frame;k++){let v=pos(s,p,k),q=pixel(v);if(!prev||Math.abs(v[0]-prev[0])>L/2||Math.abs(v[1]-prev[1])>L/2)ctx.moveTo(...q);else ctx.lineTo(...q);prev=v;}ctx.stroke();let [x,y]=pixel(pos(s,p,frame));ctx.beginPath();if(p===0)ctx.arc(x,y,5,0,2*Math.PI);else if(p===1)ctx.rect(x-5,y-5,10,10);else{ctx.moveTo(x,y-6);ctx.lineTo(x-6,y+5);ctx.lineTo(x+6,y+5);ctx.closePath();}ctx.fill();ctx.fillText('P'+(p+1),x+7,y-7);}});
seek.value=frame;root.querySelector('.time').textContent='t = '+data.times[frame].toFixed(2);}
function stop(){clearInterval(timer);timer=null;root.querySelector('.play').textContent='Play';}
root.querySelector('.play').onclick=()=>{if(timer){stop();return;}root.querySelector('.play').textContent='Pause';timer=setInterval(()=>{frame=(frame+1)%data.times.length;draw();},1000/Math.max(1,Math.min(60,Number(root.querySelector('.fps').value))));};
seek.oninput=()=>{frame=Number(seek.value);draw();};root.querySelector('.trail').oninput=draw;root.querySelector('.fps').onchange=stop;draw();})();</script></div>'''
    return HTML(html.replace('ID', ident).replace('DATA', payload))
