"""Geometry and independent-solver audit for radial reference studies."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from diagnostics import StoredReferenceTrajectory
from potential import Potential


def plot_reference_discrepancy_until(
    reference: StoredReferenceTrajectory,
    end_time: float = 50.0,
) -> tuple[Figure, plt.Axes]:
    """Plot the saved audit over a time prefix, scaling only to visible data."""
    if not np.isfinite(end_time) or end_time <= reference.times[0]:
        raise ValueError("End time must be finite and greater than the initial time.")
    selected = reference.times <= end_time
    figure, axis = plt.subplots(figsize=(9, 4), constrained_layout=True)
    for i, distances in enumerate(reference.audit_distances):
        axis.plot(reference.times[selected], distances[selected], label=f"Particle {i + 1}")
    axis.set(xlim=(reference.times[0], min(end_time, reference.times[-1])),
             xlabel="Normalized time", ylabel="Periodic distance",
             title="DOP853–Radau discrepancy")
    axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axis.legend()
    axis.grid(alpha=0.2)
    return figure, axis


def plot_radial_reference_audit(
    reference: StoredReferenceTrajectory,
    potential: Potential | None = None,
    center: np.ndarray | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot saved geometry and audit; default to metadata without loading a field."""
    if center is None:
        center = np.asarray(reference.metadata["initial_conditions"]["center"])
    count = reference.initial_state.size // 2
    x, y = reference.initial_state.reshape(2, count)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].plot([center[0], x[-1]], [center[1], y[-1]], "k--", alpha=0.4)
    axes[0].scatter(*center, marker="+", color="black", label="Cell center")
    for i in range(count):
        axes[0].scatter(x[i], y[i], label=f"Particle {i + 1}")
        # A linear axis preserves exact zeros instead of replacing them by a floor.
        axes[1].plot(reference.times, reference.audit_distances[i], label=f"Particle {i + 1}")
    if potential is None:
        grid = reference.metadata["potential_grid"]
        xmin, ymin, period = grid["xmin"], grid["ymin"], grid["period"]
    else:
        grid = potential.grid
        xmin, ymin, period = grid.xmin, grid.ymin, grid.period
    axes[0].set(xlim=(xmin, xmin + period),
                ylim=(ymin, ymin + period),
                xlabel="x", ylabel="y", title="Initial positions on one radius")
    axes[0].set_aspect("equal")
    axes[1].set(xlabel="Normalized time", ylabel="Periodic distance",
                title="DOP853–Radau discrepancy")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.2)
    return figure, axes


def animate_radial_reference(
    reference: StoredReferenceTrajectory,
    frames: int = 201,
    fps: float = 10.0,
):
    """Return standalone HTML playback with independent particle/solver filters.

    Display uniformly sampled saved states in unwrapped normalized coordinates;
    downsampling affects only the display, never the stored reference.
    """
    import json
    from uuid import uuid4
    from IPython.display import HTML

    if not isinstance(frames, int) or frames < 2:
        raise ValueError("Frames must be an integer of at least two.")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("FPS must be finite and positive.")
    indices = np.unique(np.linspace(0, reference.times.size - 1,
                                    min(frames, reference.times.size), dtype=int))
    count = reference.initial_state.size // 2
    # Packed histories contain all x coordinates followed by all y coordinates.
    histories = np.stack([reference.states, reference.audit_states])
    histories = histories.reshape(2, 2, count, -1)[..., indices]
    payload = json.dumps(dict(times=reference.times[indices].tolist(),
                              states=histories.tolist(), count=count, fps=float(fps)))
    template = r'''
<div id="__ID__" style="max-width:850px;font-family:sans-serif">
  <div><strong>Particles:</strong> <span class="particles"></span></div>
  <div><strong>Models / solvers:</strong>
    <label><input type="checkbox" class="model" value="0" checked>DOP853 (solid)</label>
    <label><input type="checkbox" class="model" value="1" checked>Radau (dashed)</label>
  </div>
  <button class="play">Play</button> <button class="reset">Reset</button>
  <input class="time" type="range" min="0" value="0" style="width:50%" aria-label="Time frame">
  <output></output>
  <canvas width="820" height="570" style="width:100%;background:white" aria-label="Animated radial trajectories"></canvas>
  <p>Unwrapped normalized coordinates. Color identifies the particle; line style identifies the solver.
  Trails show sampled positions up to the selected time. Playback uses saved data only.</p>
</div>
<script>
(() => {
 const root = document.getElementById('__ID__'), data = __DATA__;
 const canvas = root.querySelector('canvas'), ctx = canvas.getContext('2d');
 const slider = root.querySelector('.time'), play = root.querySelector('.play');
 const colors = Array.from({length:data.count}, (_,p) => `hsl(${p*137.508%360},65%,40%)`);
 let timer = null;
 slider.max = data.times.length - 1;
 for(let p=0;p<data.count;p++) {
   const label = document.createElement('label');
   label.style.color = colors[p];
   const input = document.createElement('input');
   input.type='checkbox'; input.className='particle'; input.value=p; input.checked=true;
   label.append(input, document.createTextNode(`Particle ${p+1} `));
   root.querySelector('.particles').append(label);
 }
 // Fixed equal-scale bounds avoid apparent motion caused by autoscaling.
 const xs=data.states.flatMap(s=>s[0].flat()), ys=data.states.flatMap(s=>s[1].flat());
 const extent=a=>a.reduce((r,v)=>[Math.min(r[0],v),Math.max(r[1],v)],[Infinity,-Infinity]);
 const [xmin,xmax]=extent(xs), [ymin,ymax]=extent(ys);
 const scale=Math.min(690/Math.max(xmax-xmin,1e-12),450/Math.max(ymax-ymin,1e-12))*.92;
 const X=x=>440+(x-(xmin+xmax)/2)*scale;
 const Y=y=>270-(y-(ymin+ymax)/2)*scale;
 function draw() {
   const frame=Number(slider.value);
   ctx.clearRect(0,0,820,570); ctx.fillStyle='#333'; ctx.font='14px sans-serif';
   ctx.fillText('HDF5 guiding-center trajectories',60,25);
   ctx.strokeStyle='#aaa'; ctx.setLineDash([]); ctx.strokeRect(65,40,730,465);
   ctx.fillText('x (normalized)',360,545);
   ctx.save(); ctx.translate(20,320); ctx.rotate(-Math.PI/2); ctx.fillText('y (normalized)',0,0); ctx.restore();
   for(let k=0;k<=4;k++) {
     const px=65+k*730/4, py=40+k*465/4;
     ctx.fillText(((px-440)/scale+(xmin+xmax)/2).toPrecision(4),px-20,525);
     ctx.fillText(((270-py)/scale+(ymin+ymax)/2).toPrecision(4),24,py+4);
   }
   const particles=[...root.querySelectorAll('.particle:checked')].map(e=>Number(e.value));
   const models=[...root.querySelectorAll('.model:checked')].map(e=>Number(e.value));
   for(const m of models) for(const p of particles) {
     const x=data.states[m][0][p], y=data.states[m][1][p];
     ctx.strokeStyle=colors[p]; ctx.fillStyle=colors[p]; ctx.lineWidth=1.8;
     ctx.setLineDash(m===0?[]:[7,5]); ctx.beginPath();
     for(let i=0;i<=frame;i++) {if(i===0)ctx.moveTo(X(x[i]),Y(y[i]));else ctx.lineTo(X(x[i]),Y(y[i]));}
     ctx.stroke(); ctx.setLineDash([]);
     ctx.beginPath(); ctx.arc(X(x[frame]),Y(y[frame]),m===0?5:8,0,2*Math.PI);
     if(m===0)ctx.fill(); else ctx.stroke();
   }
   if(!particles.length || !models.length) ctx.fillText('Select at least one particle and one model.',220,270);
   root.querySelector('output').textContent=`t = ${data.times[frame].toFixed(4)}`;
 }
 function stop(){clearInterval(timer);timer=null;play.textContent='Play';}
 play.onclick=()=>{
   if(timer!==null){stop();return;}
   if(Number(slider.value)===data.times.length-1)slider.value=0;
   play.textContent='Pause'; draw();
   timer=setInterval(()=>{
     if(!root.isConnected){stop();return;}
     slider.value=Number(slider.value)+1; draw();
     if(Number(slider.value)>=data.times.length-1)stop();
   },1000/data.fps);
 };
 root.querySelector('.reset').onclick=()=>{stop();slider.value=0;draw();};
 slider.oninput=draw;
 root.querySelectorAll('input[type=checkbox]').forEach(e=>e.onchange=draw);
 draw();
})();
</script>
'''
    return HTML(template.replace('__ID__', 'radial_' + uuid4().hex)
                .replace('__DATA__', payload))
