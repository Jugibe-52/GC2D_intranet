# guiding_center_intranet

The canonical Python modules live under [`src`](src). The root-level
scripts are entry points: [`run_fourier.py`](run_fourier.py) for parameter-dictionary runs and
[`run_potential.py`](run_potential.py) for the standalone showcase.

For imports outside those scripts, install the package in editable mode:
```sh
python3 -m pip install -e .
```
To install the editable package plus notebook/demo tools, use:
```sh
python3 -m pip install -r requirements.txt
```

Logging is enabled by default for the root scripts. Use `SIM_LOG_LEVEL` to change
verbosity and `SIM_LOG_FILE` to also write logs to a file:
```sh
SIM_LOG_LEVEL=DEBUG SIM_LOG_FILE=logs/simulation.log python3 run_fourier.py
```

Simulation parameters are read from JSON configuration files in [`conf`](conf).
The configuration tree is split first by execution surface, then by execution
family, then by group, then by version file:

```text
conf/
  notebook/
    fourier/test/v_1.json
    potential/test/v_1.json
  terminal/
    fourier/test/v_1.json
    fourier/assay/v_1.json
    potential/test/v_1.json
    potential/assay/v_1.json
```

The main batch runner uses `conf/terminal/fourier/test/v_1.json` by default:
```sh
python3 run_fourier.py
```
Select another profile from the same configuration file with:
```sh
python3 run_fourier.py --version notebook_demo
```
Select an experiment configuration folder with:
```sh
python3 run_fourier.py --config-group assay --config-version v_1
```
or pass an explicit config file:
```sh
python3 run_fourier.py --config conf/terminal/fourier/assay/v_1.json --version symplectic_grid
```
For background runs:
```sh
nohup python3 -u run_fourier.py &>fourier.out < /dev/null &
```

The standalone HDF5/mock-potential showcase reads `conf/terminal/potential/test/v_1.json`
by default:
```sh
python3 run_potential.py
```

The list of Python packages and their version are specified in [`requirements.txt`](https://github.com/cchandre/guiding_center_intranet/blob/main/requirements.txt)
___
##  Configuration

- *Method*: string
  - 'diffusion_fo': computes the diffusion coefficients for the full orbits
  - 'diffusion_gc': computes the diffusion coefficients for the guiding centers
  - 'rotation_fo': computes the rotation numbers for the full orbits
  - 'rotation_gc': computes the rotation numbers for the guiding centers 
  - 'poincare_fo': plots the full orbits in the plane (*x*, *y*) for every period of the potential (stroboscopic plot)
  - 'poincare_gc': plots the guiding-center trajectories in the plane (*x*, *y*) for every period of the potential (stroboscopic plot)
- *Ntraj*: integer; number of trajectories to be integrated
- *Tf*: integer; number of periods for the integration of the trajectories
####
- *A*: float or array of floats; amplitude(s) of the electrostatic potential [theory: *A*=&epsilon;<sub>&delta;</sub>/*B*]
- *rho*: float or array of floats; value(s) of the Larmor radius; for full orbits, this value corresponds to the thermal Larmor radius
####
- *TwoStepIntegration*: boolean; if True, computes trajectories from 0 to 2&pi;*T*<sub>mid</sub>, removes the trapped trajectories, and continues integration from 2&pi;*T*<sub>mid</sub> to 2&pi;*T*<sub>f</sub>
- *Tmid*: integer; number of periods for the integration of trajectories in the first step (if *TwoStepIntegration*=True)
- *threshold*: float; value used to discriminate between trapped and untrapped trajectories (recommended: 4)
- *thresh_b*: float; value of *b* used to discriminate between diffusive and ballistic trajectories (according to the law *r*<sup>2</sup>(t) = (*a* *t*)<sup>b</sup>)
####
- *init*: string; 'random', 'fixed' or 'selected; method to generate initial conditions; if 'selected', *x0* and *y0* need to be provided
- *x0*: numpy array; values of the initial *x* if *init*='selected'
- *y0*: numpy array; values of the initial *y* if *init*='selected'
####
- *pyhamsys.TimeStep*: float; time step used by the integrator (recommended: 10<sup>-1</sup> for guiding centers and 5x10<sup>-3</sup> for full orbits)
- *pyhamsys.ode_solver*: string; indicates the symplectic integration scheme to be used (see [pyHamSys](https://pypi.org/project/pyhamsys/))
- *pyhamsys.CheckEnergy*: boolean; if True, the autonomous system is integrated, and the output (`.npz` file) includes the total energy (only if *SaveData*=True)
####
- *PlotResults*: boolean; if True, the results are plotted right after the computation
- *modulo*: boolean; if True, *x* and *y* are represented modulo 2&pi; (only for Method='poincare' and PlotResults=True)
- *grid*: boolean; if True, show the grid lines on plots
- *darkmode*: boolean; if True, plots are done in dark mode
- *fig_extension*: string; e.g., '.png', '.pdf', '.svg'; format of the figures to be saved
- *dpi*: integer; number of dots per inches for figures
####
- *SaveData*: boolean; if True, the results are saved in a `.npz` file; Poincaré sections and diffusion plots *r*<sup>2</sup>(*t*) are saved as *fig_extension* files; NB: the diffusion data are saved in a `.txt` file regardless of the value of *SaveData*
- *Parallelization*: configured as `parallelization`; use an integer core count or `"all"`
####
- *M*: integer; number of modes (default = 25 for 'turbulent') 

Each JSON configuration has this structure:
```json
{
	"schema_version": 1,
	"active_version": "default",
	"versions": {
		"default": {
			"parallelization": 1,
			"pyhamsys": {},
			"defaults": {},
			"sweep": {}
		}
	}
}
```
`pyhamsys` contains the parameters sent directly to the pyHamSys integrator.
`defaults` defines shared parameters, and `sweep` defines parameter values that
are expanded into cases. A version can also provide `cases`, a list of explicit
parameter overrides.

Notebook configs are intentionally smaller: they omit terminal runner settings
such as `parallelization`, `PlotResults`, and `SaveData`.

---
Reference: 
- M. Stanzani, F. Arlotti, G. Ciraolo, X. Garbet, C. Chandre, *Transition to super-diffusive transport in turbulent plasmas*, [arXiv:2309.02461](https://arxiv.org/abs/2309.02461)
```bibtex
@unpublished{stanzani2023,
  title = {Transition to super-diffusive transport in turbulent plasmas},
  author = {Stanzani, M. and Arlotti, F. and Ciraolo, G. and Garbet, X. and Chandre, C.},
  year = {2023},
  URL = {https://arxiv.org/abs/2309.02461}
}
```
For more information: <cristel.chandre@cnrs.fr>
