# `make_params`

`make_params` is a notebook-oriented helper function, not a class. It builds a simulation parameter dictionary by copying a base dictionary and applying keyword overrides. The returned dictionary is ready to pass to `make_system()`, which creates the `GC2Dt` system that notebooks pass to `run_case()` or `integrate_case()`.

```python
from gc2d_core.gc2d_dict import dict_list
from gc2d_core.gc2d_notebook import make_params, make_system, run_case

params = make_params(
    dict_list[0],
    Tf=3,
    Tmid=1,
    Ntraj=12,
    M=6,
    TimeStep=0.1,
    TwoStepIntegration=False,
    PlotResults=True,
    SaveData=False,
    CheckEnergy=True,
)

system = make_system(params)
result = run_case(system, plot=True)
```

## Function Signature

```python
def make_params(base: dict, **overrides) -> dict:
```

## Inputs

`base`

The starting parameter dictionary. In notebooks this is usually one entry from `gc2d_core.gc2d_dict.dict_list`, for example `dict_list[0]`.

`**overrides`

Keyword arguments that replace values in `base`. This is the recommended way to create short notebook runs without editing the global parameter file.

## What It Does

1. Copies `base`, so the original dictionary is not modified.
2. Applies all `overrides`.
3. If `init == "selected"`, trims `x0` and `y0` to `Ntraj` unless those arrays were explicitly passed as overrides.
4. Calls `to_symp_params()` to normalize trajectory-related fields.
5. Returns the final parameter dictionary.

## Automatic Normalization

`make_params()` delegates trajectory normalization to `to_symp_params()`:

- `traj_type` is inferred from `Method` when it is not provided.
- Valid trajectory types are `"gc"` for guiding-center trajectories and `"fo"` for full-orbit trajectories.
- `eta` defaults to `rho` when it is not provided.
- Full-orbit runs require a non-zero `eta`; otherwise a `ValueError` is raised.

## Common Parameters

| Parameter | Type | Meaning |
| --- | --- | --- |
| `Method` | `str` | Simulation mode, for example `"poincare_gc"` or `"poincare_fo"`. The suffix is used to infer `traj_type` if needed. |
| `traj_type` | `str` | Trajectory model: `"gc"` or `"fo"`. Usually inferred from `Method`. |
| `A` | `float` | Potential amplitude. |
| `rho` | `float` | Larmor radius. For guiding-center runs it is also used in the FLR correction. |
| `eta` | `float` | Full-orbit parameter. Defaults to `rho`; must be non-zero for `traj_type="fo"`. |
| `Ntraj` | `int` | Number of trajectories. For `init="fixed"`, the model may adjust it to the nearest square grid size. |
| `Tf` | `int` | Number of periods used to build the integration sampling times. |
| `TimeStep` | `float` | Time step passed to the symplectic integrator. |
| `ode_solver` | `str` | Symplectic scheme name passed to `pyhamsys`, for example `"BM4"`. |
| `M` | `int` | Number of Fourier modes used by the turbulent potential model. |

## Initialization Parameters

| Parameter | Type | Meaning |
| --- | --- | --- |
| `init` | `str` | Initial-condition mode: `"random"`, `"fixed"`, or `"selected"`. |
| `x0` | array-like | Selected initial x positions. Required when `init="selected"`. |
| `y0` | array-like | Selected initial y positions. Required when `init="selected"`. |

When `init="selected"` and `x0`/`y0` come from `base`, `make_params()` trims them to `Ntraj`. If you pass `x0` or `y0` directly in `overrides`, the function keeps your explicit arrays unchanged.

## Plotting and Output Parameters

| Parameter | Type | Meaning |
| --- | --- | --- |
| `PlotResults` | `bool` | Whether the root-script workflow should plot results. In notebooks, `run_case(system, plot=True)` controls automatic plotting. |
| `modulo` | `bool` | If true, Poincare plots display positions modulo `2*pi`. |
| `grid` | `bool` | Plot grid option from the base configuration. |
| `darkmode` | `bool` | Plot style option from the base configuration. |
| `extension` | `str` | Figure extension used by saving paths, for example `".pdf"` or `".png"`. |
| `dpi` | `int` | Figure resolution for saved plots. |
| `SaveData` | `bool` | If true, `GC2Dt.save_data()` writes a MATLAB `.mat` file. |
| `CheckEnergy` | `bool` | If true, the integration includes energy checking where supported. |

## Notebook Usage Pattern

Use `make_params()` to keep notebook runs small and reproducible:

```python
params = make_params(
    dict_list[0],
    Tf=3,
    Ntraj=12,
    M=6,
    TimeStep=0.1,
    SaveData=False,
)

system = make_system(params)
result = run_case(system, plot=True)
```

This keeps the full project defaults in `gc2d_dict.py`, while the notebook documents only the parameters that are intentionally changed for the interactive run.
