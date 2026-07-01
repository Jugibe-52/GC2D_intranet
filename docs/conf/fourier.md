# `conf/<notebook|terminal>/fourier/<group>/v_x.json`

`conf/<notebook|terminal>/fourier/<group>/v_x.json` configura casos basados en el modelo `FourierSystem`.

- `conf/notebook/...`: configuracion reducida para uso interactivo con `workflows_api`; el bloque `output.data` controla si `run_workflow()` guarda datos.
- `conf/terminal/...`: configuracion completa para `run_fourier.py` y ejecuciones batch.

El loader correspondiente es:

```python
from config import load_fourier_config
```

## Estructura general

```json
{
    "schema_version": 1,
    "active_version": "default",
    "versions": {
        "default": {
            "parallelization": 1,
            "pyhamsys": {},
            "output": {},
            "defaults": {},
            "sweep": {},
            "cases": []
        }
    }
}
```

## Campos de nivel superior

| Campo | Tipo | Descripcion |
|---|---:|---|
| `schema_version` | `int` | Version del esquema. El codigo exige `1`. |
| `active_version` | `str` | Perfil usado si no se pasa `--version`. |
| `versions` | `object` | Diccionario de perfiles. Cada clave es un nombre de perfil. |

## Perfil de `versions`

| Campo | Tipo | Descripcion |
|---|---:|---|
| `parallelization` | `int` o `"all"` | Solo terminal. Numero de procesos para `run_fourier.py`. `"all"` usa todos los cores disponibles. |
| `pyhamsys` | `object` | Parametros enviados al integrador de `pyhamsys`. |
| `output` | `object` | Controla si se generan/guardan figuras y datos en `outputs/...`. |
| `defaults` | `object` | Parametros base compartidos por los casos del perfil. |
| `sweep` | `object` | Barrido de parametros. Se genera el producto cartesiano de sus listas. |
| `cases` | `list[object]` | Casos explicitos. Si existe y no esta vacio, tiene prioridad sobre `sweep`. |

En `notebook`, se omiten los campos que solo consume el runner de terminal (`parallelization`) y perfiles legacy como `symplectic_grid`.

## Bloque `pyhamsys`

Este bloque separa lo que controla directamente la llamada a `pyhamsys` del resto de la configuracion fisica, numerica y de salida.

| Campo | Tipo | Usado por | Descripcion |
|---|---:|---|---|
| `TimeStep` | `float` | `solve_ivp_sympext`, `solve_ivp_symp` | Paso temporal enviado como `step`. |
| `ode_solver` | `str` | `solve_ivp_sympext`, `solve_ivp_symp` | Metodo simplectico enviado como `method`, por ejemplo `"BM4"`. |
| `CheckEnergy` | `bool` | `solve_ivp_sympext`, `rectify_sol` | Activa la comprobacion energetica/autonomizacion cuando aplica. |

`config.py` fusiona estos campos en cada caso antes de construir `FourierSystem`, porque la clase todavia espera atributos planos como `system.TimeStep` y `system.CheckEnergy`.

## Expansion de valores

`config.py` permite dos formas abreviadas:

```json
"x0": {
  "linspace": [0.5, 1.0, "Ntraj"]
}
```

Genera `np.linspace(start, stop, num)`. `num` puede ser un entero o el nombre de otro parametro.

```json
"y0": {
  "constant": 4.5,
  "num": "Ntraj"
}
```

Genera `np.full(num, constant)`.

## Parametros del sistema Fourier

| Campo | Tipo | Usado por | Descripcion |
|---|---:|---|---|
| `Method` | `str` | `to_symp_params`, `run_workflow` | Nombre del metodo. Si `traj_type` no existe, se infiere del sufijo final: `poincare_gc` -> `gc`. |
| `traj_type` | `"gc"` o `"fo"` | `FourierSystem`, integracion | Tipo de trayectoria. Si falta, se infiere desde `Method`. |
| `A` | `float` | `FourierSystem` | Amplitud de los coeficientes Fourier del potencial. Normalmente se define en `sweep`. |
| `M` | `int` | `FourierSystem` | Numero maximo de modos Fourier. Modos con `sqrt(n^2 + m^2) > M` se anulan. |
| `rho` | `float` | `FourierSystem` | Radio de Larmor. En `gc` se usa en la correccion FLR con Bessel `J0`. |
| `eta` | `float` | `FourierSystem`, `chi`, `chi_star` | Parametro de orbita completa. Si falta, se rellena con `rho`. Para `fo` no puede ser cero. |

## Parametros de integracion propios del flujo

| Campo | Tipo | Usado por | Descripcion |
|---|---:|---|---|
| `Ntraj` | `int` | `initial_conditions` | Numero de trayectorias. Con `init="fixed"` se ajusta al cuadrado perfecto inferior. |
| `Tf` | `int` | `integrate_case` | Numero de periodos. Se evalua en `2*pi*arange(0, Tf + 1)`. |

## Condiciones iniciales

| Campo | Tipo | Usado por | Descripcion |
|---|---:|---|---|
| `init` | `"random"`, `"fixed"` o `"selected"` | `initial_conditions` | Estrategia de inicializacion. |
| `x0` | array o expansion | `init="selected"` | Coordenadas iniciales `x`. Debe tener la misma forma que `y0`. |
| `y0` | array o expansion | `init="selected"` | Coordenadas iniciales `y`. Debe tener la misma forma que `x0`. |

Comportamiento:

- `random`: genera `2*Ntraj` posiciones aleatorias en `[0, 2*pi)`.
- `fixed`: genera una malla regular cuadrada.
- `selected`: usa `x0` e `y0`.
- Si `traj_type="fo"`, se añaden velocidades perpendiculares aleatorias.
- Si `traj_type="fo"` y `CheckEnergy=True`, se añade una variable energetica `k`.

## Bloque `output`

| Campo | Tipo | Descripcion |
|---|---:|---|
| `plot` | `bool` | Si es `true`, genera la figura Poincare y la guarda en la carpeta `outputs/...` derivada de la configuracion. |
| `data` | `bool` | Si es `true`, `run_workflow()`/`run_fourier.py` guardan un `.npz` comprimido con `t`, `x`, `y` y, si aplica, `k`, `vx`, `vy`. Si es `false`, no se guardan datos. |
| `extension` | `str` | Extension de la figura guardada, por ejemplo `.png` o `.pdf`. |
| `dpi` | `int` | Resolucion usada al guardar la figura. |

`config.py` traduce este bloque a los flags internos que todavia esperan algunos workflows (`PlotResults`, `SavePlot`, `SaveData`, `extension`, `dpi`).

## Plot y salida por caso

| Campo | Tipo | Usado por | Descripcion |
|---|---:|---|---|
| `modulo` | `bool` | `SimulationResult.plot_poincare` | Si es `true`, representa `x` e `y` modulo `2*pi`. |
| `grid` | `bool` | `SimulationResult.plot_poincare` | Activa rejilla en la figura Poincare. |

La carpeta de salida no se define en la configuracion: se deriva automaticamente de la ruta de configuracion. Por ejemplo:

```text
conf/notebook/fourier/test/v_1.json -> outputs/notebook/fourier/test/v_1/
conf/terminal/fourier/test/v_1.json -> outputs/terminal/fourier/test/v_1/
conf/terminal/fourier/assay/v_1.json -> outputs/terminal/fourier/assay/v_1/
```

Los archivos guardados usan el perfil como prefijo compacto y una fecha. Por ejemplo, el perfil `notebook_demo` genera nombres como `notebook_YYYYmmdd_HHMMSS.npz`.

## Parametros heredados o reservados

Estos campos aparecen en algunos perfiles, pero no controlan el flujo actual `run_fourier.py -> run_workflow -> integrate_case`:

| Campo | Estado |
|---|---|
| `TwoStepIntegration` | Heredado/reservado para flujos de dos etapas. |
| `Tmid` | Heredado/reservado para dos etapas. |
| `threshold` | Heredado/reservado para clasificar trayectorias. |
| `thresh_b` | Heredado/reservado para diagnosticos de transporte. |
| `darkmode` | No se usa en el plotting actual. |
| `PlotResults` | Compatibilidad interna; usar `output.plot`. |
| `SavePlot` | Compatibilidad interna; usar `output.plot`. |
| `SaveData` | Compatibilidad interna; usar `output.data`. |
