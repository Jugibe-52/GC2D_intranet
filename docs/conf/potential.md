# `conf/<notebook|terminal>/potential/<group>/v_x.json`

`conf/<notebook|terminal>/potential/<group>/v_x.json` configura el flujo:

```text
Potential -> PotentialSystem -> pyhamsys
```

El loader correspondiente es:

```python
from config import load_potential_config
```

## Estructura general

```json
{
    "schema_version": 1,
    "active_version": "default",
    "versions": {
        "default": {
            "potential": {},
            "trajectory": {},
            "integration": {},
            "pyhamsys": {},
            "output": {}
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

## Bloque `potential`

Este bloque construye un `Potential`. Puede cargar datos HDF5 o generar un potencial mock.

| Campo | Tipo | Valores | Descripcion |
|---|---:|---|---|
| `type` | `str` | `"hdf5"`, `"mock"`, `"hdf5_or_mock"` | Selecciona la fuente del potencial. |
| `path` | `str` | ruta | Ruta del HDF5. Solo se usa con `hdf5` o `hdf5_or_mock`. |
| `B` | `float` | positivo | Campo usado para reescalar datos HDF5. En mock, si falta `mock.A`, se usa `1 / B`. |
| `indx` | `list[int]` | indices | Seleccion de modos HDF5. `0` selecciona la media; indices positivos seleccionan fluctuaciones tras ordenar por amplitud. |
| `nx` | `int` o `null` | resolucion | Resolucion final en `x`. Si existe, `Potential` reinterpola a esa malla. |
| `ny` | `int` o `null` | resolucion | Resolucion final en `y`. Si existe, `Potential` reinterpola a esa malla. |
| `denoising` | `bool` | `true`/`false` | Si es `true`, aplica filtro gaussiano a campos HDF5. |
| `sigma` | `float` | positivo | Sigma del filtro gaussiano cuando `denoising=true`. |
| `mock` | `object` | ver abajo | Parametros del potencial mock. |

Comportamiento por `type`:

- `"hdf5"`: exige que `path` exista. Si no existe, lanza `ConfigError`.
- `"mock"`: ignora `path` y genera un potencial sintetico.
- `"hdf5_or_mock"`: usa HDF5 si `path` existe; si no, cae a mock.

## Bloque `potential.mock`

| Campo | Tipo | Descripcion |
|---|---:|---|
| `A` | `float` | Amplitud global del potencial mock. Si falta, se usa `1 / B`. |
| `M` | `int` | Numero maximo de modos Fourier del mock. |
| `seed` | `int` | Semilla para fases aleatorias reproducibles. |
| `nx` | `int` | Resolucion mock en `x`, si `potential.nx` no esta definido. |
| `ny` | `int` | Resolucion mock en `y`, si `potential.ny` no esta definido. |

El mock crea un potencial periodico en `[0, 2*pi)` como suma de modos Fourier con fases aleatorias y amplitud decreciente:

```text
A / (n^2 + m^2)^1.5
```

## Bloque `trajectory`

Este bloque define la clase `PotentialSystem`.

| Campo | Tipo | Valores | Descripcion |
|---|---:|---|---|
| `type` | `str` | `"gc"` o `"fo"` | Tipo de trayectoria: centro guia u orbita completa. |
| `rho` | `float` | >= 0 | Radio de Larmor. Si no es cero, el sistema aplica `gyroaverage`. |
| `eta` | `float` | numero | Parametro de orbita completa. |
| `k` | `int` | entero | Orden de interpolacion pasado a `PotentialSystem`. Debe ser suficiente para `rho`. |
| `Ntraj` | `int` | entero | Numero de trayectorias iniciales. |
| `init` | `str` | `"random"` o `"fixed"` | Metodo de condiciones iniciales para `PotentialSystem`. |

Notas:

- `PotentialSystem.initial_conditions` acepta `random` y `fixed`.
- En `fo`, se añaden velocidades perpendiculares aleatorias.
- Si `rho` es mayor que `k * dx` o `k * dy`, el constructor lanza error para evitar interpolacion insuficiente.

## Bloque `integration`

| Campo | Tipo | Descripcion |
|---|---:|---|
| `n_max` | `int` | Numero de muestras de Poincare. Se evalua en `2*pi*arange(n_max)`. |

## Bloque `pyhamsys`

Este bloque agrupa solo parametros pasados a `pyhamsys`.

| Campo | Tipo | Descripcion |
|---|---:|---|
| `TimeStep` | `float` | Paso temporal enviado como `step`. |
| `ode_solver` | `str` | Metodo simplectico enviado como `method`, por ejemplo `"BM4"`. |
| `CheckEnergy` | `bool` | Activa comprobacion energetica cuando el integrador la soporta. |

## Bloque `output`

| Campo | Tipo | Descripcion |
|---|---:|---|
| `wrap` | `bool` | Si es `true`, `plot_sol` envuelve posiciones al dominio periodico. |
| `plot` | `bool` | Si es `true`, `run_potential.py` llama a `plot_sol` y guarda la figura en `outputs/...`. |
| `data` | `bool` | Si es `true`, guarda la solucion en formato `.npz` con `t`, `y` y, si existen, `err` y `k`. |
| `extension` | `str` | Extension de la figura guardada, por ejemplo `.png` o `.pdf`. |
| `dpi` | `int` | Resolucion usada al guardar la figura. |

El bloque `output` tambien puede existir en configuraciones de notebook. En ese caso permite decidir si una ejecucion programatica basada en esa configuracion debe persistir figura o datos.

La carpeta de salida se deriva automaticamente de la ruta de configuracion:

```text
conf/notebook/potential/test/v_1.json -> outputs/notebook/potential/test/v_1/
conf/terminal/potential/test/v_1.json -> outputs/terminal/potential/test/v_1/
conf/terminal/potential/assay/v_1.json -> outputs/terminal/potential/assay/v_1/
```

## Relacion con HDF5

Cuando se usa HDF5, `extract_potential` espera datasets:

```text
Rcells
Zcells
freqs
fields
```

El shape esperado de `fields` es:

```text
(len(freqs), len(y), len(x))
```

Despues:

- separa la frecuencia cero como valor medio;
- elimina frecuencias negativas;
- ordena fluctuaciones por amplitud;
- reescala por `omega * B`;
- aplica `indx`;
- opcionalmente aplica `denoising`.
