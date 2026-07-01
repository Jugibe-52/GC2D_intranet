# Ejecucion `run_gc2d.py`: modelo `GC2D` con `Potential`

Esta ejecucion esta pensada para probar o correr el modelo `GC2D` basado en potenciales discretos. El potencial puede venir de un archivo HDF5 o generarse como potencial mock.

## Entrada principal

```bash
python run_gc2d.py
```

Tambien puede recibir configuracion explicita:

```bash
python run_gc2d.py --config-group test --config-version v_1
```

## Configuracion

El script carga perfiles desde:

```text
conf/<group>/<version>/run_gc2d.json
```

La funcion responsable es:

```python
load_run_gc2d_config(...)
```

El objeto de configuracion construye el sistema con:

```python
config.build_system()
```

Ese metodo crea:

```text
Potential -> GC2D
```

## Flujo

1. `run_gc2d.py` configura imports, logging y argumentos CLI.
2. Carga el JSON con `load_run_gc2d_config`.
3. `PotentialConfig.build()` carga un HDF5 con `extract_potential` o genera uno con `mock_potential`.
4. `RunGC2DConfig.build_system()` construye `GC2D(potential, traj, k=...)`.
5. Genera condiciones iniciales con `hs.initial_conditions(...)`.
6. Integra:
   - `traj_type == "gc"` usa `solve_ivp_sympext(hs, ...)`.
   - `traj_type == "fo"` usa `solve_ivp_symp(hs.chi, hs.chi_star, ...)`.
7. Si `plot=True`, llama a `plot_sol(hs, sol, ...)`.

## Modulos principales

- `run_gc2d.py`: entry point de terminal para el flujo `GC2D + Potential`.
- `src/config.py`: carga de `run_gc2d.json`, `PotentialConfig` y `RunGC2DConfig`.
- `src/classes/potential.py`: clase `Potential`, interpolacion y gyroaverage.
- `src/classes/gc2d.py`: clase `GC2D` y dinamica del sistema.
- `src/workflows/potentials.py`: `extract_potential`, `mock_potential`.
- `src/workflows/plotting.py`: `plot_sol`, `plot_potential`.

## Diferencia con `gc2d.py`

`run_gc2d.py` usa potenciales discretos/interpolados:

```text
Potential -> GC2D -> pyhamsys
```

`gc2d.py` usa el modelo Fourier sintetico:

```text
GC2Dt -> pyhamsys
```

Por eso `run_gc2d.py` es mas adecuado para validar el flujo con datos HDF5 o potenciales mock.
