# Ejecucion `run_potential.py`: modelo `PotentialSystem` con `Potential`

Esta ejecucion esta pensada para probar o correr el modelo `PotentialSystem` basado en potenciales discretos. El potencial puede venir de un archivo HDF5 o generarse como potencial mock.

## Entrada principal

```bash
python run_potential.py
```

Tambien puede recibir configuracion explicita:

```bash
python run_potential.py --config-group test --config-version v_1
```

## Configuracion

El script carga perfiles desde:

```text
conf/potential/<group>/<version>.py
```

La funcion responsable es:

```python
load_potential_config(...)
```

El objeto de configuracion construye el sistema con:

```python
config.build_system()
```

Ese metodo crea:

```text
Potential -> PotentialSystem
```

## Flujo

1. `run_potential.py` configura imports, logging y argumentos CLI.
2. Carga la configuracion con `load_potential_config`.
3. `PotentialConfig.build()` carga un HDF5 con `extract_potential` o genera uno con `mock_potential`.
4. `PotentialRunConfig.build_system()` construye `PotentialSystem(potential, traj, k=...)`.
5. Genera condiciones iniciales con `hs.initial_conditions(...)`.
6. Integra:
   - `traj_type == "gc"` usa `solve_ivp_sympext(hs, ...)`.
   - `traj_type == "fo"` usa `solve_ivp_symp(hs.chi, hs.chi_star, ...)`.
7. Si `plot=True`, llama a `plot_sol(hs, sol, ...)`.

## Modulos principales

- `run_potential.py`: entry point de terminal para el flujo `PotentialSystem + Potential`.
- `src/config.py`: carga de configuracion, `PotentialConfig` y `PotentialRunConfig`.
- `src/classes/potential.py`: clase `Potential`, interpolacion y gyroaverage.
- `src/classes/potential_system.py`: clase `PotentialSystem` y dinamica del sistema.
- `src/workflows/potentials.py`: `extract_potential`, `mock_potential`.
- `src/workflows/plotting.py`: `plot_sol`, `plot_potential`.

## Diferencia con `run_fourier.py`

`run_potential.py` usa potenciales discretos/interpolados:

```text
Potential -> PotentialSystem -> pyhamsys
```

`run_fourier.py` usa el modelo Fourier sintetico:

```text
FourierSystem -> pyhamsys
```

Por eso `run_potential.py` es mas adecuado para validar el flujo con datos HDF5 o potenciales mock.
