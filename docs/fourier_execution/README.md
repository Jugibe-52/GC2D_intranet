# Ejecucion `run_fourier.py`: modelo Fourier `FourierSystem`

Esta ejecucion esta pensada para correr casos por lotes desde terminal. Usa el modelo Fourier sintetico `FourierSystem`, definido en `src/classes/fourier_system.py`, y los workflows publicos de `src/workflows_api.py`.

## Entrada principal

```bash
python run_fourier.py
```

Tambien puede recibir configuracion explicita:

```bash
python run_fourier.py --config-group test --config-version v_1
```

## Configuracion

El script carga perfiles desde:

```text
conf/terminal/fourier/<group>/<version>.json
```

La funcion responsable es:

```python
load_fourier_config(...)
```

El objeto de configuracion devuelve una lista de casos con:

```python
config.cases()
```

Cada caso es un diccionario de parametros para construir un `FourierSystem`.

## Flujo

1. `run_fourier.py` configura imports, logging y argumentos CLI.
2. Carga la configuracion con `load_fourier_config`.
3. Expande la lista de casos.
4. Decide si ejecuta en paralelo con `multiprocess`.
5. Para cada caso llama a `run_case(params, plot=..., save=False)`.
6. `run_case` construye o recibe un `FourierSystem`.
7. `integrate_case` genera condiciones iniciales e integra.
8. Si `SaveData=True`, `save_data(system, sol)` escribe un `.npz` una sola vez desde el runner.
9. Al final, `plt.show()` muestra figuras pendientes.

## Modulos principales

- `run_fourier.py`: entry point de terminal para ejecucion por lotes.
- `src/config.py`: carga y expansion de configuracion.
- `src/classes/fourier_system.py`: modelo Fourier `FourierSystem`.
- `src/workflows/params.py`: normalizacion y construccion del sistema.
- `src/workflows/integration.py`: integracion `gc`/`fo`.
- `src/workflows/cases.py`: workflow de alto nivel `run_case`.
- `src/workflows/export.py`: exportacion `.npz`.
- `src/workflows/plotting.py`: graficos Poincare y diagnosticos.

## Diferencia con notebook

En notebook no conviene usar `run_fourier.py` directamente. La API reutilizable es:

```python
from workflows_api import make_system, run_case, integrate_case, plot_poincare
```

`run_fourier.py` existe para ejecucion batch fuera de notebook.
En notebook, `run_case(..., save=True)` respeta `output.data` de `conf/notebook/...`.
