# Ejecucion `gc2d.py`: modelo Fourier `GC2Dt`

Esta ejecucion esta pensada para correr casos por lotes desde terminal. Usa el modelo Fourier sintetico `GC2Dt`, definido en `src/classes/gc2dt.py`, y los workflows publicos de `src/gc2d_workflows.py`.

## Entrada principal

```bash
python gc2d.py
```

Tambien puede recibir configuracion explicita:

```bash
python gc2d.py --config-group test --config-version v_1
```

## Configuracion

El script carga perfiles desde:

```text
conf/<group>/<version>/gc2d.json
```

La funcion responsable es:

```python
load_gc2dt_config(...)
```

El objeto de configuracion devuelve una lista de casos con:

```python
config.cases()
```

Cada caso es un diccionario de parametros para construir un `GC2Dt`.

## Flujo

1. `gc2d.py` configura imports, logging y argumentos CLI.
2. Carga el JSON con `load_gc2dt_config`.
3. Expande la lista de casos.
4. Decide si ejecuta en paralelo con `multiprocess`.
5. Para cada caso llama a `run_case(params, plot=...)`.
6. `run_case` construye o recibe un `GC2Dt`.
7. `integrate_case` genera condiciones iniciales e integra.
8. Si `SaveData=True`, `save_data(system, sol)` escribe un `.mat`.
9. Al final, `plt.show()` muestra figuras pendientes.

## Modulos principales

- `gc2d.py`: entry point de terminal para ejecucion por lotes.
- `src/config.py`: carga y expansion de configuracion.
- `src/classes/gc2dt.py`: modelo Fourier `GC2Dt`.
- `src/workflows/params.py`: normalizacion y construccion del sistema.
- `src/workflows/integration.py`: integracion `gc`/`fo`.
- `src/workflows/cases.py`: workflow de alto nivel `run_case`.
- `src/workflows/export.py`: exportacion `.mat`.
- `src/workflows/plotting.py`: graficos Poincare y diagnosticos.

## Diferencia con notebook

En notebook no conviene usar `gc2d.py` directamente. La API reutilizable es:

```python
from gc2d_workflows import make_system, run_case, integrate_case, plot_poincare
```

`gc2d.py` existe para ejecucion batch fuera de notebook.
