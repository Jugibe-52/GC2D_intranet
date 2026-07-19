# Ejecucion `run_potential.py`: trayectorias sobre `Potential`

Esta ejecucion usa un potencial discreto con una de las dos dinamicas concretas:
`PotentialHamsysGC` (centro guia) o `PotentialHamsysFC` (ciclotron completo). El potencial
puede venir de un archivo HDF5 o generarse como potencial mock.

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
conf/terminal/potential/<group>/<version>.json
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
Potential -> create_potential_hamsys() -> PotentialHamsysGC | PotentialHamsysFC
```

## Flujo

1. `run_potential.py` configura imports, logging y argumentos CLI.
2. Carga la configuracion con `load_potential_config`.
3. `PotentialConfig.build()` carga un HDF5 con `extract_potential` o genera uno con `mock_potential`.
4. `PotentialRunConfig.build_system()` llama a `create_potential_hamsys(potential, traj)`; el orden de interpolacion `k` pertenece a `potential`.
5. Genera condiciones iniciales con `make_initial_conditions(hs, ...)`.
6. Integra:
   - `traj_type == "gc"` usa `solve_ivp_sympext(hs, ...)`.
   - `traj_type == "fo"` usa `solve_ivp_symp(hs.chi, hs.chi_star, ...)`.
7. Si `plot=True`, llama a `plot_sol(hs, sol, ...)`.

## Diagnósticos de investigación

Las clases de trayectoria conservan solo el potencial y la dinamica. La
visualizacion de campos y trayectorias, junto con los diagnosticos que estudian
propiedades numericas, se construye aparte sobre una trayectoria ya creada:

```python
from classes import PotentialResearch

research = PotentialResearch(system)
z0 = research.guiding_center_square_initial_conditions(points_per_side=50)
# solution = solve_ivp_sympext(system, z0, ...)
area = research.guiding_center_polygon_area(solution)
animation = research.animate_electric_psi_area_conservation(solution)
# Otros ejemplos: research.plot_phi_psi(), research.animate_electric_psi(...)
```

También se exporta `potential_researche` como alias de la clase para conservar
el nombre solicitado en los notebooks de investigación.

## Secuencia de ejecución de `PotentialHamsysGC.y_dot`

En una integración de centro guía, `solve_ivp_sympext` llama repetidamente a
`system.y_dot(t, z)` para obtener la derivada del estado en cada paso interno.
La cadena principal de llamadas es:

```text
solve_ivp_sympext
└── PotentialHamsysGC.y_dot(t, z)
    ├── get_positions(z)
    ├── electric_field(t, x, y)
    │   ├── psi(t, x, y, dx=1) / psi(t, x, y, dy=1)
    │   └── field_at_time(...)
    │       ├── phic_interp(...)
    │       │   ├── wrap_or_clip(...)
    │       │   └── derivadas de RectBivariateSpline
    │       └── reconstrucción temporal de los modos
    └── concatenate(...) -> z_dot
```

### Centro guía (`gc`)

`PotentialHamsysGC` sigue estos pasos:

1. `get_positions(z)` interpreta el vector como
   `z = [x_1, ..., x_N, y_1, ..., y_N]` y devuelve por separado los bloques
   `x` e `y`.
2. `electric_field(t, x, y)` utiliza el potencial efectivo `psi`. Este potencial
   contiene los campos giro-promediados cuando `rho != 0`; si `rho == 0`,
   coincide con los campos originales.
3. El campo eléctrico efectivo se calcula como

   $$
   E_x=-\frac{\partial\psi}{\partial x},\qquad
   E_y=-\frac{\partial\psi}{\partial y}.
   $$

4. `psi` delega en `field_at_time`. Como se solicitan derivadas en posiciones
   concretas, `phic_interp` aplica la política de contorno mediante
   `wrap_or_clip` y evalúa las derivadas de los interpoladores
   `RectBivariateSpline`.
5. Para cada modo complejo $\psi_k(x,y)$ de frecuencia $\omega_k$,
   `field_at_time` reconstruye el potencial en el instante `t`:

   $$
   \psi(x,y,t)=\bar\psi(x,y)
   +2\sum_k\operatorname{Re}\left[\psi_k(x,y)e^{i\omega_k t}\right].
   $$

   La misma reconstrucción se aplica a $\partial_x\psi$ y
   $\partial_y\psi$.
6. `y_dot` concatena las componentes y devuelve

   $$
   \dot z=[\dot x,\dot y]=[E_y,-E_x]
   =\left[-\frac{\partial\psi}{\partial y},
   \frac{\partial\psi}{\partial x}\right].
   $$

El integrador usa este resultado para avanzar la trayectoria y vuelve a llamar
a `y_dot` con nuevos valores de `t` y `z`.

### Órbita completa (`fo`)

`PotentialHamsysFC` conserva el valor historico `traj["type"] == "fo"` en la
configuracion. Su estado se organiza como
`z = [x, y, vx, vy]` y `electric_field(..., effective=False)` calcula el campo
físico no giro-promediado $\mathbf E=-\nabla\phi$.

Con `output="reduced"`, `y_dot` devuelve `[Ey, -Ex]`. Esta es la forma que
utilizan `chi` y `chi_star` durante la integración con `solve_ivp_symp`.

Con `output="full"`, devuelve el campo vectorial completo:

$$
\dot z=\left[
v_xv_{fo},\ v_yv_{fo},\
E_x\phi_{fo}+v_y\omega_L,\
E_y\phi_{fo}-v_x\omega_L
\right].
$$

## Modulos principales

- `run_potential.py`: entry point de terminal para el flujo de potenciales y trayectorias.
- `src/config.py`: carga de configuracion, `PotentialConfig` y `PotentialRunConfig`.
- `src/classes/potential/potential.py`: clase `Potential`, interpolacion y gyroaverage.
- `src/classes/potential/potential_hamsys.py`: clase base `PotentialHamsys` y comportamiento comun.
- `src/classes/potential/potential_hamsys_gc.py`: dinamica `PotentialHamsysGC`.
- `src/classes/potential/potential_hamsys_fc.py`: dinamica `PotentialHamsysFC`.
- `src/classes/potential/potential_hamsys_research.py`: estabilidad y dinamica tangente `PotentialHamsysResearch`.
- `src/classes/potential/potential_research.py`: visualizaciones y diagnosticos `PotentialResearch`.
- `src/workflows/initial_conditions.py`: construccion de estados iniciales.
- `src/workflows/potentials.py`: `extract_potential`, `mock_potential`.
- `src/workflows/plotting.py`: `plot_sol`, `plot_potential`.

## Diferencia con `run_fourier.py`

`run_potential.py` usa potenciales discretos/interpolados:

```text
Potential -> PotentialHamsysGC | PotentialHamsysFC -> pyhamsys
```

`run_fourier.py` usa el modelo Fourier sintetico:

```text
FourierSystem -> pyhamsys
```

Por eso `run_potential.py` es mas adecuado para validar el flujo con datos HDF5 o potenciales mock.
