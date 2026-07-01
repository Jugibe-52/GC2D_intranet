# `FourierSystem` y `HamSys`

Este documento explica como se conectan la clase local `FourierSystem` y la clase externa `HamSys` de `pyhamsys`.

Archivos relevantes:

- `src/classes/fourier_system.py`: modelo Fourier sintetico del sistema.
- `.venv/lib/python3.11/site-packages/pyhamsys.py`: implementacion instalada de `HamSys` y de los integradores simplecticos.
- `src/workflows/integration.py`: decide que integrador usar segun `traj_type`.
- `src/workflows/params.py`: normaliza parametros y construye `FourierSystem`.

## Idea general

`FourierSystem` representa el potencial turbulento como una serie de Fourier y proporciona las funciones dinamicas que necesitan los integradores de `pyhamsys`.

La relacion principal es:

```text
FourierSystem(HamSys)
        |
        | implementa y_dot(), k_dot(), hamiltonian(), chi(), chi_star()
        v
pyhamsys.solve_ivp_sympext(...) o pyhamsys.solve_ivp_symp(...)
```

`HamSys` no conoce el modelo fisico concreto. Solo define el contrato que debe cumplir un sistema hamiltoniano para poder integrarse.

## `HamSys`

`HamSys` es la clase base de `pyhamsys` para sistemas hamiltonianos.

En su constructor recibe `ndof`, el numero de grados de libertad. Puede ser entero o semi-entero:

- `1`: sistema autonomo con un grado de libertad.
- `1.5`: sistema con un grado de libertad y dependencia temporal explicita.
- `2`: sistema autonomo con dos grados de libertad.
- `2.5`: sistema con dos grados de libertad y dependencia temporal explicita.

Internamente guarda:

- `_ndof`: parte entera de `ndof`.
- `_time_dependent`: `True` si `ndof` termina en `.5`.

En este proyecto se usa:

```python
super().__init__(ndof=1.5 if dict_["traj_type"] == "gc" else 2.5)
```

Esto significa:

- `traj_type="gc"`: centro guia, estado reducido `(x, y)`, sistema dependiente del tiempo.
- `traj_type="fo"`: orbita completa, estado `(x, y, vx, vy)`, sistema dependiente del tiempo.

### Metodos importantes de `HamSys`

`compute_vector_field(hamiltonian)` puede construir automaticamente `y_dot`, `k_dot` y `hamiltonian` a partir de un Hamiltoniano simbolico con `sympy`.

`FourierSystem` no usa este camino: implementa esos metodos directamente con arrays de NumPy porque el potencial Fourier vectorizado ya esta disponible de forma explicita.

`rectify_sol(sol, check_energy=False)` ajusta la solucion devuelta por el integrador cuando se ha anadido una variable energetica `k`. Si `check_energy=True`, tambien calcula `sol.err`.

`compute_energy(sol)` evalua `hamiltonian(sol.t, sol.y)` y, si el sistema depende explicitamente del tiempo, suma `sol.k` para comprobar la conservacion de la energia extendida.

## Integradores de `pyhamsys`

Hay dos integradores usados por este proyecto.

### `solve_ivp_sympext`

Se usa para `traj_type="gc"`:

```python
sol = solve_ivp_sympext(
    system,
    (0, t_eval.max()),
    y0,
    step=system.TimeStep,
    t_eval=t_eval,
    method=system.ode_solver,
    check_energy=system.CheckEnergy,
)
```

Recibe un objeto `HamSys`. Por eso `FourierSystem` debe implementar:

- `y_dot(t, y)`: campo vectorial.
- `k_dot(t, y)`: derivada de la variable energetica si `CheckEnergy=True`.
- `hamiltonian(t, y)`: necesario para calcular el error energetico.

`solve_ivp_sympext` usa una extension en espacio de fases para integrar sistemas no separables de forma simplectica.

### `solve_ivp_symp`

Se usa para `traj_type="fo"`:

```python
sol = solve_ivp_symp(
    system.chi,
    system.chi_star,
    (0, t_eval.max()),
    y0,
    step=system.TimeStep,
    t_eval=t_eval,
    method=system.ode_solver,
)
sol = system.rectify_sol(sol, check_energy=system.CheckEnergy)
```

Aqui no se pasa el objeto completo, sino dos mapas de flujo:

- `chi(h, t, y)`
- `chi_star(h, t, y)`

Estos mapas aplican los subflujos exactos que componen el esquema simplectico.

## `FourierSystem`

`FourierSystem` hereda de `HamSys` y representa el modelo Fourier sintetico usado por el flujo `fourier`.

Se construye con un diccionario de parametros:

```python
system = FourierSystem(params)
```

El constructor copia cada clave del diccionario como atributo:

```python
for key in dict_:
    setattr(self, key, dict_[key])
```

Por eso la clase espera parametros planos como:

- `traj_type`
- `A`
- `M`
- `rho`
- `eta`
- `Ntraj`
- `Tf`
- `init`
- `TimeStep`
- `ode_solver`
- `CheckEnergy`

El loader de configuracion traduce los bloques JSON (`pyhamsys`, `output`, etc.) a estos atributos antes de crear el sistema.

## Construccion del potencial Fourier

El potencial se representa mediante coeficientes complejos `phic`.

Pasos principales:

1. Se fija una semilla aleatoria (`27`) para que las fases sean reproducibles.
2. Se generan fases aleatorias para los modos Fourier.
3. Se crea una malla de indices `(n, m)` con valores de `0` a `M`.
4. Para `n, m >= 1`, se asigna:

```text
phic[n, m] = A / (n^2 + m^2)^1.5 * exp(i * phase[n, m])
```

5. Los modos fuera del disco `sqrt(n^2 + m^2) > M` se anulan.
6. Si `traj_type="gc"`, se aplica la correccion FLR con `J0(rho * sqrt(n^2 + m^2))`.
7. Se precomputan coeficientes del campo:

```python
fft_phi_ = [-m * phic, n * phic]
```

Esta precalculacion evita recomputar derivadas Fourier en cada evaluacion de `y_dot`.

## Estructura del estado

La forma del vector inicial depende de `traj_type` y `CheckEnergy`.

| Caso | Estado por trayectoria | Vector completo |
|---|---|---|
| `gc`, sin energia | `(x, y)` | longitud `2 * Ntraj` |
| `gc`, con energia | `(x, y)` mas `k` interno del integrador extendido | entrada longitud `2 * Ntraj` |
| `fo`, sin energia | `(x, y, vx, vy)` | longitud `4 * Ntraj` |
| `fo`, con energia | `(x, y, vx, vy, k)` | longitud `5 * Ntraj` |

Para `gc`, si `CheckEnergy=True`, `solve_ivp_sympext` anade y maneja `k`.

Para `fo`, `FourierSystem.initial_conditions()` anade explicitamente `k` cuando `CheckEnergy=True`, y `chi()`/`chi_star()` lo actualizan.

## Condiciones iniciales

`initial_conditions(type=...)` admite:

- `random`: posiciones aleatorias en `[0, 2*pi)`.
- `fixed`: malla regular cuadrada. Si `Ntraj` no es cuadrado perfecto, se reduce al cuadrado perfecto inferior.
- `selected`: usa `x0` e `y0` de la configuracion. Ambos deben tener la misma forma.

Para orbita completa (`fo`) se anaden velocidades perpendiculares aleatorias:

```text
vx = cos(phi_perp)
vy = sin(phi_perp)
```

## Metodos dinamicos de `FourierSystem`

### `y_dot(t, y)`

Devuelve el campo vectorial del sistema reducido:

```text
y_dot = (dx/dt, dy/dt)
```

Internamente evalua todos los modos Fourier de forma vectorizada y devuelve un array con la misma forma que `y`.

Este metodo es obligatorio para `solve_ivp_sympext`.

### `k_dot(t, y)`

Devuelve la derivada de la variable energetica extendida `k`.

`pyhamsys` lo necesita cuando:

- el sistema depende explicitamente del tiempo;
- `CheckEnergy=True`;
- se usa `solve_ivp_sympext`.

### `potential(t, y)`

Evalua el potencial Fourier en las posiciones dadas.

En `gc`, el Hamiltoniano es directamente este potencial.

### `hamiltonian(t, y)`

Devuelve la energia del sistema:

- Para `gc`: `potential(t, y)`.
- Para `fo`: termino cinetico mas potencial reescalado por `rho` y `eta`.

Es necesario para `compute_energy()` y para guardar `sol.err` cuando `CheckEnergy=True`.

### `chi(h, t, y)` y `chi_star(h, t, y)`

Implementan los dos mapas de flujo usados por `solve_ivp_symp` en orbita completa.

Ambos actualizan:

- posiciones `(x, y)`;
- velocidades `(vx, vy)`;
- energia extendida `k`, si `CheckEnergy=True`.

`chi` y `chi_star` aplican los mismos subflujos en orden inverso, que es el contrato esperado por el integrador simplectico.

## Flujo de ejecucion en el proyecto

El camino normal es:

```text
JSON config
  -> load_fourier_config()
  -> config.cases()
  -> run_workflow(params)
  -> integrate_simulation(params)
  -> make_system()/ensure_system()
  -> FourierSystem(params)
  -> pyhamsys integrator
  -> SimulationResult(system, sol, elapsed)
```

`integrate_simulation()` decide el integrador:

- `traj_type="gc"`: usa `solve_ivp_sympext(system, ...)`.
- `traj_type="fo"`: usa `solve_ivp_symp(system.chi, system.chi_star, ...)`.

`run_workflow()` anade comportamiento de nivel superior:

- ejecutar la integracion;
- plotear si corresponde;
- guardar datos si `SaveData=True`.

## Puntos delicados

`FourierSystem` depende de que el diccionario de parametros ya este normalizado. En particular:

- `traj_type` debe ser `"gc"` o `"fo"`.
- `eta` debe existir para `fo` y no puede ser cero.
- `x0` e `y0` deben tener la misma forma si `init="selected"`.
- `CheckEnergy` cambia la forma esperada del estado en `fo`.

La clase tambien modifica `self.Ntraj` en dos casos:

- `init="fixed"`: ajusta a una malla cuadrada.
- `init="selected"`: lo reemplaza por el numero real de puntos iniciales.

Esto es importante para interpretar la forma de `sol.y` y para plotear despues.

## Resumen

`HamSys` aporta el contrato numerico y utilidades para integracion hamiltoniana.

`FourierSystem` aporta el modelo fisico concreto:

- genera el potencial Fourier;
- define el campo vectorial;
- define el Hamiltoniano;
- define los mapas simplecticos de orbita completa;
- prepara condiciones iniciales compatibles con `pyhamsys`.

La separacion permite que el codigo del proyecto cambie la configuracion y la generacion del potencial sin reimplementar los integradores simplecticos.
