# GC2D

Código de investigación para estudiar trayectorias de partículas en un
potencial electrostático bidimensional.

El proyecto se organiza alrededor de cuatro entidades:

```text
Potential + Trajectory -> System -> Solution
```

- `Potential` representa el campo físico sobre una rejilla periódica.
- `TrajectoryGC` y `TrajectoryFC` describen el estado de una o varias
  partículas.
- `Area` especializa una trayectoria GC como contorno cuadrado o circular.
- `SystemGC` y `SystemFC` combinan un potencial con una trayectoria y realizan
  la integración temporal.

Los notebooks de `notebooks/developements/` son los únicos puntos de entrada.
No hay ejecutables de terminal, configuración externa ni una API de workflows.

## Instalación

Se requiere Python 3.11 o posterior.

```bash
python -m pip install -r requirements.txt
```

El paquete queda instalado en modo editable, por lo que los cambios en `src/`
están disponibles inmediatamente en los notebooks.

## Ejemplo GC

```python
import numpy as np

from classes import Potential, SystemGC, TrajectoryGC

potential = Potential.random(
    A=0.7,
    M=25,
    nx=64,
    ny=64,
    seed=27,
    interpolation_order=5,
)

x0 = potential.grid.xmin + potential.grid.period / 2
y0 = potential.grid.ymin + potential.grid.period / 2

trajectory = TrajectoryGC.from_components(
    x=np.array([x0]),
    y=np.array([y0]),
    rho=0.3,
)

system = SystemGC(potential, trajectory)
solution = system.simulate(
    t_span=(0.0, 6 * np.pi),
    step=0.001,
    n_save_step=361,
    check_energy=True,
    progress=True,
)
```

`SystemGC` integra el movimiento del centro guía sobre el potencial efectivo
giro-promediado. El potencial físico original permanece en `potential`.

## Ejemplo FC

```python
import numpy as np

from classes import Potential, SystemFC, TrajectoryFC

potential = Potential.random(
    A=0.7,
    M=25,
    nx=64,
    ny=64,
    seed=27,
    interpolation_order=5,
)

x0 = potential.grid.xmin + potential.grid.period / 2
y0 = potential.grid.ymin + potential.grid.period / 2

trajectory = TrajectoryFC.from_components(
    x=np.array([x0]),
    y=np.array([y0]),
    vx=np.array([1.0]),
    vy=np.array([0.0]),
    rho=0.3,
    eta=0.01,
)

system = SystemFC(potential, trajectory)
solution = system.simulate(
    t_span=(0.0, 2 * np.pi),
    step=0.001,
    n_save_step=101,
    check_energy=True,
)
```

`SystemFC` integra la órbita ciclotrónica completa sobre el potencial físico.

## Organización del estado

Los estados usan bloques, no valores intercalados. Para `N` trayectorias:

```text
GC: [x_1, ..., x_N, y_1, ..., y_N]
FC: [x_1, ..., x_N, y_1, ..., y_N,
     vx_1, ..., vx_N, vy_1, ..., vy_N]
```

Por ejemplo, dos estados GC se escriben así:

```python
trajectory = TrajectoryGC(rho=0.3)
state = trajectory.pack_components(
    np.array([x1, x2]),
    np.array([y1, y2]),
)
trajectory.set_initial_state(state)

components = trajectory.split(state)
print(components.x, components.y)
print(trajectory.particle_count(state))  # 2
```

Para código de usuario se recomienda el constructor semántico, que evita
depender de ese orden interno:

```python
trajectory = TrajectoryGC.from_components(
    x=np.array([x1, x2]),
    y=np.array([y1, y2]),
    rho=0.3,
)
```

`as_blocks(...)` expone una vista con forma
`(componentes, partículas, *muestras)` y `from_blocks(...)` realiza la
transformación inversa. El integrador usa estos ejes explícitos internamente,
pero conserva el vector plano como formato estable de entrada y salida.
`pack_components(...)` permite obtener ese vector directamente desde la clase;
`from_components(...)` lo utiliza para construir la trayectoria en un solo paso.

Las trayectorias aceptan el estado inicial en el constructor y también permiten
reemplazarlo con `set_initial_state(...)`. Las geometrías reutilizables de
contornos pertenecen a `Area`; otras condiciones específicas del experimento
se preparan en el notebook.

`TrajectoryGC.split(...)` devuelve componentes con nombre `x` e `y`;
`TrajectoryFC.split(...)` añade `vx` y `vy`. El método de clase
`pack_components(...)` realiza la operación inversa. De esta forma, el formato
físico del estado pertenece a la trayectoria y el integrador no necesita repetir
su estructura.

## Áreas

`Area` es una trayectoria GC cuyos puntos delimitan un contorno orientado. Se
puede construir como un cuadrado o un círculo y pasar directamente a
`SystemGC`:

```python
from classes import Area

area = Area.square(
    center=(np.pi, np.pi),
    side=1.0,
    points_per_side=40,
    rho=0.3,
)
system = SystemGC(potential, area)
solution = system.simulate(step=0.005)

transported_area = area.calculate_area(
    solution.y,
    period=potential.grid.period,
)

animation = system.animate_area(
    solution,
    frames=120,
    interval=50,
)
```

El constructor alternativo `Area.circle(...)` recibe `center`, `radius` y el
número total de puntos del contorno. `calculate_area(...)` acepta tanto el
estado inicial como una serie temporal completa y aplica la fórmula del cordón;
con `period` también trata correctamente los cruces del borde periódico.
`SystemGC.animate_area(...)` muestra el contorno sobre el potencial efectivo y
su campo eléctrico, junto con
`(A(t) - A(0)) / abs(A(0))` en un segundo panel.

## Integración y resultado

`System.simulate(...)` utiliza la composición simpléctica BM4. El método es
fijo para mantener una sola ruta numérica comprensible.

Los argumentos habituales son:

- `t_span`: tiempo inicial y final.
- `step`: paso interno máximo.
- `n_save_step`: número de muestras que se guardan, incluidos los extremos.
- `check_energy`: calcula la energía generalizada y su error.
- `progress`: muestra el avance de integraciones GC largas.

La solución ofrece como mínimo:

- `solution.t`: tiempos guardados.
- `solution.y`: estados, con una columna por tiempo.
- `solution.n_steps`: número de pasos internos.
- `solution.k`: momento extendido cuando se comprueba la energía.
- `solution.err`: error energético máximo cuando se solicita.
- `solution.components()`: bloques físicos con nombre según la trayectoria.

Por ejemplo:

```python
components = solution.components()
x = components.x  # forma (partículas, tiempos)
y = components.y
```

El Hamiltoniano también puede evaluarse directamente:

```python
energy = system.hamiltonian(solution.t, solution.y)
```

## Potencial

`Potential.random(...)` genera el potencial periódico reproducible utilizado
por los notebooks de desarrollo. El potencial permite:

- evaluar el campo y sus derivadas;
- obtener el giro-promedio requerido por GC;
- representar el campo con `potential.plot()`;
- animarlo con `potential.animate(...)`.

## Notebooks de desarrollo

- `test_generalized_energy_.ipynb`: convergencia y conservación de la energía
  generalizada en GC, más una comprobación corta de FC.
- `test_dX_dY.ipynb`: conservación del área de un contorno transportado en GC.

La estructura interna se resume en
[`docs/architecture.md`](docs/architecture.md).
