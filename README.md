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

trajectory = TrajectoryGC(np.array([x0, y0]), rho=0.3)

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

trajectory = TrajectoryFC(
    np.array([x0, y0, 1.0, 0.0]),
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
trajectory.set_initial_state(
    np.array([x1, x2, y1, y2])
)
```

Las trayectorias aceptan el estado inicial en el constructor y también permiten
reemplazarlo con `set_initial_state(...)`. Las geometrías reutilizables de
contornos pertenecen a `Area`; otras condiciones específicas del experimento
se preparan en el notebook.

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
