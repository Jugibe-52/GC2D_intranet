# Arquitectura

GC2D tiene cuatro entidades públicas y dos variantes dinámicas.

```text
Area --> TrajectoryGC --+
                        +--> SystemGC --+
Potential --------------|               |--> Solution
                        +--> SystemFC --+
TrajectoryFC -----------+
```

## Responsabilidades

### Potential

Contiene la rejilla periódica y el campo electrostático. Evalúa el potencial y
sus derivadas espaciales o temporales. También contiene las visualizaciones que
usan los notebooks.

`Potential` no conoce partículas, estados ni integradores. `SystemGC` construye
una versión giro-promediada para su dinámica efectiva; `SystemFC` usa el
potencial físico directamente.

### Trajectory

Contiene los parámetros de la partícula y su estado inicial:

- `TrajectoryGC`: bloques `[x, y]` y radio de Larmor `rho`.
- `TrajectoryFC`: bloques `[x, y, vx, vy]`, `rho` y `eta`.

El estado inicial puede asignarse en el constructor o posteriormente con
`set_initial_state(...)`. Una trayectoria no conoce el potencial.

### Area

`Area` hereda de `TrajectoryGC`: sus bloques `[x, y]` son puntos ordenados en
sentido antihorario que delimitan un cuadrado o un círculo. Sus constructores
`Area.square(...)` y `Area.circle(...)` generan el contorno, y
`calculate_area(...)` calcula el área orientada inicial o transportada. Al ser
una trayectoria GC también puede utilizarse directamente con `SystemGC`.

### System

Combina exactamente un `Potential` con una `Trajectory` compatible:

- `SystemGC` construye el potencial efectivo y las ecuaciones de centro guía.
- `SystemFC` construye las ecuaciones y los flujos de ciclotrón completo.

Ambos sistemas exponen `hamiltonian(...)` y `simulate(...)`. La implementación
numérica BM4 es privada: no constituye otra API ni puede seleccionarse desde el
exterior.

Cuando contiene un `Area`, `SystemGC.animate_area(...)` combina la solución con
el potencial efectivo y el campo eléctrico. La animación transporta el contorno
y representa simultáneamente el error relativo
`(A(t) - A(0)) / abs(A(0))`.

### Solution

Es el resultado de `simulate(...)`. Solo transporta tiempos, estados y
diagnósticos de la integración. No decide cómo representar, guardar o analizar
los resultados.

## Dependencias

```text
Potential       Trajectory
     \             /
      \           /
        System GC/FC
             |
       integración BM4
             |
          Solution
```

Las dependencias solo avanzan hacia la composición y la integración. No hay
dependencias desde `Potential` hacia `Trajectory`, ni desde `Trajectory` hacia
`Potential`.

## Superficie pública

Los notebooks importan únicamente:

```python
from classes import (
    Potential,
    Area,
    SystemFC,
    SystemGC,
    TrajectoryFC,
    TrajectoryGC,
)
```

Los detalles de rejilla, interpolación y composición simpléctica permanecen
internos. No existen aliases de compatibilidad, workflows ni puntos de entrada
de terminal.
