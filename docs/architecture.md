# Arquitectura

GC2D tiene tres entidades públicas y dos variantes dinámicas.

```text
                     +--> TrajectoryGC --> SystemGC --+
Potential -----------|                                |--> Solution
                     +--> TrajectoryFC --> SystemFC --+
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

Una trayectoria no genera condiciones iniciales y no conoce el potencial. La
preparación de puntos, cuadrados u otras geometrías pertenece al notebook que
define el experimento.

### System

Combina exactamente un `Potential` con una `Trajectory` compatible:

- `SystemGC` construye el potencial efectivo y las ecuaciones de centro guía.
- `SystemFC` construye las ecuaciones y los flujos de ciclotrón completo.

Ambos sistemas exponen `hamiltonian(...)` y `simulate(...)`. La implementación
numérica BM4 es privada: no constituye otra API ni puede seleccionarse desde el
exterior.

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
    SystemFC,
    SystemGC,
    TrajectoryFC,
    TrajectoryGC,
)
```

Los detalles de rejilla, interpolación y composición simpléctica permanecen
internos. No existen factories, aliases de compatibilidad, workflows ni puntos
de entrada de terminal.
