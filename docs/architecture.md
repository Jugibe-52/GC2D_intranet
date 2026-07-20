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
`set_initial_state(...)`. `split(...)`, `pack_components(...)` y `particle_count(...)`
centralizan el formato físico del estado. Los resultados de `split(...)` tienen
componentes con nombre: `[x, y]` para GC y `[x, y, vx, vy]` para FC. Una
trayectoria no conoce el potencial ni el algoritmo de integración.

### Area

`Area` hereda de `TrajectoryGC`: sus bloques `[x, y]` son puntos ordenados en
sentido antihorario que delimitan un cuadrado o un círculo. Sus constructores
`Area.square(...)` y `Area.circle(...)` generan el contorno, y
`calculate_area(...)` calcula el área orientada inicial o transportada. Al ser
una trayectoria GC también puede utilizarse directamente con `SystemGC`.

### System

Combina exactamente un `Potential` con una `Trajectory` compatible:

- `SystemGC` construye el potencial efectivo y las ecuaciones de centro guía.
- `SystemFC` construye las ecuaciones de ciclotrón completo y la aceleración
  eléctrica que consume el integrador.

Ambos sistemas exponen `hamiltonian(...)` y `simulate(...)`. La implementación
numérica BM4 es privada: no constituye otra API ni puede seleccionarse desde el
exterior.

Cuando contiene un `Area`, `SystemGC.animate_area(...)` combina la solución con
el potencial efectivo y el campo eléctrico. La animación transporta el contorno
y representa simultáneamente el error relativo
`(A(t) - A(0)) / abs(A(0))`.

La implementación gráfica vive en un módulo privado; `SystemGC` conserva el
método público porque es quien dispone simultáneamente del potencial efectivo,
la trayectoria y la solución.

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

El integrador mantiene estructuras privadas diferentes del estado físico:

- GC usa dos copias del estado y un momento extendido opcional.
- FC añade únicamente el momento extendido opcional.

Estas estructuras, el acoplamiento GC y los flujos directo/adjunto FC pertenecen
a `_integration`. Se construyen y descomponen mediante `Trajectory.split(...)`
y `Trajectory.pack_components(...)`, sin duplicar el layout físico dentro de
`System`.

La interfaz permite construir trayectorias con `from_components(...)`, por lo
que los notebooks no necesitan conocer el orden plano. Dentro de los algoritmos,
`Trajectory.as_blocks(...)` interpreta el estado como
`(componentes, partículas, *muestras)` y `from_blocks(...)` recupera el formato
plano. Ambas transformaciones son vistas cuando el layout de memoria lo permite.
`Solution` conserva la trayectoria que produjo el resultado y
`solution.components()` devuelve directamente sus bloques físicos con nombre.

El orden por componentes mantiene contiguos todos los valores de una magnitud y
favorece las evaluaciones vectorizadas del potencial. El vector plano se conserva
como contrato estable del motor de composición; los flujos evitan reempaquetar
una misma representación física y el acoplamiento GC aplana directamente los
bloques que produce la operación matricial.

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
