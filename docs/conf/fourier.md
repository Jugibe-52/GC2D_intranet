# Configuración de potenciales Fourier

Los archivos conf/<notebook|terminal>/fourier/<group>/v_x.json describen lotes
cuya fuente física es un FourierPotential.

El flujo construido para cada caso es:

~~~text
FourierPotential + (TrajectoryGC | TrajectoryFC)
    -> create_system(...)
    -> SystemGC | SystemFC
    -> system.simulate(...)
~~~

## Estructura

~~~json
{
  "schema_version": 1,
  "active_version": "default",
  "versions": {
    "default": {
      "parallelization": 1,
      "solver": {
        "TimeStep": 0.1,
        "ode_solver": "BM4",
        "CheckEnergy": true
      },
      "output": {
        "plot": false,
        "data": false,
        "extension": ".png",
        "dpi": 200
      },
      "defaults": {},
      "sweep": {}
    }
  }
}
~~~

schema_version debe ser 1. active_version selecciona el perfil predeterminado
y versions contiene uno o varios perfiles.

## Generación de casos

defaults define los valores compartidos. sweep expande el producto cartesiano
de listas de parámetros. Como alternativa, cases puede contener una lista de
sobrescrituras explícitas.

El runner puede usar parallelization como número máximo de procesos o como el
valor all.

## Construcción de FourierPotential

Los parámetros físicos de la fuente Fourier son:

| Campo | Tipo | Descripción |
|---|---:|---|
| A | float | Amplitud global del espectro. |
| M | int | Número máximo de modos; se anulan los situados fuera del disco espectral. |
| seed | int, opcional | Semilla de las fases; si se omite se conserva el valor reproducible del modelo. |

FourierPotential evalúa directamente los modos y sus derivadas. No se convierte
en un potencial interpolado, porque eso modificaría la evaluación espectral del
modelo original.

## Construcción de Trajectory

| Campo | Tipo | Descripción |
|---|---:|---|
| traj_type | gc o fc | Selecciona TrajectoryGC o TrajectoryFC. |
| rho | float | Radio de Larmor, no negativo. |
| eta | float | Parámetro del modelo; debe ser distinto de cero para FC. |
| Ntraj | int | Número solicitado de condiciones iniciales. |
| init | random, fixed o selected | Estrategia de inicialización. |
| x0, y0 | arrays | Posiciones usadas cuando init es selected. |

El alias histórico fo se acepta solo al cargar datos antiguos y se normaliza a
fc. Todo objeto y toda rama interna usan gc o fc.

Method se conserva como opción de workflow. Si traj_type no está presente, el
loader puede inferirlo del sufijo de valores históricos como poincare_gc o
poincare_fo; el resultado se normaliza antes de construir Trajectory.

## Horizonte de simulación

Tf es el número de periodos. El intervalo físico es:

~~~text
(0, 2*pi*Tf)
~~~

y se guardan Tf + 1 secciones uniformes, incluidos ambos extremos.

## Bloque solver

solver contiene únicamente opciones numéricas consumidas por
system.simulate(...):

| Campo | Tipo | Argumento normalizado |
|---|---:|---|
| TimeStep | float positivo | step |
| ode_solver | str | method |
| CheckEnergy | bool | check_energy |

Los nombres de las claves se mantienen para compatibilidad de configuración.
El bloque ya no identifica un paquete externo: el solver forma parte de
classes/system.

## Condiciones iniciales

- random genera posiciones uniformes en el dominio periódico.
- fixed genera una malla cuadrada y puede ajustar Ntraj al cuadrado inferior.
- selected usa x0 e y0, que deben tener la misma forma.
- TrajectoryFC añade vx y vy a partir de una fase perpendicular.

La trayectoria define el layout del estado; System solo lo consume al simular.

## Bloque output

| Campo | Tipo | Descripción |
|---|---:|---|
| plot | bool | Genera la figura configurada por el workflow. |
| data | bool | Guarda el resultado en NPZ. |
| extension | str | Extensión de la figura. |
| dpi | int | Resolución de la figura. |
| modulo | bool | Representa posiciones en el toro cuando corresponda. |
| grid | bool | Activa la rejilla de la figura. |

La carpeta de salida se deriva de la ruta del perfil, por ejemplo:

~~~text
conf/terminal/fourier/test/v_1.json
-> outputs/terminal/fourier/test/v_1/
~~~

Los campos TwoStepIntegration, Tmid, threshold, thresh_b y darkmode son opciones
de workflows experimentales; no forman parte de las tres entidades del dominio.
