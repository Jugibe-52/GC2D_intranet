# Configuración de potenciales sobre malla

Los archivos conf/<notebook|terminal>/potential/<group>/v_x.json describen
simulaciones cuyo campo se materializa como GridPotential.

El flujo es:

~~~text
GridPotential + (TrajectoryGC | TrajectoryFC)
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
      "potential": {},
      "trajectory": {},
      "integration": {},
      "solver": {},
      "output": {}
    }
  }
}
~~~

## Bloque potential

Este bloque construye exclusivamente el campo físico. No contiene parámetros de
trayectoria.

| Campo | Tipo | Descripción |
|---|---:|---|
| type | hdf5, mock o hdf5_or_mock | Selecciona la fuente de datos. |
| path | str | Ruta del archivo HDF5. |
| B | float | Escala empleada al normalizar datos HDF5. |
| indx | list[int] | Selección de media y modos. |
| nx, ny | int o null | Resolución final de la malla. |
| k | int entre 1 y 5 | Orden de interpolación de GridPotential. |
| denoising | bool | Activa el filtrado gaussiano. |
| sigma | float | Anchura del filtro. |
| mock | object | Parámetros de la fuente sintética muestreada. |

El orden k pertenece a potential porque describe cómo GridPotential interpola
el campo; no debe aparecer dentro de trajectory.

Comportamiento por type:

- hdf5 exige que path exista.
- mock genera un campo periódico y lo materializa sobre una malla.
- hdf5_or_mock usa HDF5 cuando está disponible y recurre al mock en otro caso.

### potential.mock

| Campo | Tipo | Descripción |
|---|---:|---|
| A | float | Amplitud global. |
| M | int | Número máximo de modos usados para generar el campo. |
| seed | int | Semilla reproducible. |
| nx, ny | int | Resolución si no se define en el bloque padre. |

Aunque el mock se genera a partir de una suma de modos, este flujo crea un
GridPotential. El flujo de configuración fourier crea en cambio un
FourierPotential analítico.

### Formato HDF5

Se esperan los datasets:

~~~text
Rcells
Zcells
freqs
fields
~~~

fields debe tener forma (len(freqs), len(y), len(x)). La carga separa la
frecuencia cero, descarta frecuencias negativas, ordena por amplitud, normaliza
con B, aplica indx y, si se solicita, el filtrado.

## Bloque trajectory

| Campo | Tipo | Descripción |
|---|---:|---|
| type | gc o fc | Selecciona TrajectoryGC o TrajectoryFC. |
| rho | float no negativo | Radio de Larmor. |
| eta | float | Parámetro del modelo; no puede ser cero en FC. |
| Ntraj | int positivo | Número solicitado de trayectorias. |
| init | random o fixed | Estrategia de condiciones iniciales. |

fo se acepta únicamente como alias legado de entrada y se normaliza a fc.
Trajectory no contiene ni copia GridPotential.

## Bloque integration

| Campo | Tipo | Descripción |
|---|---:|---|
| n_max | int positivo | Número de muestras de la sección, incluidos los extremos. |

El intervalo termina en 2*pi*(n_max - 1), de modo que las muestras se sitúan en
periodos completos.

## Bloque solver

| Campo | Tipo | Argumento de system.simulate |
|---|---:|---|
| TimeStep | float positivo | step |
| ode_solver | str | method |
| CheckEnergy | bool | check_energy |

El motor numérico está dentro de classes/system. solver sustituye al nombre del
antiguo paquete y no forma parte de Potential ni de Trajectory.

## Bloque output

| Campo | Tipo | Descripción |
|---|---:|---|
| wrap | bool | Envuelve las posiciones al dominio periódico al dibujar. |
| plot | bool | Genera y, si corresponde, guarda la figura. |
| data | bool | Guarda t, y y los diagnósticos disponibles. |
| extension | str | Formato de la figura. |
| dpi | int | Resolución de salida. |

La carpeta de salida refleja la ruta de configuración:

~~~text
conf/terminal/potential/assay/v_1.json
-> outputs/terminal/potential/assay/v_1/
~~~
