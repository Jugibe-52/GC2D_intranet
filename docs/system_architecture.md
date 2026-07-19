# Arquitectura de Potential, Trajectory y System

## Objetivo

El dominio se divide en tres entidades con una única dirección de dependencia:

~~~text
Potential ----+
              +--> System --> SimulationResult
Trajectory ---+
~~~

Potential y Trajectory pueden crearse, validarse y probarse de forma
independiente. System es la raíz de composición: conoce ambas entidades y añade
las ecuaciones y el proceso numérico.

## Estructura de paquetes

~~~text
classes/
  potential/
    Potential
    GridPotential
    FourierPotential
    Grid, fields e interpolación
  trajectory/
    Trajectory
    TrajectoryGC
    TrajectoryFC
    estado y condiciones iniciales
  system/
    System
    SystemGC
    SystemFC
    create_system
    integración, solución y diagnósticos
~~~

No existe una cuarta capa de dominio para el integrador. El motor numérico es
un detalle interno de system.

## Potential

Potential es una clase abstracta que define la evaluación común del campo:

- valor del potencial en tiempo y posición;
- derivadas espaciales;
- derivada temporal;
- campo eléctrico;
- dominio espacial cuando existe una malla explícita.

No contiene:

- rho ni eta;
- tipo GC o FC;
- estado de partículas;
- parámetros del solver;
- resultados de simulación.

### GridPotential

GridPotential representa campos discretos. Es responsable de:

- Grid y política de contorno;
- campo medio y modos temporales;
- interpolación de partes reales y complejas;
- evaluación y remuestreo;
- transformación giro-promediada cuando SystemGC la solicita.

Puede construirse a partir de HDF5 o de un generador mock.

### FourierPotential

FourierPotential representa el potencial mediante coeficientes espectrales.
Evalúa directamente la suma de Fourier y sus derivadas, sin pasar por splines.

Esta implementación conserva:

- las fases asociadas a la semilla;
- el truncamiento por número de modos;
- la convención temporal;
- las derivadas espectrales;
- el filtro de Bessel que SystemGC solicita para el modelo efectivo.

GridPotential y FourierPotential son intercambiables para System porque cumplen
el mismo contrato, no porque usen la misma técnica de evaluación.

## Trajectory

Trajectory describe una familia de estados físicos. Es independiente del campo
sobre el que evolucionará.

Responsabilidades comunes:

- validar rho, eta y número de trayectorias;
- declarar kind y state_dimension;
- construir o almacenar el estado inicial;
- separar posiciones y velocidades;
- validar la forma del estado.

### TrajectoryGC

Usa kind=gc y dos componentes:

~~~text
[x, y]
~~~

No posee velocidades independientes.

### TrajectoryFC

Usa kind=fc y cuatro componentes:

~~~text
[x, y, vx, vy]
~~~

También define las escalas físicas derivadas de rho y eta:

~~~text
velocity_scale = rho / (2 * abs(eta))
electric_scale = sign(eta) / rho
larmor_frequency = 1 / (2 * eta)
~~~

rho y eta deben ser distintos de cero.

### Compatibilidad del identificador

Las entidades solo usan gc y fc. La cadena fo pertenece a formatos antiguos:

~~~text
input "fo" -> normalización -> "fc" -> TrajectoryFC
~~~

Ninguna comprobación interna debe ramificar por fo.

## System

System recibe las dos entidades:

~~~python
system = create_system(potential, trajectory)
~~~

La fábrica devuelve:

- SystemGC para TrajectoryGC;
- SystemFC para TrajectoryFC.

System conserva referencias explícitas a potential y trajectory, y expone las
operaciones que requieren ambas:

- y_dot(t, state);
- hamiltonian(t, state);
- k_dot(t, state);
- electric_field(...);
- simulate(...).

### SystemGC

SystemGC prepara dos vistas distintas:

- physical_potential: phi, el campo de entrada;
- effective_potential: psi, el campo que incorpora el giro-promedio.

Las ecuaciones GC utilizan exclusivamente psi. El potencial físico no se
sobrescribe.

### SystemFC

SystemFC utiliza phi sin giro-promedio. Contiene los mapas chi y chi_star que
componen la integración explícita y puede convertir posiciones FC a centros
guía para análisis.

## Simulación

La API de alto nivel es:

~~~python
solution = system.simulate(
    y0=None,
    t_span=(0.0, final_time),
    step=time_step,
    n_save_step=sample_count,
    method="BM4",
    check_energy=True,
    progress=False,
)
~~~

Si y0 es None, System pide a Trajectory que prepare un estado en el
dominio de Potential. La trayectoria no almacena el potencial después de esa
operación.

simulate selecciona el algoritmo según la subclase de System. El llamador no
importa funciones de integración ni bifurca por kind.

## Solution y SimulationResult

System.simulate(...) devuelve una Solution con:

- solution.t;
- solution.y con el estado físico;
- solution.k si se comprobó la energía dependiente del tiempo;
- solution.err si se calculó el error energético.

Los workflows envuelven esa solución en un SimulationResult que contiene:

- system;
- solution;
- tiempo transcurrido;
- figuras opcionales producidas por workflows.

La interpretación del estado se delega en result.system.trajectory. Esto evita
duplicar np.split según GC o FC en plotting y exportación.

## Configuración y workflows

La configuración actúa como adaptador:

~~~text
JSON
  -> Potential config
  -> Trajectory config
  -> solver config
  -> create_system(...)
  -> system.simulate(...)
~~~

Los workflows coordinan carga, barridos, paralelización, gráficos y
persistencia. No contienen ecuaciones de movimiento.

## Reglas de dependencia

1. potential no importa trajectory ni system.
2. trajectory no importa potential ni system.
3. system puede importar potential y trajectory.
4. workflows pueden importar las tres carpetas.
5. el motor numérico no aparece en la API de Potential o Trajectory.
6. los objetos de investigación que necesitan campo y dinámica reciben System.

Estas reglas son parte del contrato arquitectónico y evitan que una de las
entidades vuelva a absorber las responsabilidades de las otras.
