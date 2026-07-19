# Ejecución de potenciales Fourier

run_fourier.py ejecuta perfiles por lotes cuya fuente de campo es
FourierPotential. Ya no existe una clase que mezcle el espectro, la trayectoria
y el integrador.

## Entrada

~~~sh
python3 run_fourier.py
python3 run_fourier.py --version symplectic_grid
python3 run_fourier.py --config-group assay --config-version v_1
~~~

Los perfiles se leen de:

~~~text
conf/<terminal|notebook>/fourier/<group>/<version>.json
~~~

## Construcción de cada caso

Para cada combinación de defaults y sweep, la capa de configuración realiza
conceptualmente:

~~~python
potential = FourierPotential(...)
trajectory = TrajectoryGC(...)  # o TrajectoryFC(...)
initialize_trajectory(trajectory, potential.grid)
system = create_system(potential, trajectory)
result = system.simulate(...)
~~~

FourierPotential conserva los coeficientes complejos, las fases reproducibles,
el truncamiento espectral y la evaluación directa de derivadas.

Trajectory conserva rho, eta, el layout del estado y la estrategia de
condiciones iniciales. No conoce el potencial.

create_system devuelve SystemGC o SystemFC y comprueba que la combinación es
válida.

## Flujo batch

1. El runner carga el perfil y expande los casos.
2. Cada caso construye un FourierPotential.
3. El selector gc o fc construye la Trajectory correspondiente. fo se acepta
   únicamente como alias de configuración antigua.
4. El workflow inicializa Trajectory usando potential.grid.
5. create_system(potential, trajectory) compone las entidades.
6. system.simulate(...) integra el estado de Trajectory con el motor simpléctico.
7. Se devuelve SimulationResult.
8. El workflow representa o guarda el resultado según output.
9. Los casos pueden distribuirse con multiprocess sin cambiar las entidades.

## Selección numérica

El runner no bifurca por tipo de trayectoria. La decisión vive en System:

- SystemGC integra el campo no separable en espacio de fases extendido.
- SystemFC integra mediante chi y chi_star y rectifica la solución extendida.

Los parámetros proceden del bloque solver:

| Configuración | system.simulate |
|---|---|
| TimeStep | step |
| ode_solver | method |
| CheckEnergy | check_energy |

Los coeficientes y el orden de los subflujos son los mismos que en el proceso
numérico anterior; solo cambia su ubicación dentro de classes/system.

## Resultados

SimulationResult reúne:

- el System que se simuló;
- la solución física t, y;
- la variable extendida k y el error energético, si se solicitaron;
- el tiempo de ejecución;
- los manejadores de figura cuando se representa el resultado.

Plotting y exportación consultan system.trajectory para separar x, y y, en FC,
vx y vy. No inspeccionan el nombre de una clase de potencial concreta.

## Módulos conceptuales

~~~text
classes/potential/fourier.py    FourierPotential
classes/trajectory/gc.py       TrajectoryGC
classes/trajectory/fc.py       TrajectoryFC
classes/system/gc.py           SystemGC
classes/system/fc.py           SystemFC
classes/system/                solver y resultado
workflows/                     batch, plotting y exportación
~~~
