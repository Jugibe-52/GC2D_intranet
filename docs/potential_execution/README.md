# Ejecución de potenciales sobre malla

run_potential.py ejecuta trayectorias sobre un GridPotential cargado desde HDF5
o generado como campo mock. El potencial, la trayectoria y el sistema son
objetos separados durante todo el flujo.

## Entrada

~~~sh
python3 run_potential.py
python3 run_potential.py --config-group assay --config-version v_1
~~~

Los perfiles se leen de:

~~~text
conf/<terminal|notebook>/potential/<group>/<version>.json
~~~

## Construcción

La configuración produce tres pasos explícitos:

~~~python
potential = GridPotential(...)
trajectory = TrajectoryGC(...)  # o TrajectoryFC(...)
system = create_system(potential, trajectory)
~~~

GridPotential contiene la malla, los campos y los interpoladores. Trajectory
contiene los parámetros de la partícula y la definición de su estado. System
recibe ambos y es el único objeto que conoce las ecuaciones completas.

## Flujo

1. Se carga el perfil JSON.
2. potential.type selecciona HDF5, mock o hdf5_or_mock.
3. Los datos se validan y se construye GridPotential.
4. trajectory.type selecciona TrajectoryGC o TrajectoryFC. El alias fo se
   normaliza a fc antes de crear el objeto.
5. create_system(potential, trajectory) devuelve SystemGC o SystemFC.
6. system.simulate(...) prepara las condiciones iniciales si no se proporcionan
   explícitamente y ejecuta la integración.
7. SimulationResult se entrega a plotting y exportación.

El entry point no importa solvers concretos ni contiene una rama GC/FC.

## Responsabilidades de GridPotential

GridPotential:

- valida que cada campo tenga la forma de Grid;
- construye los interpoladores;
- aplica la política periódica o de recorte;
- evalúa phi y sus derivadas espaciales y temporales;
- puede remuestrear o producir una versión giro-promediada.

GridPotential no guarda rho, eta, estados ni opciones de integración.

## Responsabilidades de Trajectory

TrajectoryGC define el estado:

~~~text
[x_1, ..., x_N, y_1, ..., y_N]
~~~

TrajectoryFC define:

~~~text
[x_1, ..., x_N, y_1, ..., y_N,
 vx_1, ..., vx_N, vy_1, ..., vy_N]
~~~

Además valida rho y eta, separa posiciones y velocidades y construye las
condiciones iniciales random o fixed. La malla puede entregarse como dominio
para inicializar, pero no queda almacenada dentro de Trajectory.

## SystemGC

SystemGC construye el potencial efectivo psi a partir del campo físico phi y
del rho de TrajectoryGC. La copia física y la versión efectiva permanecen
separadas.

Con E = -grad(psi):

~~~text
x_dot = E_y = -dpsi/dy
y_dot = -E_x = dpsi/dx
H = psi
k_dot = -dpsi/dt
~~~

simulate(...) utiliza la extensión simpléctica de espacio de fases y devuelve
el estado físico sin exponer las variables duplicadas del algoritmo.

## SystemFC

SystemFC usa siempre el potencial físico phi, sin giro-promedio. Define:

~~~text
velocity_scale = rho / (2 * abs(eta))
electric_scale = sign(eta) / rho
larmor_frequency = 1 / (2 * eta)
~~~

y mantiene las ecuaciones:

~~~text
x_dot  = vx * velocity_scale
y_dot  = vy * velocity_scale
vx_dot = Ex * electric_scale + vy * larmor_frequency
vy_dot = Ey * electric_scale - vx * larmor_frequency
~~~

La integración conserva los mismos mapas chi y chi_star y el mismo orden de
composición. Si check_energy está activo, System añade y elimina internamente
la variable k de forma simétrica.

## Configuración del solver

El bloque solver se pasa a system.simulate:

~~~json
"solver": {
  "TimeStep": 0.02,
  "ode_solver": "BM4",
  "CheckEnergy": true
}
~~~

El motor vive bajo classes/system y no es una cuarta entidad del dominio.

## Investigación y visualización

Los diagnósticos que necesitan el campo y la dinámica reciben un System ya
compuesto. Los gráficos puramente espaciales pueden recibir Potential. Por
ejemplo, el cálculo de área de un conjunto de trayectorias GC usa:

- system.trajectory para separar posiciones;
- system.effective_potential para psi;
- SimulationResult para los estados y tiempos.

Así, ninguna clase llamada PotentialResearch necesita fingir que el sistema y
el potencial son la misma entidad.

## Módulos conceptuales

~~~text
classes/potential/grid.py       Grid
classes/potential/grid_potential.py
classes/trajectory/gc.py
classes/trajectory/fc.py
classes/system/gc.py
classes/system/fc.py
classes/system/                 solver, solution y diagnósticos
workflows/                      carga, plotting y exportación
~~~
