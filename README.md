# guiding_center_intranet

Herramientas para simular centros guía y órbitas ciclotrónicas completas en
potenciales electrostáticos bidimensionales.

La arquitectura pública se basa en tres entidades independientes:

~~~text
Potential + Trajectory -> System -> SimulationResult
~~~

- Potential describe el campo electrostático y cómo evaluarlo.
- Trajectory describe el modelo de partícula, sus parámetros y el estado inicial.
- System combina ambas entidades, construye las ecuaciones e integra la simulación.

No hay una capa pública independiente para el solver. Los integradores
simplécticos forman parte de classes/system y System los selecciona
internamente al ejecutar simulate(...).

## Instalación

El código Python está bajo src. Para instalarlo en modo editable:

~~~sh
python3 -m pip install -e .
~~~

Para instalar también las herramientas de notebooks:

~~~sh
python3 -m pip install -r requirements.txt
~~~

## Entidades del dominio

### Potential

Potential es la clase abstracta común. Sus implementaciones son:

- GridPotential: potencial definido sobre una malla, normalmente procedente de
  HDF5 o de un campo sintético muestreado; utiliza interpolación espacial.
- FourierPotential: potencial definido directamente mediante modos de Fourier;
  conserva la evaluación espectral del modelo Fourier.

Ambas implementaciones ofrecen la misma interfaz para evaluar el potencial y
sus derivadas. El potencial físico no conoce la trayectoria ni decide si debe
aplicarse un giro-promedio.

### Trajectory

Trajectory es independiente del potencial y de la integración. Las variantes
son:

- TrajectoryGC: centro guía, con estado físico organizado en bloques x, y.
- TrajectoryFC: ciclotrón completo, con estado físico organizado en bloques
  x, y, vx, vy.

Los identificadores internos son gc y fc. El valor histórico fo solo se acepta
como alias al leer configuraciones antiguas y se normaliza inmediatamente a fc.

### System

System recibe un Potential y una Trajectory:

~~~python
from classes import FourierPotential, TrajectoryGC, create_system
from workflows import initialize_trajectory

potential = FourierPotential(...)
trajectory = TrajectoryGC(...)
initialize_trajectory(trajectory, potential.grid)
system = create_system(potential, trajectory)

solution = system.simulate(
    t_span=(0.0, 20.0 * 3.141592653589793),
    step=0.1,
    n_save_step=11,
    method="BM4",
    check_energy=True,
)
~~~

create_system(...) devuelve SystemGC o SystemFC según la trayectoria. La
selección del método numérico queda encapsulada:

- SystemGC usa la extensión de espacio de fases para el sistema no separable.
- SystemFC usa los flujos adjuntos chi y chi_star de la composición simpléctica.

La solución conserva los tiempos, los estados físicos y, cuando se solicita,
la variable extendida y el error de energía.

## Ejecución desde terminal

Se mantienen dos superficies de entrada, diferenciadas por la fuente del
potencial, pero ambas construyen las mismas tres entidades.

Para un potencial de Fourier:

~~~sh
python3 run_fourier.py
~~~

Para un potencial de malla procedente de HDF5 o generado como mock:

~~~sh
python3 run_potential.py
~~~

Opciones comunes:

~~~sh
python3 run_fourier.py --config-group assay --config-version v_1
python3 run_potential.py --config-group test --config-version v_1
~~~

También puede seleccionarse un perfil concreto:

~~~sh
python3 run_fourier.py --version symplectic_grid
~~~

El logging se controla con SIM_LOG_LEVEL y SIM_LOG_FILE:

~~~sh
SIM_LOG_LEVEL=DEBUG SIM_LOG_FILE=logs/simulation.log python3 run_fourier.py
~~~

## Configuración

Los perfiles JSON se organizan por superficie, fuente, grupo y versión:

~~~text
conf/
  notebook/
    fourier/test/v_1.json
    potential/test/v_1.json
  terminal/
    fourier/test/v_1.json
    fourier/assay/v_1.json
    potential/test/v_1.json
    potential/assay/v_1.json
~~~

Todos los perfiles se normalizan conceptualmente en potencial, trayectoria,
solver y salida. La familia potential expresa esos bloques directamente:

~~~json
{
  "schema_version": 1,
  "active_version": "default",
  "versions": {
    "default": {
      "potential": {},
      "trajectory": {},
      "solver": {},
      "output": {}
    }
  }
}
~~~

La familia fourier conserva defaults, sweep y cases para generar lotes; el
loader extrae de cada caso los parámetros de FourierPotential y Trajectory.
En ambas familias, solver contiene exclusivamente los parámetros del
integrador:

- TimeStep: paso interno máximo.
- ode_solver: composición simpléctica, por ejemplo BM4.
- CheckEnergy: activa la comprobación de energía generalizada.

El nombre solver expresa una capacidad interna de System; no hace referencia a
una librería o paquete externo.

La documentación detallada está en:

- docs/conf/fourier.md
- docs/conf/potential.md
- docs/fourier_execution/README.md
- docs/potential_execution/README.md
- docs/system_architecture.md

## Modelo físico y proceso numérico

Para centro guía, SystemGC usa el potencial efectivo psi obtenido a partir del
potencial físico phi y del radio de Larmor de TrajectoryGC. Conserva:

~~~text
z = [x, y]
z_dot = [-dpsi/dy, dpsi/dx]
H = psi
~~~

Para ciclotrón completo, SystemFC usa el potencial físico sin giro-promedio y
conserva:

~~~text
z = [x, y, vx, vy]
velocity_scale = rho / (2 * abs(eta))
electric_scale = sign(eta) / rho
larmor_frequency = 1 / (2 * eta)
~~~

La reorganización cambia responsabilidades y nombres, no las fórmulas, los
coeficientes de los integradores ni el orden de los subflujos.

## Referencia

M. Stanzani, F. Arlotti, G. Ciraolo, X. Garbet y C. Chandre,
Transition to super-diffusive transport in turbulent plasmas,
arXiv:2309.02461.

Para más información: cristel.chandre@cnrs.fr
