# Proceso numérico de System

La reorganización del dominio no modifica las ecuaciones, los coeficientes de
composición ni el orden de los subflujos. El motor se encapsula bajo
classes/system y se invoca mediante system.simulate(...).

## Contrato común

Todo System ofrece:

~~~text
y_dot(t, y)       campo vectorial
hamiltonian(t, y) energía instantánea
k_dot(t, y)       derivada del momento conjugado al tiempo
simulate(...)     integración y postproceso
~~~

Los sistemas dependen explícitamente del tiempo. SystemGC tiene un grado de
libertad físico y SystemFC tiene dos.

## Estado por bloques

Para N trayectorias GC:

~~~text
y = [x_1 ... x_N, y_1 ... y_N]
~~~

Para N trayectorias FC:

~~~text
y = [
  x_1 ... x_N,
  y_1 ... y_N,
  vx_1 ... vx_N,
  vy_1 ... vy_N
]
~~~

Las variables auxiliares del algoritmo nunca cambian este contrato público.
Cuando se añade k para comprobar energía, se retira antes de exponer el estado
físico final.

## Potencial físico y efectivo

phi denota el Potential recibido por System. SystemGC construye psi mediante el
giro-promedio asociado a rho:

~~~text
psi_k = J0(rho * |k|) * phi_k
~~~

SystemFC usa phi directamente. Esta distinción debe mantenerse tanto para
GridPotential como para FourierPotential.

## Dinámica GC

Con:

~~~text
E_x = -dpsi/dx
E_y = -dpsi/dy
~~~

las ecuaciones son:

~~~text
x_dot = E_y
y_dot = -E_x
~~~

El Hamiltoniano y la variable extendida son:

~~~text
H(t, x, y) = psi(t, x, y)
k_dot = -dH/dt
~~~

### Extensión de espacio de fases

El integrador de gc_solver.py duplica internamente el estado, alterna
evaluaciones del campo en las dos copias y aplica el acoplamiento armónico.
Después promedia ambas copias para producir la solución física.

La integración:

- limita cada paso interno por step;
- usa la composición indicada por method;
- guarda exactamente n_save_step tiempos uniformes;
- incluye los extremos de t_span;
- alcanza siempre el extremo final.

## Dinámica FC

TrajectoryFC proporciona:

~~~text
v_scale = rho / (2 * abs(eta))
e_scale = sign(eta) / rho
omega_L = 1 / (2 * eta)
~~~

Con el campo físico:

~~~text
E_x = -dphi/dx
E_y = -dphi/dy
~~~

el campo vectorial completo es:

~~~text
x_dot  = vx * v_scale
y_dot  = vy * v_scale
vx_dot = E_x * e_scale + vy * omega_L
vy_dot = E_y * e_scale - vx * omega_L
~~~

El Hamiltoniano es:

~~~text
H = rho/(4*abs(eta)) * (vx^2 + vy^2)
    + sign(eta)/rho * phi(t, x, y)
~~~

y:

~~~text
k_dot = -e_scale * dphi/dt
~~~

## Flujos FC

SystemFC implementa dos mapas adjuntos:

- chi aplica primero rotación y deriva espacial, después el impulso eléctrico;
- chi_star aplica primero el impulso eléctrico, después rotación y deriva.

El motor de fc_solver.py alterna ambos mapas con los coeficientes del método.
La simetría del orden es parte del algoritmo y no debe alterarse al reorganizar
archivos.

Si check_energy está activo:

1. simulate añade k=0 al estado interno;
2. chi y chi_star actualizan k en el subflujo correspondiente;
3. se calcula H+k;
4. el postproceso retira k de solution.y y lo conserva en solution.k.

## Métodos de composición

El motor conserva las familias:

~~~text
Verlet, FR, Yos6, YoN, M2, M4,
EFRL, PEFRL, VEFRL,
BM4, BM6,
RKN4b, RKN6a, RKN6b,
ABA104, ABA864, ABA1064
~~~

method selecciona una tabla de coeficientes. System no cambia esos coeficientes
según el tipo de Potential.

## Muestreo y solución

step es el máximo paso interno, no la separación obligatoria entre muestras.
Para cada intervalo entre tiempos guardados se elige el mínimo número entero de
pasos que respeta ese máximo.

Solution expone metadatos como:

~~~text
requested_step
min_step
max_step
n_steps
~~~

además de t, y, k y err cuando correspondan.

## Conservación de energía

Para sistemas dependientes del tiempo se comprueba:

~~~text
K(t) = H(t, y(t)) + k(t)
~~~

El error informado es la máxima desviación absoluta respecto de K(0), evaluada
para cada trayectoria. Desactivar check_energy evita tanto la variable
extendida como este cálculo.

## Invariantes de regresión

Una modificación del motor debe comprobar al menos:

1. misma forma y orden de bloques del estado;
2. mismos tiempos guardados;
3. mismos coeficientes de composición;
4. mismo orden de chi y chi_star;
5. mismo tratamiento de phi frente a psi;
6. misma convención espectral de FourierPotential;
7. misma evolución y retirada de k;
8. resultados numéricos equivalentes para casos GC y FC de ambas clases de
   Potential.
