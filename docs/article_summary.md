# Resumen matemático-físico del artículo

## 1. Objetivo

El artículo estudia el movimiento de partículas cargadas en un potencial electrostático turbulento usando teoría de centro guía. El objetivo matemático principal es explicar una transición de transporte difusivo a transporte superdifusivo/casi balístico cuando aumenta el radio de Larmor `rho`.

El mecanismo identificado es la aparición de **toros invariantes twistless** en la dinámica reducida de centro guía. Estos toros actúan como barreras de transporte en una dirección y, al mismo tiempo, organizan capas de transporte casi balístico en la dirección transversal.

## 2. Modelo físico de partida

Se considera:

- campo magnético constante y uniforme,
- campo magnético dirigido según `z`,
- potencial electrostático turbulento independiente de la coordenada longitudinal `z`,
- dinámica transversal en el plano `(x,y)`.

La ecuación dimensional de Newton-Lorentz es

```math
m \frac{d v}{dt} = q\left(-\nabla \Phi(x,t) + v \times B\right),
```

con

```math
x=(x,y,z), \qquad v=(v_x,v_y,v_z), \qquad B = B\,\hat z.
```

Como `Phi` no depende de `z`, la dinámica longitudinal se desacopla y el problema se reduce al movimiento perpendicular al campo magnético.

## 3. Reducción a variables adimensionales

El artículo trabaja en variables adimensionales. Las escalas espaciales y temporales se escogen de modo que la longitud y el periodo característicos del potencial turbulento quedan normalizados a `2*pi`.

Los parámetros adimensionales relevantes son:

```math
A, \qquad \rho, \qquad \eta.
```

Interpretación:

- `A`: amplitud efectiva del potencial electrostático dividido por `B`.
- `rho`: radio de Larmor adimensional.
- `eta`: parámetro asociado a la frecuencia de Larmor y a la escala temporal del potencial.

En la mayor parte del artículo se fija

```math
M=25, \qquad A=0.7,
```

y se estudia cómo cambia la dinámica al variar `(rho, eta)`.

## 4. Dos niveles de dinámica

El artículo distingue dos sistemas:

### 4.1 Órbitas completas

Sistema en variables `(x,y,v_x,v_y)`:

```math
\dot x = \frac{\rho}{2|\eta|} v,
```

```math
\dot v = -\frac{\operatorname{sgn}(\eta)}{\rho}\nabla \phi(x,y,t)
          + \frac{1}{2\eta} v \times \hat z.
```

Este sistema conserva la girodinámica rápida y tiene más dimensión. Es más costoso de simular.

### 4.2 Dinámica de centro guía

Se elimina la rotación rápida de Larmor y se obtiene una dinámica efectiva en posiciones de centro guía `(X,Y)`. Esta dinámica está gobernada por un potencial efectivo `psi`:

```math
\dot X = -\nabla \psi(X,Y,t) \times \hat z.
```

En componentes:

```math
\dot X = -\partial_Y \psi(X,Y,t),
```

```math
\dot Y = \partial_X \psi(X,Y,t).
```

Esta es la ecuación fundamental que conviene implementar primero.

## 5. Potencial electrostático turbulento

El potencial adimensional usado para las simulaciones es

```math
\phi(x,y,t)
= A \sum_{(n,m)\in I_M}
\frac{1}{(n^2+m^2)^{3/2}}
\sin(n x + m y + \varphi_{nm} - t),
```

con

```math
I_M = \{(n,m): 1\leq n,m\leq M,\; n^2+m^2 < M^2\}.
```

Las fases `varphi_nm` son aleatorias uniformes en `[0,2*pi)`. En las simulaciones del artículo se usa un conjunto fijo de fases.

## 6. Potencial efectivo de centro guía

El potencial efectivo de centro guía es

```math
\psi = J_0[\phi] - \eta\left(J_1[\phi^2] - 2J_0[\phi]J_1[\phi]\right).
```

Aquí `J_0` es el operador de gyroaverage:

```math
J_0[\phi](X,Y,t;\rho)
= \frac{1}{2\pi}\int_0^{2\pi}
\phi(X+\rho\cos\theta,\,Y-\rho\sin\theta,\,t)\,d\theta,
```

y

```math
J_1[\phi] = \rho^{-1}\frac{\partial}{\partial \rho}J_0[\phi].
```

El término de primer orden `J_0[phi]` depende de `rho`, pero no de `eta`; el término de segundo orden introduce la dependencia en `eta`.

## 7. Magnitudes diagnósticas

Las magnitudes principales son:

1. **Secciones de Poincaré**: puntos `(X(2*pi*n),Y(2*pi*n))`.
2. **Mean-square displacement** o MSD.
3. **Exponente de transporte** `b` en la ley aproximada

```math
\operatorname{MSD}(t) \approx (a t)^b.
```

Interpretación:

- `b ≈ 1`: difusión normal.
- `b > 1`: superdifusión.
- `b ≈ 2`: transporte casi balístico.

4. **Números de rotación** para detectar toros invariantes twistless.

## 8. Resultados matemáticos principales a reproducir

### Caso difusivo

Para

```math
A=0.7, \qquad \eta=0, \qquad \rho=0,
```

el artículo muestra coexistencia de trayectorias atrapadas y caóticas. Las trayectorias caóticas producen transporte difusivo normal.

### Caso superdifusivo

Para

```math
A=0.7, \qquad \eta=0.14, \qquad \rho=0.3,
```

aparecen capas alargadas de transporte casi balístico. La sección de Poincaré muestra estructuras regulares asociadas a toros invariantes twistless.

### Barrido en parámetros

Al variar `(rho, eta)`, el artículo encuentra una región grande de comportamiento superdifusivo aproximadamente para

```math
\rho \gtrsim 0.2, \qquad |\eta| \lesssim 0.2.
```

La transición aparece alrededor de

```math
\rho \approx 0.2 - 0.25.
```

## 9. Validación con órbitas completas

El artículo compara la dinámica de centro guía con la dinámica de órbitas completas. Para reconstruir las coordenadas de centro guía desde una órbita completa en variables reescaladas se usa

```math
\rho_v = \rho\sqrt{v_x^2+v_y^2},
```

```math
\theta = \pi + \operatorname{atan2}(v_x,v_y),
```

```math
X = x - \rho_v \cos\theta,
```

```math
Y = y + \rho_v \sin\theta.
```

La comparación confirma que las regiones superdifusivas observadas en el modelo de centro guía siguen apareciendo al mirar las órbitas completas.

## 10. Conclusión matemática útil para el proyecto

El objeto central que debes programar no es inicialmente la órbita completa, sino la aplicación no autónoma y periódica en tiempo generada por

```math
\dot X = -\partial_Y \psi(X,Y,t),
```

```math
\dot Y = \partial_X \psi(X,Y,t),
```

con `psi` calculado a partir del potencial turbulento `phi` mediante gyroaveraging y términos de segundo orden.

