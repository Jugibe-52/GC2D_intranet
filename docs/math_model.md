# Modelo matemático extraído del artículo

## 1. Variables

### Variables físicas dimensionales

```math
x=(x,y,z), \qquad v=(v_x,v_y,v_z).
```

- `x`: posición de la partícula.
- `v`: velocidad.
- `m`: masa.
- `q`: carga.
- `B = B \hat z`: campo magnético constante y uniforme.
- `Phi(x,t)`: potencial electrostático dimensional.

### Variables adimensionales

El artículo trabaja finalmente con variables adimensionales y omite el sombrero. Las variables relevantes son:

```math
x=(x,y), \qquad v=(v_x,v_y), \qquad X=(X,Y), \qquad t.
```

- `(x,y)`: posición de la partícula en el plano perpendicular.
- `v`: velocidad perpendicular reescalada.
- `(X,Y)`: posición del centro guía.
- `t`: tiempo adimensional.

## 2. Parámetros

```math
A, \qquad \rho, \qquad \eta, \qquad M.
```

- `A`: amplitud adimensional del potencial.
- `rho`: radio de Larmor adimensional.
- `eta`: parámetro asociado a la escala temporal de la girodinámica.
- `M`: truncación de modos de Fourier.

Valores usados frecuentemente:

```math
M=25, \qquad A=0.7.
```

Casos de referencia:

```math
(\rho,\eta)=(0,0)
```

para difusión normal, y

```math
(\rho,\eta)=(0.3,0.14)
```

para superdifusión/casi balística.

## 3. Ecuación dimensional de partida

La dinámica de la partícula cargada se modela mediante

```math
m\frac{dv}{dt}
= q\left(-\nabla \Phi(x,t) + v\times B\right).
```

Se asume que `Phi` no depende de `z`, por lo que la dinámica longitudinal se desacopla. La dinámica de interés es la transversal.

## 4. Órbitas completas adimensionales

En el plano perpendicular, las ecuaciones reescaladas son

```math
\dot x = \frac{\rho}{2|\eta|} v,
```

```math
\dot v = -\frac{\operatorname{sgn}(\eta)}{\rho}\nabla \phi(x,y,t)
          + \frac{1}{2\eta}v\times \hat z.
```

Si

```math
v=(v_x,v_y,0), \qquad \hat z=(0,0,1),
```

entonces

```math
v\times \hat z = (v_y,-v_x,0).
```

Por tanto, en componentes:

```math
\dot x = \frac{\rho}{2|\eta|} v_x,
```

```math
\dot y = \frac{\rho}{2|\eta|} v_y,
```

```math
\dot v_x = -\frac{\operatorname{sgn}(\eta)}{\rho}\partial_x\phi(x,y,t)
           + \frac{1}{2\eta}v_y,
```

```math
\dot v_y = -\frac{\operatorname{sgn}(\eta)}{\rho}\partial_y\phi(x,y,t)
           - \frac{1}{2\eta}v_x.
```

Nota: esta formulación tiene singularidades formales cuando `rho=0` o `eta=0`; esos límites deben tratarse en la dinámica reducida o como límites teóricos, no sustituyendo directamente en el sistema completo.

## 5. Potencial electrostático turbulento

El potencial adimensional es

```math
\phi(x,y,t)
= A\sum_{(n,m)\in I_M}
\frac{1}{(n^2+m^2)^{3/2}}
\sin(n x + m y + \varphi_{nm} - t),
```

con conjunto de modos

```math
I_M = \{(n,m)\in\{1,\ldots,M\}^2 : n^2+m^2<M^2\}.
```

Las fases son

```math
\varphi_{nm}\sim \operatorname{Unif}(0,2\pi).
```

Para reproducibilidad, las fases deben fijarse con una semilla o guardarse en disco.

### Gradiente de `phi`

Dado

```math
\phi(x,y,t)
= A\sum_{(n,m)\in I_M} c_{nm}\sin(n x + m y + \varphi_{nm}-t),
```

con

```math
c_{nm}=\frac{1}{(n^2+m^2)^{3/2}},
```

se tiene

```math
\partial_x\phi(x,y,t)
= A\sum_{(n,m)\in I_M} n c_{nm}\cos(n x + m y + \varphi_{nm}-t),
```

```math
\partial_y\phi(x,y,t)
= A\sum_{(n,m)\in I_M} m c_{nm}\cos(n x + m y + \varphi_{nm}-t).
```

Estas fórmulas permiten implementar una versión directa sin FFT para tests pequeños.

## 6. Cambio a centro guía

A primer orden, la posición de partícula `x` y la posición de centro guía `X` se relacionan por

```math
x = X + \hat z\times v\,\rho\,\operatorname{sgn}(\eta).
```

La idea matemática es separar:

- giro rápido de Larmor,
- deriva lenta transversal.

El resultado es un sistema reducido de dimensión menor.

## 7. Operador de gyroaverage `J0`

El operador de gyroaverage aplicado a un potencial `phi` es

```math
J_0[\phi](X,Y,t;\rho)
= \frac{1}{2\pi}\int_0^{2\pi}
\phi(X+\rho\cos\theta,\,Y-\rho\sin\theta,\,t)\,d\theta.
```

Interpretación: media del potencial sobre la circunferencia de Larmor de radio `rho` alrededor del centro guía `(X,Y)`.

## 8. Operador `J1`

El artículo define

```math
J_1[\phi] = \rho^{-1}\frac{\partial}{\partial\rho}J_0[\phi].
```

Cuidado: el símbolo `J_1` también aparece como función de Bessel de primer orden en el apéndice. En código conviene distinguir:

```text
J1_operator[f]
```

frente a

```text
besselj1(s)
```

## 9. Potencial efectivo `psi`

El potencial efectivo de centro guía a segundo orden es

```math
\psi(X,Y,t)
= J_0[\phi](X,Y,t)
- \eta\left(J_1[\phi^2](X,Y,t)
- 2J_0[\phi](X,Y,t)J_1[\phi](X,Y,t)\right).
```

Este es el objeto matemático central de la implementación.

A primer orden:

```math
\psi \approx J_0[\phi].
```

A segundo orden aparece `eta`, lo cual permite estudiar la dependencia con este parámetro.

## 10. Ecuaciones de centro guía

La dinámica efectiva está dada por

```math
\dot X = -\nabla\psi(X,Y,t)\times \hat z.
```

En componentes:

```math
\dot X = -\partial_Y \psi(X,Y,t),
```

```math
\dot Y = \partial_X \psi(X,Y,t).
```

Es un sistema Hamiltoniano no autónomo de un grado y medio de libertad.

## 11. Formulación de Fourier del apéndice A

El potencial se escribe como

```math
\phi(x,y,t)=\operatorname{Re}\left[\phi_c(x,y)e^{-it}\right],
```

con

```math
\phi_c(x,y)=\sum_{(n,m)\in I_M}K_{nm}e^{i(nx+my)},
```

```math
K_{nm}=\frac{A e^{i\varphi_{nm}}}{(n^2+m^2)^{3/2}}.
```

Entonces

```math
J_0[\phi_c](X,Y)
=\sum_{(n,m)\in I_M}
K_{nm}J_0^{\mathrm{Bessel}}\left(\rho\sqrt{n^2+m^2}\right)e^{i(nX+mY)}.
```

Para un potencial complejo genérico

```math
\varphi_c(X,Y)=\sum L_{nm}e^{i(nX+mY)},
```

se tiene

```math
J_1[\varphi_c]
= -\sum_{(n,m)\in I_M}
L_{nm}\frac{\sqrt{n^2+m^2}}{\rho}
J_1^{\mathrm{Bessel}}\left(\rho\sqrt{n^2+m^2}\right)e^{i(nX+mY)}.
```

## 12. Potencial efectivo en forma compleja

El artículo introduce

```math
\phi_2^{(0)}
= J_0[\phi_c]J_1[\phi_c^*]
- \frac{1}{2}J_1[|\phi_c|^2],
```

```math
\phi_2^{(2)}
= J_0[\phi_c]J_1[\phi_c]
- \frac{1}{2}J_1[\phi_c^2].
```

Entonces

```math
\psi
= \operatorname{Re}\left[
J_0[\phi_c]e^{-it}
+ \eta\phi_2^{(0)}
- \eta\phi_2^{(2)}e^{-2it}
\right].
```

Esta forma es la más útil para implementar el potencial efectivo mediante FFT.

## 13. Mean-square displacement

Para un conjunto de `N_traj` trayectorias de centro guía, muestreadas en tiempos

```math
 t_l = 2\pi l, \qquad l=0,\ldots,K-1,
```

el MSD para

```math
 t=2\pi k
```

se define como

```math
\operatorname{MSD}(2\pi k)
= \frac{1}{N_{traj}}\sum_{n=1}^{N_{traj}}
\frac{1}{K-k}\sum_{l=0}^{K-k-1}
\left\|X_n(2\pi(k+l))-X_n(2\pi l)\right\|^2.
```

Aquí `X_n` representa la posición bidimensional del centro guía de la trayectoria `n`.

Para estudiar transporte, conviene usar posiciones no envueltas en `R^2`, no solo posiciones módulo `2*pi`.

## 14. Exponente de transporte

Se ajusta una ley de potencia

```math
\operatorname{MSD}(t)\approx (a t)^b.
```

Tomando logaritmos:

```math
\log\operatorname{MSD}(t) \approx b\log t + b\log a.
```

El exponente `b` clasifica el transporte:

```text
b < 1    subdifusión
b ≈ 1    difusión normal
1 < b < 2 superdifusión
b ≈ 2    transporte balístico
```

## 15. Números de rotación

Para detectar toros invariantes twistless se calcula el número de rotación en la dirección `X`:

```math
r(Y_0)
= \lim_{S\to\infty}\frac{1}{S}\sum_{n=0}^{S-1}(X_{n+1}-X_n)
= \lim_{S\to\infty}\frac{X_S-X_0}{S},
```

con

```math
X_n=X(2\pi n).
```

Para mejorar la convergencia, se usa una media de Birkhoff ponderada:

```math
r(Y_0)
= \lim_{S\to\infty}\frac{1}{C_S}\sum_{n=1}^{S-1}
\omega\left(\frac{n}{S}\right)(X_{n+1}-X_n),
```

```math
C_S=\sum_{n=1}^{S-1}\omega\left(\frac{n}{S}\right),
```

```math
\omega(t)=\exp\left[-\frac{1}{t(1-t)}\right].
```

Un toro twistless se detecta cuando la curva `r(Y0)` presenta un extremo local.

## 16. Energía extendida para validación

Al autonomizar el sistema, se conserva una energía extendida. Para los centros guía:

```math
h = k + J_0[\phi]
-\eta\left(J_1[\phi^2]-2J_0[\phi]J_1[\phi]\right),
```

es decir,

```math
h = k + \psi.
```

Aquí `k` es la variable canónicamente conjugada al tiempo. El artículo usa el error relativo medio

```math
\left\langle\frac{h-h_0}{h_0}\right\rangle
```

para validar la precisión de la discretización temporal y espacial.

