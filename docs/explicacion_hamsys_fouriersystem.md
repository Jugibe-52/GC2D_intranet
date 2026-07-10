# HamSys y FourierSystem: explicación matemática y de implementación

> Nota redactada a partir de la conversación y del archivo `src/classes/fourier_system.py` del repositorio `GC2D_intranet`, rama `juan-0`.
>
> Archivo principal: <https://github.com/Jugibe-52/GC2D_intranet/blob/juan-0/src/classes/fourier_system.py>

---

## 1. Idea general

La idea central es:

```text
HamSys = clase general para sistemas hamiltonianos
FourierSystem = sistema hamiltoniano concreto con un potencial turbulento escrito en Fourier
```

`HamSys` viene de la librería `pyhamsys`. Es una clase pensada para integrar sistemas hamiltonianos, especialmente con métodos simplécticos. `FourierSystem` hereda de `HamSys` y le proporciona los ingredientes concretos de este problema:

- el campo vectorial `y_dot`,
- el Hamiltoniano `hamiltonian`,
- la derivada extendida `k_dot`,
- y, para el caso de órbitas completas, los mapas `chi` y `chi_star`.

En el código aparece:

```python
from pyhamsys import HamSys

class FourierSystem(HamSys):
    ...
```

Por tanto, `FourierSystem` es una especialización de `HamSys`.

---

## 2. Qué significa `HamSys`

Un sistema hamiltoniano se describe mediante un Hamiltoniano

$$
H(q,p),
$$

donde normalmente:

- $q$ son posiciones,
- $p$ son momentos.

Las ecuaciones de Hamilton son:

$$
\dot q = \frac{\partial H}{\partial p},
\qquad
\dot p = -\frac{\partial H}{\partial q}.
$$

Si juntamos las variables en un vector

$$
y = (q,p),
$$

entonces el campo vectorial hamiltoniano es:

$$
\dot y = X_H(y).
$$

En `pyhamsys`, la clase `HamSys` representa este tipo de sistema. Para que la librería pueda integrar un sistema, hay que definir métodos como:

```python
def y_dot(self, t, y):
    ...

def hamiltonian(self, t, y):
    ...
```

Conceptualmente:

```python
y_dot(t, y)       # devuelve el campo vectorial \dot y
hamiltonian(t, y) # devuelve el valor H(t,y)
```

`HamSys` no sabe por sí solo cuál es tu potencial ni cuál es tu problema físico. Solo da la estructura general. `FourierSystem` rellena esa estructura.

---

## 3. Qué significa `ndof`

`ndof` significa:

```text
number of degrees of freedom
```

es decir, número de grados de libertad.

Un sistema con un grado de libertad tiene un par canónico:

$$
(q,p).
$$

Un sistema con dos grados de libertad tiene dos pares canónicos:

$$
(q_1,q_2,p_1,p_2).
$$

En `pyhamsys`, la convención es que:

```python
ndof = 1
```

representa un sistema hamiltoniano autónomo con un grado de libertad:

$$
H = H(q,p).
$$

Mientras que:

```python
ndof = 1.5
```

representa un sistema con un grado de libertad pero con dependencia explícita del tiempo:

$$
H = H(t,q,p).
$$

La parte `.5` no significa que haya “medio grado de libertad físico”. Significa que el Hamiltoniano depende explícitamente del tiempo.

Por tanto:

$$
\boxed{\texttt{ndof=1.5} = \text{1 grado de libertad + dependencia explícita en } t}
$$

Y análogamente:

$$
\boxed{\texttt{ndof=2.5} = \text{2 grados de libertad + dependencia explícita en } t}
$$

En `FourierSystem` aparece:

```python
super().__init__(ndof=1.5 if dict_['traj_type']=='gc' else 2.5)
```

Esto significa:

- si `traj_type == 'gc'`, usa `ndof=1.5`;
- si `traj_type == 'fo'`, usa `ndof=2.5`.

Aquí:

- `gc` significa aproximadamente *guiding center*, centro guía;
- `fo` significa aproximadamente *full orbit*, órbita completa.

---

## 4. Por qué el caso `gc` tiene `ndof=1.5`

En el caso `gc`, el estado está formado por las coordenadas espaciales:

$$
(x,y).
$$

Aunque parecen dos variables de posición, en este modelo se pueden interpretar como un par canónico:

$$
q = x,
\qquad
p = y.
$$

Por tanto, hay un grado de libertad:

$$
(q,p) = (x,y).
$$

Pero el Hamiltoniano depende explícitamente del tiempo porque el potencial contiene términos del tipo:

$$
e^{i(nx+my-t)}.
$$

El término `-t` hace que:

$$
H = H(t,x,y).
$$

Por eso el código usa:

```python
ndof = 1.5
```

Es decir:

$$
\boxed{\text{centro guía} = 1 \text{ grado de libertad no autónomo}}
$$

---

## 5. Por qué el caso `fo` tiene `ndof=2.5`

En el caso `fo`, el estado contiene posición y velocidad:

$$
(x,y,v_x,v_y).
$$

Hay cuatro variables dinámicas, que corresponden a dos grados de libertad. Además, el potencial también depende explícitamente del tiempo. Por eso el código usa:

```python
ndof = 2.5
```

Es decir:

$$
\boxed{\text{órbita completa} = 2 \text{ grados de libertad no autónomos}}
$$

---

## 6. Construcción del potencial de Fourier

`FourierSystem` construye un potencial electrostático turbulento como suma de modos de Fourier.

El código crea una malla de modos:

$$
(n,m),
\qquad
0 \leq n,m \leq M.
$$

Después define coeficientes complejos:

$$
\phi_{n,m}
=
\frac{A}{(n^2+m^2)^{3/2}}
e^{i\theta_{n,m}},
$$

con fases aleatorias $\theta_{n,m}$.

En código, de forma esquemática:

```python
self.phases = 2 * pi * random((M, M))
self.nm = meshgrid(arange(M+1), arange(M+1))
self.phic[1:, 1:] = A / (n**2 + m**2)**1.5 * exp(1j * phases)
```

Luego elimina los modos que quedan fuera del círculo:

$$
\sqrt{n^2+m^2} > M.
$$

Es decir, el espectro de Fourier queda truncado.

---

## 7. El potencial `potential(t,y)`

El método `potential` calcula:

```python
def potential(self, t, y):
    exp_xy = exp(1j * (n*x + m*y - t))
    return sum(phic[n,m] * exp_xy).imag
```

Matemáticamente, define:

$$
\phi(t,x,y)
=
\operatorname{Im}
\sum_{n,m}
\phi_{n,m} e^{i(nx+my-t)}.
$$

Por tanto, `potential(t,y)` devuelve el valor del potencial en el punto $(x,y)$ y en el tiempo $t$.

---

## 8. Corrección FLR en el caso `gc`

Si `traj_type == 'gc'`, el código multiplica los coeficientes de Fourier por una función de Bessel:

```python
flr1_coeff = jv(0, rho * sqrt_nm)
self.phic *= flr1_coeff
```

Es decir:

$$
\phi_{n,m}
\longmapsto
J_0\left(\rho\sqrt{n^2+m^2}\right)\phi_{n,m}.
$$

Esto es una corrección de radio de Larmor finito. Su efecto es modificar los modos de Fourier según el parámetro $\rho$.

---

## 9. El campo `y_dot(t,y)` en el caso `gc`

El método `y_dot` calcula el campo vectorial hamiltoniano.

Para el centro guía, el Hamiltoniano es el potencial:

$$
H(t,x,y) = \phi(t,x,y).
$$

Las ecuaciones tienen la forma:

$$
\dot x = -\frac{\partial \phi}{\partial y},
\qquad
\dot y = \frac{\partial \phi}{\partial x}.
$$

En el código se prepara:

```python
self.fft_phi_ = [-m * phic, n * phic]
```

Eso corresponde a las derivadas del potencial en Fourier.

Si definimos:

$$
S(t,x,y) = \sum_{n,m}\phi_{n,m} e^{i(nx+my-t)},
$$

entonces:

$$
\phi(t,x,y) = \operatorname{Im} S(t,x,y).
$$

Derivando:

$$
\frac{\partial \phi}{\partial x}
=
\operatorname{Re}\sum_{n,m}n\phi_{n,m} e^{i(nx+my-t)},
$$

$$
\frac{\partial \phi}{\partial y}
=
\operatorname{Re}\sum_{n,m}m\phi_{n,m} e^{i(nx+my-t)}.
$$

Por tanto:

$$
\dot x = -\operatorname{Re}\sum_{n,m}m\phi_{n,m} e^{i(nx+my-t)},
$$

$$
\dot y = \operatorname{Re}\sum_{n,m}n\phi_{n,m} e^{i(nx+my-t)}.
$$

Eso es lo que devuelve `y_dot`.

---

## 10. Qué devuelve `hamiltonian(t,y)`

`hamiltonian(t,y)` devuelve el valor del Hamiltoniano:

$$
\boxed{\texttt{hamiltonian(t,y)} = H(t,y)}
$$

No devuelve una norma $L^2$, ni una derivada, ni una energía integrada en todo el dominio.

### Caso `gc`

En el caso centro guía:

```python
if self.traj_type == 'gc':
    return self.potential(t, y)
```

Por tanto:

$$
H_{gc}(t,x,y) = \phi(t,x,y).
$$

### Caso `fo`

En el caso órbita completa:

```python
return rho / (4 * abs(eta)) * (vx**2 + vy**2) \
       + potential(t, x, y) * sign(eta) / rho
```

Es decir:

$$
H_{fo}(t,x,y,v_x,v_y)
=
\frac{\rho}{4|\eta|}(v_x^2+v_y^2)
+
\frac{\operatorname{sign}(\eta)}{\rho}\phi(t,x,y).
$$

La primera parte es cinética. La segunda parte viene del potencial.

---

## 11. ¿La energía del Hamiltoniano debería conservarse?

Depende.

Si el Hamiltoniano es autónomo:

$$
H = H(q,p),
$$

entonces se conserva a lo largo de las trayectorias:

$$
\frac{d}{dt}H(q(t),p(t)) = 0.
$$

Pero si depende explícitamente del tiempo:

$$
H = H(t,q,p),
$$

entonces en general no se conserva.

La fórmula general es:

$$
\frac{d}{dt}H(t,q(t),p(t))
=
\frac{\partial H}{\partial t}
+
\frac{\partial H}{\partial q}\dot q
+
\frac{\partial H}{\partial p}\dot p.
$$

Usando las ecuaciones de Hamilton:

$$
\dot q = \frac{\partial H}{\partial p},
\qquad
\dot p = -\frac{\partial H}{\partial q},
$$

los dos últimos términos se cancelan:

$$
\frac{\partial H}{\partial q}\dot q
+
\frac{\partial H}{\partial p}\dot p
=
0.
$$

Por tanto:

$$
\boxed{
\frac{dH}{dt} = \frac{\partial H}{\partial t}
}
$$

Así que:

$$
\boxed{
H \text{ se conserva si } \partial_t H = 0.
}
$$

En `FourierSystem`, el potencial contiene:

$$
e^{i(nx+my-t)},
$$

por tanto:

$$
\partial_t H \neq 0
$$

en general. Así que el Hamiltoniano instantáneo no tiene por qué ser constante.

---

## 12. Entonces, ¿qué se conserva?

Aunque $H(t,y)$ no tenga por qué conservarse, el sistema sigue siendo hamiltoniano. Lo que se conserva estructuralmente es la geometría simpléctica.

Además, un sistema no autónomo puede convertirse en autónomo añadiendo una variable conjugada al tiempo.

Se introduce una variable $k$ y se define el Hamiltoniano extendido:

$$
K(t,y,k) = H(t,y) + k.
$$

Entonces:

$$
\dot t = \frac{\partial K}{\partial k} = 1,
$$

y

$$
\dot k = -\frac{\partial K}{\partial t}
= -\frac{\partial H}{\partial t}.
$$

Por tanto:

$$
\boxed{
\dot k = -\partial_t H
}
$$

Y entonces:

$$
\frac{d}{dt}(H+k)
=
\frac{dH}{dt}+\dot k
=
\partial_t H - \partial_t H
=0.
$$

Así que el Hamiltoniano extendido:

$$
\boxed{
K = H+k
}
$$

sí se conserva idealmente.

---

## 13. Qué hace `k_dot(t,y)`

`k_dot(t,y)` no mueve la partícula. Sirve para actualizar la variable auxiliar $k$.

En un sistema dependiente del tiempo:

$$
\boxed{
\texttt{k\_dot}(t,y) = -\partial_t H(t,y)
}
$$

En el caso `gc`:

$$
H(t,x,y)=\phi(t,x,y).
$$

Recordemos:

$$
\phi(t,x,y)
=
\operatorname{Im}
\sum_{n,m}\phi_{n,m}e^{i(nx+my-t)}.
$$

Definimos:

$$
S(t,x,y)
=
\sum_{n,m}\phi_{n,m}e^{i(nx+my-t)}.
$$

Entonces:

$$
\phi = \operatorname{Im}S.
$$

Como:

$$
\partial_t S = -iS,
$$

si:

$$
S=a+ib,
$$

entonces:

$$
-iS = b-ia.
$$

Por tanto:

$$
\partial_t \phi
=
\operatorname{Im}(-iS)
=
-a
=
-\operatorname{Re}S.
$$

Luego:

$$
-\partial_t\phi = \operatorname{Re}S.
$$

Y el código de `k_dot` hace justamente:

```python
return sum(phic[n,m] * exp(i*(n*x + m*y - t))).real
```

Es decir:

$$
\boxed{
\texttt{k\_dot}(t,y)=\operatorname{Re}S(t,x,y)=-\partial_t\phi(t,x,y)
}
$$

En el caso `fo`, como el Hamiltoniano contiene el potencial multiplicado por

$$
\frac{\operatorname{sign}(\eta)}{\rho},
$$

la actualización de $k$ lleva ese factor:

$$
\dot k
=
\frac{\operatorname{sign}(\eta)}{\rho}(-\partial_t\phi).
$$

Por eso en `chi` y `chi_star` aparece:

```python
k += h * sign(eta) / rho * self.k_dot(...)
```

---

## 14. Relación entre `hamiltonian` y `k_dot`

Es importante no confundirlos.

`hamiltonian(t,y)` devuelve:

$$
H(t,y).
$$

`k_dot(t,y)` devuelve:

$$
-\partial_t H(t,y)
$$

o la parte correspondiente en la normalización usada por el código.

Por tanto:

```text
hamiltonian(t,y) = valor instantáneo de H
k_dot(t,y)       = compensación temporal para la variable extendida k
```

No son lo mismo.

Tampoco `hamiltonian(t,y)` devuelve:

$$
\|\partial_t H\|_{L^2}.
$$

Una norma $L^2$ de $\partial_t H$ sería algo global, por ejemplo:

$$
\|\partial_t H\|_{L^2}^2
=
\int |\partial_t H(t,x,y)|^2\,dx\,dy.
$$

Eso no se calcula en `hamiltonian(t,y)`.

---

## 15. Condiciones iniciales

El método `initial_conditions` construye el vector inicial.

Tiene tres opciones principales:

```python
type='random'
type='fixed'
type='selected'
```

### `random`

Genera posiciones aleatorias en el toro:

$$
[0,2\pi]\times[0,2\pi].
$$

### `fixed`

Construye una malla regular en el toro. Si `Ntraj` no es un cuadrado perfecto, lo cambia al cuadrado perfecto inferior:

```python
self.Ntraj = int(sqrt(self.Ntraj))**2
```

Esto permite organizar las partículas en una red cuadrada.

### `selected`

Usa posiciones dadas manualmente:

```python
x0 = self.x0
y0 = self.y0
```

Si se usa `fo`, añade velocidades iniciales:

$$
v_x = \cos\varphi,
\qquad
v_y = \sin\varphi,
$$

con $\varphi$ aleatorio.

Si `CheckEnergy` está activado, también añade una variable extra $k$ inicializada a cero.

---

## 16. Qué hacen `chi` y `chi_star`

Los métodos `chi` y `chi_star` se usan para integradores simplécticos por splitting.

La idea de un splitting es separar el flujo en partes más simples. En vez de integrar todo el sistema de golpe, se compone una sucesión de flujos parciales.

En `FourierSystem`, `chi` y `chi_star` aparecen en el caso `fo`, órbita completa.

### `chi`

Hace esencialmente:

1. una rotación libre de la velocidad;
2. una actualización de posición asociada a esa rotación;
3. un `kick` debido al potencial;
4. si `CheckEnergy` está activado, actualiza también $k$.

El código usa números complejos para representar:

$$
x+iy,
\qquad
v_x+iv_y.
$$

La rotación aparece mediante:

$$
e^{-ih/(2\eta)}.
$$

En código:

```python
exp_ = exp(-1j * h / (2 * eta))
```

Esto rota el vector velocidad complejo:

$$
v_x+iv_y.
$$

Después calcula el efecto del potencial usando `y_dot`.

### `chi_star`

`chi_star` hace una composición parecida, pero en orden inverso:

1. primero aplica el `kick` del potencial;
2. después aplica la rotación libre.

Por eso puede verse como una versión adjunta de `chi`. Esta pareja `chi`, `chi_star` permite construir integradores simétricos.

---

## 17. Cómo se integra en el proyecto

En `src/workflows/params.py`, la función `make_system` construye el sistema:

```python
def make_system(params):
    params = to_symp_params(params)
    return FourierSystem(params)
```

Es decir, el flujo es:

```text
parámetros / JSON
      ↓
to_symp_params
      ↓
FourierSystem(params)
      ↓
initial_conditions()
      ↓
integración
```

En `src/workflows/integration.py`, si el sistema es `gc`, usa:

```python
solve_ivp_sympext(system, ...)
```

Si el sistema es `fo`, usa:

```python
solve_ivp_symp(system.chi, system.chi_star, ...)
```

Así que hay dos caminos:

```text
gc → solve_ivp_sympext → usa y_dot, hamiltonian, k_dot
fo → solve_ivp_symp    → usa chi y chi_star
```

---

## 18. Resumen conceptual corto

El sistema `gc` es:

$$
H_{gc}(t,x,y)=\phi(t,x,y),
$$

con:

$$
\dot x = -\partial_y\phi,
\qquad
\dot y = \partial_x\phi.
$$

Como $\phi$ depende de $t$, el Hamiltoniano no se conserva necesariamente:

$$
\frac{dH}{dt}=\partial_t H.
$$

Para controlar la energía extendida se introduce $k$:

$$
K = H+k,
\qquad
\dot k=-\partial_t H.
$$

Eso es lo que implementa `k_dot`.

El sistema `fo` tiene Hamiltoniano:

$$
H_{fo}(t,x,y,v_x,v_y)
=
\frac{\rho}{4|\eta|}(v_x^2+v_y^2)
+
\frac{\operatorname{sign}(\eta)}{\rho}\phi(t,x,y).
$$

Y se integra usando mapas simplécticos `chi` y `chi_star`.

---

## 19. Tabla resumen

| Elemento | Significado |
|---|---|
| `HamSys` | Clase base para sistemas hamiltonianos |
| `FourierSystem` | Sistema hamiltoniano concreto con potencial de Fourier |
| `ndof=1.5` | Un grado de libertad con dependencia explícita del tiempo |
| `ndof=2.5` | Dos grados de libertad con dependencia explícita del tiempo |
| `potential(t,y)` | Calcula $\phi(t,x,y)$ |
| `y_dot(t,y)` | Calcula el campo hamiltoniano $\dot y$ |
| `hamiltonian(t,y)` | Devuelve $H(t,y)$ |
| `k_dot(t,y)` | Devuelve $-\partial_t H$, o la parte correspondiente del potencial |
| `chi` | Flujo parcial usado por el integrador simpléctico |
| `chi_star` | Flujo parcial adjunto, con orden inverso |
| `CheckEnergy` | Activa el seguimiento de la energía extendida |

---

## 20. Frase para recordar

La frase más importante es:

$$
\boxed{
\texttt{hamiltonian(t,y)} \text{ evalúa } H(t,y),
\quad
\texttt{k\_dot(t,y)} \text{ evalúa } -\partial_t H(t,y).
}
$$

Y:

$$
\boxed{
\texttt{ndof=1.5} \text{ no significa medio grado de libertad; significa dependencia explícita del tiempo.}
}
$$

---

## 21. Fuentes consultadas

- `fourier_system.py`: <https://github.com/Jugibe-52/GC2D_intranet/blob/juan-0/src/classes/fourier_system.py>
- `integration.py`: <https://github.com/Jugibe-52/GC2D_intranet/blob/juan-0/src/workflows/integration.py>
- `params.py`: <https://github.com/Jugibe-52/GC2D_intranet/blob/juan-0/src/workflows/params.py>
- `pyhamsys` en PyPI: <https://pypi.org/project/pyhamsys/0.2/>
- Repositorio de `pyhamsys`: <https://github.com/cchandre/pyhamsys>
