# Especificación computacional del modelo numérico

## 1. Objetivo computacional

Implementar la dinámica de centro guía y, en una segunda fase, la dinámica de órbitas completas del artículo.

Prioridad:

```text
1. Potencial turbulento phi.
2. Potencial efectivo psi.
3. Dinámica de centro guía.
4. Secciones de Poincaré.
5. MSD y exponente b.
6. Números de rotación y detección de toros twistless.
7. Comparación con órbitas completas.
```

## 2. Representación del dominio

El dominio espacial es el toro bidimensional:

```math
(X,Y)\in (\mathbb R/(2\pi\mathbb Z))^2.
```

Para visualizar transporte en `R^2`, se deben guardar dos versiones de la posición:

```text
position_mod      # posición reducida módulo 2*pi para Poincaré
position_unwrapped # posición no envuelta para MSD/transporte
```

## 3. Parámetros de simulación

Objeto `Params` sugerido:

```python
@dataclass
class Params:
    A: float = 0.7
    rho: float = 0.3
    eta: float = 0.14
    M: int = 25
    grid_N: int = 4096
    dt: float = 0.005
    seed: int = 1234
```

Para desarrollo inicial no uses `grid_N=4096`: empieza con `128`, `256` o `512`. El valor `4096` aparece como referencia de precisión en el artículo, pero es caro en memoria.

## 4. Fases aleatorias

Función sugerida:

```python
def generate_phases(M: int, seed: int) -> dict[tuple[int, int], float]:
    ...
```

Requisitos:

- fases uniformes en `[0, 2*pi)`,
- mismas fases para todas las simulaciones de un barrido,
- posibilidad de guardarlas/cargarlas.

## 5. Conjunto de modos

```python
def mode_set(M: int) -> list[tuple[int, int]]:
    return [(n, m) for n in range(1, M+1)
                   for m in range(1, M+1)
                   if n*n + m*m < M*M]
```

## 6. Implementación directa de `phi`

Para tests y prototipos:

```python
def phi_direct(x, y, t, params, phases):
    total = 0.0
    for n, m in mode_set(params.M):
        c = 1.0 / (n*n + m*m)**1.5
        total += c * np.sin(n*x + m*y + phases[n, m] - t)
    return params.A * total
```

Gradiente directo:

```python
def grad_phi_direct(x, y, t, params, phases):
    dx = 0.0
    dy = 0.0
    for n, m in mode_set(params.M):
        c = 1.0 / (n*n + m*m)**1.5
        angle = n*x + m*y + phases[n, m] - t
        dx += n * c * np.cos(angle)
        dy += m * c * np.cos(angle)
    return params.A * np.array([dx, dy])
```

Esta versión es lenta para muchas trayectorias, pero excelente para validar la versión FFT.

## 7. Implementación FFT del potencial efectivo

El artículo recomienda calcular campos estáticos en una malla `N x N` del toro usando FFT 2D antes de integrar las trayectorias.

Campos estáticos a preparar:

```text
phi_c
J0_phi_c
phi2_0
phi2_2
d/dX de esos campos
d/dY de esos campos
```

La forma compleja del potencial efectivo es

```math
\psi(X,Y,t)
= \operatorname{Re}\left[
J_0[\phi_c](X,Y)e^{-it}
+ \eta\phi_2^{(0)}(X,Y)
- \eta\phi_2^{(2)}(X,Y)e^{-2it}
\right].
```

Por tanto, sus derivadas espaciales son

```math
\partial_X\psi
= \operatorname{Re}\left[
\partial_X J_0[\phi_c]e^{-it}
+ \eta\partial_X\phi_2^{(0)}
- \eta\partial_X\phi_2^{(2)}e^{-2it}
\right],
```

```math
\partial_Y\psi
= \operatorname{Re}\left[
\partial_Y J_0[\phi_c]e^{-it}
+ \eta\partial_Y\phi_2^{(0)}
- \eta\partial_Y\phi_2^{(2)}e^{-2it}
\right].
```

## 8. Interpolación periódica

Para evaluar `psi`, `grad_psi` en posiciones que no caen en nodos de la malla:

```text
- usar interpolación bilineal,
- imponer periodicidad 2*pi en X e Y,
- trabajar con X_mod = X % (2*pi), Y_mod = Y % (2*pi).
```

Funciones sugeridas:

```python
def periodic_bilinear_interpolate(field, x, y, L=2*np.pi):
    ...
```

```python
def grad_psi(x, y, t, precomputed_fields, params):
    ...
```

## 9. Campo de centro guía

El lado derecho del sistema de centro guía es

```math
\dot X = -\partial_Y\psi,
```

```math
\dot Y = \partial_X\psi.
```

Función sugerida:

```python
def rhs_guiding_center(t, state, fields, params):
    X, Y = state
    dpsi_dx, dpsi_dy = grad_psi(X, Y, t, fields, params)
    return np.array([-dpsi_dy, dpsi_dx])
```

## 10. Integrador

El artículo usa Runge-Kutta explícito de orden 4.

```python
def rk4_step(f, t, y, dt, *args):
    k1 = f(t, y, *args)
    k2 = f(t + 0.5*dt, y + 0.5*dt*k1, *args)
    k3 = f(t + 0.5*dt, y + 0.5*dt*k2, *args)
    k4 = f(t + dt, y + dt*k3, *args)
    return y + dt*(k1 + 2*k2 + 2*k3 + k4)/6
```

Valores de referencia del artículo:

```text
dt ≈ 0.005
grid_N = 4096
```

Para desarrollo:

```text
dt = 0.01 o 0.02
grid_N = 256, 512, 1024
```

## 11. Muestreo para Poincaré

Como el potencial tiene periodo temporal `2*pi`, las secciones de Poincaré se toman en

```math
t_n=2\pi n.
```

Función sugerida:

```python
def sample_poincare(trajectory, times, period=2*np.pi):
    ...
```

Durante la integración conviene guardar estado cada `2*pi`, no necesariamente cada paso temporal.

## 12. MSD

Entrada:

```text
positions_unwrapped: array shape (N_traj, K, 2)
```

Salida:

```text
taus: array shape (K-1,)
msd: array shape (K-1,)
```

Pseudocódigo:

```python
def compute_msd(positions):
    N_traj, K, dim = positions.shape
    msd = np.zeros(K-1)
    for k in range(1, K):
        acc = 0.0
        count = 0
        for n in range(N_traj):
            diffs = positions[n, k:K, :] - positions[n, 0:K-k, :]
            acc += np.sum(np.sum(diffs*diffs, axis=1))
            count += K-k
        msd[k-1] = acc / count
    return msd
```

Nota: esta fórmula ya combina promedio temporal y promedio en ensamble.

## 13. Estimación del exponente `b`

Ajuste en escala log-log:

```python
def estimate_transport_exponent(times, msd, fit_range=None):
    # elegir solo tiempos positivos y MSD positivo
    # ajustar log(msd) = b log(t) + c
    ...
```

Hay que excluir:

- tiempos demasiado cortos,
- regiones transitorias,
- valores de MSD nulos o contaminados por trayectorias atrapadas.

## 14. Números de rotación

Para una familia de condiciones iniciales con `X0=pi` y `Y0` variable:

```python
def weighted_birkhoff_rotation_number(X_samples):
    S = len(X_samples) - 1
    n = np.arange(1, S)
    w = np.exp(-1.0 / ((n/S)*(1 - n/S)))
    increments = X_samples[2:S+1] - X_samples[1:S]
    return np.sum(w * increments) / np.sum(w)
```

Cuidado: `X_samples` debe ser no envuelto. Si se usa `X mod 2*pi`, los saltos artificiales destruyen el número de rotación.

## 15. Clasificación de trayectorias

El artículo distingue visualmente:

```text
- atrapadas,
- caóticas/difusivas,
- balísticas/superdifusivas.
```

Para una primera implementación se pueden clasificar por criterios simples:

```text
trapped: desplazamiento neto pequeño durante mucho tiempo
ballistic: desplazamiento neto aproximadamente lineal en una dirección
chaotic/diffusive: MSD compatible con b cercano a 1
```

La clasificación fina requiere análisis de Poincaré y números de rotación.

## 16. Órbitas completas

Implementación posterior:

```python
def rhs_full_orbit(t, state, params, phases):
    x, y, vx, vy = state
    gx, gy = grad_phi(x, y, t, params, phases)
    dxdt = params.rho * vx / (2 * abs(params.eta))
    dydt = params.rho * vy / (2 * abs(params.eta))
    dvxdt = -np.sign(params.eta) * gx / params.rho + vy / (2 * params.eta)
    dvydt = -np.sign(params.eta) * gy / params.rho - vx / (2 * params.eta)
    return np.array([dxdt, dydt, dvxdt, dvydt])
```

Esta parte debe implementarse después de tener validada la dinámica de centro guía.

## 17. Reconstrucción del centro guía desde órbitas completas

```python
def reconstruct_guiding_center_from_full_orbit(x, y, vx, vy, rho):
    rho_v = rho * np.sqrt(vx*vx + vy*vy)
    theta = np.pi + np.arctan2(vx, vy)
    X = x - rho_v * np.cos(theta)
    Y = y + rho_v * np.sin(theta)
    return X, Y
```

## 18. Orden de trabajo recomendado

### Fase 1: prototipo pequeño

```text
M = 5 o 10
grid_N = 128 o 256
N_traj = 10-100
dt = 0.01 o 0.02
```

Objetivo: ver potenciales, trayectorias y Poincaré.

### Fase 2: reproducción cualitativa

```text
M = 25
A = 0.7
rho = 0.3
eta = 0.14
```

Objetivo: obtener capas superdifusivas y MSD con pendiente cercana a 2.

### Fase 3: barrido de parámetros

```text
rho in [0,0.3]
eta in [-0.2,0.3]
```

Objetivo: mapa de exponentes `b` comparable a la Fig. 7.

### Fase 4: validación fuerte

```text
grid_N = 4096
dt = 0.005
```

Objetivo: error de energía controlado y comparación con resultados del artículo.

