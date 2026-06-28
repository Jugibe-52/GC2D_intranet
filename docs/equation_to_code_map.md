# Mapa fórmula -> código

Este archivo conecta las expresiones matemáticas del artículo con funciones de código sugeridas.

| Objeto matemático | Fórmula | Función sugerida | Test sugerido |
|---|---|---|---|
| Conjunto de modos | `I_M={(n,m): 1≤n,m≤M, n²+m²<M²}` | `mode_set(M)` | comprobar número de modos y condición `n²+m²<M²` |
| Fases aleatorias | `varphi_nm ~ Unif(0,2π)` | `generate_phases(M, seed)` | reproducibilidad con misma semilla |
| Potencial turbulento | `phi=A Σ c_nm sin(nx+my+varphi_nm-t)` | `phi_direct(x,y,t,params,phases)` | periodicidad espacial y temporal |
| Gradiente de `phi` | derivadas analíticas del seno | `grad_phi_direct(...)` | comparar con diferencias finitas |
| Potencial complejo | `phi_c=Σ K_nm exp(i(nx+my))` | `complex_potential_coeffs(...)` | reconstruir `phi=Re(phi_c e^{-it})` |
| Gyroaverage | `J0[phi]=1/(2π)∫ phi(...) dθ` | `J0_direct(...)`, `J0_fft(...)` | límite `rho->0` |
| Operador `J1` | `J1[f]=rho^{-1}∂_rho J0[f]` | `J1_operator_fft(...)` | potencial constante da cero |
| Potencial efectivo | `psi=J0[phi]-eta(J1[phi²]-2J0[phi]J1[phi])` | `psi(x,y,t,fields,params)` | potencial constante implica `grad_psi=0` |
| Gradiente efectivo | `∂X psi`, `∂Y psi` | `grad_psi(...)` | comparar FFT/interpolación con diferencias finitas |
| Campo centro guía | `dX/dt=-∂Y psi`, `dY/dt=∂X psi` | `rhs_guiding_center(...)` | potencial `sin X` tiene solución exacta |
| RK4 | método explícito orden 4 | `rk4_step(...)`, `integrate(...)` | convergencia orden 4 en sistema test |
| Poincaré | `(X(2πn),Y(2πn))` | `sample_poincare(...)` | tiempos múltiplos de `2π` |
| MSD | promedio temporal y de ensamble | `compute_msd(...)` | trayectorias balísticas dan `b≈2` |
| Exponente de transporte | `MSD≈(at)^b` | `estimate_transport_exponent(...)` | datos sintéticos con exponente conocido |
| Número de rotación | `r=lim (X_S-X_0)/S` | `rotation_number(...)` | movimiento lineal en X |
| Birkhoff ponderado | media con `omega(t)=exp[-1/(t(1-t))]` | `weighted_birkhoff_rotation_number(...)` | datos sintéticos |
| Órbita completa | sistema en `(x,y,vx,vy)` | `rhs_full_orbit(...)` | límites y comparación con centro guía |
| Reconstrucción de centro guía | `rho_v`, `theta`, `X`, `Y` | `reconstruct_guiding_center_from_full_orbit(...)` | trayectoria circular simple |
| Energía extendida | `h=k+psi` | `extended_energy(...)` | deriva de energía controlada |

## Módulos Python sugeridos

```text
src/phases.py
  mode_set
  generate_phases
  save_phases
  load_phases

src/potential.py
  phi_direct
  grad_phi_direct
  complex_potential_coeffs
  precompute_effective_potential_fft
  psi
  grad_psi

src/interpolation.py
  periodic_bilinear_interpolate
  periodic_gradient_interpolate

src/integrators.py
  rk4_step
  integrate_guiding_center
  integrate_full_orbit

src/diagnostics.py
  compute_msd
  estimate_transport_exponent
  rotation_number
  weighted_birkhoff_rotation_number
  unwrap_positions

src/full_orbit.py
  rhs_full_orbit
  reconstruct_guiding_center_from_full_orbit

src/plots.py
  plot_potential_contours
  plot_poincare
  plot_msd_loglog
  plot_rotation_curve
```

## Implementación mínima viable

La primera versión no debe intentar hacer todo el paper. Debe implementar:

```text
1. mode_set
2. generate_phases
3. phi_direct
4. grad_phi_direct
5. una versión simple de J0 por cuadratura angular
6. psi de primer orden: psi≈J0[phi]
7. rhs_guiding_center
8. RK4
9. una trayectoria
10. un gráfico de Poincaré simple
```

Después añadir:

```text
1. término de segundo orden en psi
2. FFT e interpolación
3. MSD
4. números de rotación
5. barrido rho/eta
6. órbitas completas
```

