# Carpeta `docs`: extracción matemática del artículo

Artículo base: **“Transition to superdiffusive transport in turbulent plasmas”**, Physical Review E 110, 025204 (2024), Matteo Stanzani, Filippo Arlotti, Guido Ciraolo, Xavier Garbet y Cristel Chandre.

Esta carpeta convierte el contenido matemático del artículo en una base de trabajo para programar simulaciones numéricas. La idea es evitar pasar directamente de “paper” a “código”, y hacerlo mediante documentos intermedios verificables.

## Archivos

- `fourier_execution/README.md`: explica la ejecucion `run_fourier.py` con el modelo Fourier `FourierSystem`.
- `fourier_execution/flow.puml`: diagrama de flujo de `run_fourier.py`.
- `potential_execution/README.md`: explica la ejecucion `run_potential.py` con `PotentialSystem + Potential`.
- `potential_execution/flow.puml`: diagrama de flujo de `run_potential.py`.
- `article_summary.md`: resumen matemático-físico del artículo y resultados que habría que reproducir.
- `math_model.md`: ecuaciones, variables, parámetros, operadores y magnitudes matemáticas principales.
- `numerical_model.md`: especificación computacional: estructuras de datos, funciones, algoritmos y esquema de integración.
- `validation_tests.md`: tests físicos y numéricos para comprobar que el código implementa correctamente el modelo.
- `equation_to_code_map.md`: tabla que conecta fórmulas del artículo con módulos/funciones Python sugeridas.
- `open_questions.md`: puntos que conviene aclarar antes de implementar una reproducción fiel.

## Flujo recomendado

1. Leer `math_model.md` hasta entender las ecuaciones mínimas.
2. Implementar primero el modelo de centro guía, no las órbitas completas.
3. Crear tests de `validation_tests.md` antes de intentar reproducir todas las figuras.
4. Reproducir una figura simple: potencial `phi`, potencial efectivo `psi`, una trayectoria y una sección de Poincaré.
5. Solo después añadir barridos en `(rho, eta)` y cálculo del exponente de transporte.

## Orden de implementación sugerido

```text
src/
  phases.py        # fases aleatorias fijas phi_nm
  potential.py     # phi, phi_c, J0, J1, psi
  fields.py        # gradientes, campo efectivo
  integrators.py   # RK4, solve_ivp opcional
  diagnostics.py   # MSD, exponente b, rotación
  plots.py         # contornos, Poincaré, log-log MSD

tests/
  test_potential.py
  test_guiding_center.py
  test_msd.py
  test_rotation_number.py
  test_numerical_accuracy.py
```

## Regla de oro

Cada fórmula debe tener tres representaciones:

```text
fórmula del artículo -> función Python -> test físico/numérico
```
