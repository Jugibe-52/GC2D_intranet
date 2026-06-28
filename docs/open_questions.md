# Puntos a aclarar antes de una reproducción fiel

## 1. Fases aleatorias exactas

El artículo indica que en todas las simulaciones se usa un conjunto fijo de fases aleatorias `varphi_nm`. Para reproducir exactamente las figuras habría que conocer esas fases o descargar el código enlazado por el artículo.

Sin las fases exactas, se puede reproducir el fenómeno cualitativo, pero no necesariamente las mismas secciones de Poincaré.

## 2. Repositorio original

El artículo menciona un código Python disponible en GitHub para integrar las órbitas completas y la dinámica de centro guía. Para una reproducción exacta, conviene mirar:

```text
https://github.com/cchandre/Guiding-Center
```

El código original debería aclarar:

```text
- fases exactas,
- condiciones iniciales,
- tamaño del ensamble,
- detalles de clasificación de partículas,
- rangos de ajuste del exponente b,
- detalles de unwrapping,
- valores exactos de malla y tiempo final.
```

## 3. Condiciones iniciales

El artículo dice que se integran grandes ensambles de condiciones iniciales en `[0,2*pi)^2`, pero para reproducir figuras concretas hay que fijar:

```text
- número de condiciones iniciales,
- distribución exacta,
- semilla aleatoria,
- condiciones usadas para curvas especiales/twistless.
```

## 4. Tratamiento de `rho=0` y `eta=0`

El caso de referencia difusivo aparece como `rho=eta=0`. En la dinámica de órbitas completas algunas fórmulas tienen divisiones por `rho` y `eta`, así que este caso debe entenderse dentro del modelo reducido o como límite.

Antes de programar, conviene decidir cómo se tratará:

```text
- usar directamente el modelo de centro guía de primer orden,
- fijar rho muy pequeño,
- fijar eta=0 en la expresión de psi eliminando el segundo orden,
- o consultar el código original.
```

## 5. Ajuste del exponente `b`

El artículo ajusta

```math
MSD(t) \approx (a t)^b.
```

Pero para reproducir la Fig. 7 hay que saber:

```text
- intervalo temporal usado para el ajuste,
- si se excluyen partículas atrapadas,
- criterios para decidir que no hay superdifusión significativa,
- número de trayectorias,
- tiempo final.
```

## 6. Clasificación de trayectorias

Las figuras distinguen trayectorias atrapadas, caóticas y balísticas. La clasificación visual es clara, pero para automatizarla hacen falta criterios concretos.

Posibles criterios:

```text
- desplazamiento neto,
- varianza transversal,
- exponente local de MSD,
- pertenencia a capas en Poincaré,
- número de rotación regular.
```

## 7. Precisión numérica

El artículo usa `N=4096` para la malla y `dt≈0.005`. Esto puede ser pesado para un portátil. Para desarrollo se deben usar valores más bajos y solo al final intentar acercarse a la resolución del paper.

## 8. Notación `J1`

El artículo usa `J1` tanto para el operador

```math
J_1[f]=\rho^{-1}\partial_\rho J_0[f]
```

como para la función de Bessel de primer orden. En código hay que separar ambos nombres para evitar errores.

Propuesta:

```text
J1_operator
bessel_j1
```

