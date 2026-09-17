### Resultado de esta ejecución

- **Oscilador**, h = 0.2, T = 10000: BM4 alcanza un máximo observado de **2.243002e-07** y RK4 de **2.162935e-02**. En BM4, E_max(T)/E_max(1000) = **1.000001** y la pendiente respecto a h es **4.0020**.
- **Péndulo**, h = 0.2, T = 10000: BM4 alcanza un máximo observado de **3.454186e-08** y RK4 de **2.356108e-02**. En BM4, E_max(T)/E_max(1000) = **1.000001** y la pendiente respecto a h es **4.0298**.
- **Cota analítica del oscilador para BM4:** B_h = **2.243002e-07**, independiente de n para el mapa estable en aritmética exacta.
- **Contraejemplo:** con h = 7, BM4 sigue siendo simpléctico, pero H_final/H_0 = **1.091e+17** tras 100 pasos.

**Interpretación:** en el régimen ensayado, comprobar cocientes cercanos a uno y pendientes cercanas a cuatro respalda una envolvente de tamaño O(h⁴). En el oscilador la cota se deduce analíticamente; en el péndulo el resultado es evidencia finita. La simplecticidad por sí sola no garantiza una cota pequeña universal.

**Matiz sobre RK4:** su error energético en el oscilador estable también está acotado por H₀ a tiempo infinito, porque H_n tiende a cero. Eso no equivale a conservar bien la energía.
