# Proposal

For each particle use M_0=I and M_(n+1)=A_n M_n, with A_n evaluated along the numerical trajectory. Save all local 2-by-2 matrices and the accumulated matrices at 10, 20, 40 and 80 cycles. Evaluate abs(det(M_N)-1) directly. Use decimal matrix arithmetic and a higher-precision repeat to check cancellation without replacing the product by determinant identities. Keep the accepted methods, fixed steps, radius and groups. Recalculate the sweep in a new immutable run, regenerate boxplots and update the thesis interpretation.
