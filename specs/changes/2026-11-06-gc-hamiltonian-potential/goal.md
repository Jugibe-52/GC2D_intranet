# Name the guiding-center Hamiltonian explicitly

The guiding-center object currently calls its prepared Hamiltonian `effective_potential` and obtains drift through `electric_field`. Since this object already includes gyroaveraging and division by the signed magnetic field B, the electric-field terminology obscures which scalar function generates the motion.

Make that role explicit while preserving numerical behavior and the existing spline representation.
