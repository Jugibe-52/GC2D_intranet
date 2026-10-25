# Proposed behavior

ABBA2Implicit, ABBA4Implicit and BM4Implicit directly inherit
ProjectedPalindromicMethod. The common base owns projection preparation,
solver validation and advancement; concrete methods retain their recipes,
public controls, metadata and observation formats. Remove _ABBAImplicitMethod
and _ComposedABBAImplicit rather than replace them with another family layer.

Keep the existing ABBA2, ABBA4 and BM4 coefficients and 5/15/30 elementary
operations. ABBA4 executes its full triple-jump tuple before one exterior
Hairer projection. Preserve reduced/simultaneous ABBA equations, reduced BM4,
Newton/Broyden controls, signed durations, passive energy and NumPy/JAX support.
Accepted states remain 2N physical components with temporary 4N spatial copies;
time and passive momentum do not enter the nonlinear root.

Remove ABBA6, its public exports, dedicated study, event and derivative adapter,
and active consumers. Do not provide compatibility aliases. Retain historical
specification records and saved scientific data; label retained model documents
as historical and remove ABBA6 from supported-method lists.
