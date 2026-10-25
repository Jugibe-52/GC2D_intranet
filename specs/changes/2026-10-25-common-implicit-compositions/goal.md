# One base for implicit compositions

Make ABBA2Implicit, ABBA4Implicit and BM4Implicit directly share the same
palindromic projection base. Remove ABBA-specific intermediate classes and the
ABBA6 method. Each remaining recipe stays an explicit tuple on its method class.

ABBA4 remains a complete operation composition with one exterior projection.
It does not become a wrapper around three separately projected ABBA2 steps.
