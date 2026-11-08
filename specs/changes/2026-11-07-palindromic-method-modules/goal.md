# Goal

Name the numerical family `palindromic` instead of `extended`, and give each
method in that family its own module. The current `abba.py`, `bm4.py`, and
`palindromic.py` group several method classes, making their definitions harder
to locate and their individual responsibilities harder to follow.

The new layout should expose each method's recipe and special behavior while
retaining the shared integration machinery and existing inheritance. This is
an organizational and import-path change; the numerical results must remain
unchanged. The previously implemented Runge–Kutta work remains intact.
