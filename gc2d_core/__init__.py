"""Compatibility package loader for the src-based GC2D core package."""

from pathlib import Path

_SRC_PACKAGE = Path(__file__).resolve().parent.parent / "src" / "gc2d_core"
if _SRC_PACKAGE.is_dir():
	__path__.append(str(_SRC_PACKAGE))
