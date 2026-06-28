"""Backward-compatible imports and script entry point for symplectic GC2D runs."""

from gc2d_core.gc2d_symp import *  # noqa: F401,F403
from gc2d_core.gc2d_symp import main


if __name__ == "__main__":
	main()
