"""Compatibility exports for the symplectic GC2Dt model and legacy CLI."""

from classes.gc2dt import GC2Dt, real_imag
from workflows.symplectic_cli import main, parse_args

__all__ = ["GC2Dt", "main", "parse_args", "real_imag"]


if __name__ == '__main__':
	main()
