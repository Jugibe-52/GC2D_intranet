"""Compatibility exports for the symplectic FourierSystem model and legacy CLI."""

from classes.fourier_system import FourierSystem, real_imag
from workflows.symplectic_cli import main, parse_args

__all__ = ["FourierSystem", "main", "parse_args", "real_imag"]


if __name__ == '__main__':
	main()
