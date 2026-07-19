"""Fourier simulation domain exports and command-line entry point."""

from classes import (
	FourierPotential,
	System,
	SystemFC,
	SystemGC,
	TrajectoryFC,
	TrajectoryGC,
	create_system,
)
from classes.potential import real_imag
from workflows.params import make_system
from workflows.symplectic_cli import main, parse_args

__all__ = [
	"FourierPotential",
	"System",
	"SystemFC",
	"SystemGC",
	"TrajectoryFC",
	"TrajectoryGC",
	"create_system",
	"main",
	"make_system",
	"parse_args",
	"real_imag",
]


if __name__ == "__main__":
	main()
