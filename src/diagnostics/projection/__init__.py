"""Projected GC symplecticity and transported-area diagnostics."""

from .observer import (
	ProjectedAreaOutputBlock,
	ProjectedAreaRecord,
	ProjectedSymplecticityAreaObserver,
	gc_average_projection,
	gc_diagonal_embedding,
	gc_physical_symplectic_form,
)

__all__ = [
	"ProjectedAreaOutputBlock",
	"ProjectedAreaRecord",
	"ProjectedSymplecticityAreaObserver",
	"gc_average_projection",
	"gc_diagonal_embedding",
	"gc_physical_symplectic_form",
]
