"""Verify the exact factorized Jacobian of the twelve-stage BM4 map.

Run this documentation helper from the project root with
``.venv/bin/python docs/models/bm4/bm4_jacobian_sympy.py``. The calculation treats
the stage blocks as noncommuting objects because field Jacobians evaluated at
different BM4 stage points cannot in general be reordered.
"""

from __future__ import annotations

from collections.abc import Sequence

import sympy as sp


_BM4_COEFFICIENT_NAMES = (
	"a_1",
	"a_2",
	"a_3",
	"a_4",
	"a_5",
	"a_6",
	"a_6",
	"a_5",
	"a_4",
	"a_3",
	"a_2",
	"a_1",
)


def _join_blocks(
	top_left: sp.Matrix,
	top_right: sp.Matrix,
	bottom_left: sp.Matrix,
	bottom_right: sp.Matrix,
) -> sp.Matrix:
	"""Assemble a four-by-four matrix from four two-by-two blocks."""
	return top_left.row_join(top_right).col_join(
		bottom_left.row_join(bottom_right)
	)


def _generic_stage_matrices() -> tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
	"""Return coupling, direct, and adjoint Jacobians for one generic stage."""
	duration, frequency = sp.symbols("s omega", real=True)
	w_entries = sp.symbols("w11 w12 w21 w22")
	x_entries = sp.symbols("x11 x12 x21 x22")
	w_first = sp.Matrix(2, 2, w_entries)
	w_second = sp.Matrix(2, 2, x_entries)
	identity = sp.eye(2)

	angle = 2 * frequency * duration
	rotation = sp.Matrix(
		[
			[sp.cos(angle), -sp.sin(angle)],
			[sp.sin(angle), sp.cos(angle)],
		]
	)
	mean_block = (identity + rotation) / 2
	difference_block = (identity - rotation) / 2
	coupling = _join_blocks(
		mean_block,
		difference_block,
		difference_block,
		mean_block,
	)

	direct_shears = _join_blocks(
		identity + duration**2 * w_second * w_first,
		duration * w_second,
		duration * w_first,
		identity,
	)
	direct = coupling * direct_shears
	direct_explicit = _join_blocks(
		mean_block * (identity + duration**2 * w_second * w_first)
		+ difference_block * duration * w_first,
		mean_block * duration * w_second + difference_block,
		difference_block * (identity + duration**2 * w_second * w_first)
		+ mean_block * duration * w_first,
		difference_block * duration * w_second + mean_block,
	)

	adjoint_shears = _join_blocks(
		identity,
		duration * w_first,
		duration * w_second,
		identity + duration**2 * w_second * w_first,
	)
	adjoint = adjoint_shears * coupling
	adjoint_explicit = _join_blocks(
		mean_block + duration * w_first * difference_block,
		difference_block + duration * w_first * mean_block,
		duration * w_second * mean_block
		+ (identity + duration**2 * w_second * w_first) * difference_block,
		duration * w_second * difference_block
		+ (identity + duration**2 * w_second * w_first) * mean_block,
	)

	_require_zero(direct - direct_explicit, "direct stage factorization")
	_require_zero(adjoint - adjoint_explicit, "adjoint stage factorization")
	_require_zero(coupling.T * coupling - sp.eye(4), "coupling orthogonality")
	return coupling, direct, adjoint


def _require_zero(matrix: sp.Matrix, description: str) -> None:
	"""Raise when symbolic simplification does not prove a matrix identity."""
	if any(sp.simplify(entry) != 0 for entry in matrix):
		raise RuntimeError(f"SymPy could not verify {description}.")


def _generic_block_stage(index: int) -> sp.Matrix:
	"""Create one two-by-two matrix of noncommuting two-by-two block symbols."""
	return sp.Matrix(
		2,
		2,
		lambda row, column: sp.Symbol(
			f"F{index}_{row + 1}{column + 1}",
			commutative=False,
		),
	)


def _expanded_block_term_counts(stage_count: int = 12) -> tuple[int, ...]:
	"""Count ordered monomials in each final block after exact expansion."""
	product = sp.eye(2)
	for index in range(1, stage_count + 1):
		# Left multiplication follows the chain rule: J_j ... J_2 J_1.
		product = (_generic_block_stage(index) * product).applyfunc(sp.expand)
	return tuple(len(sp.Add.make_args(entry)) for entry in product)


def _ordered_product(coefficient_names: Sequence[str]) -> str:
	"""Format the BM4 Jacobian factors in chain-rule order."""
	factors = []
	for stage in range(len(coefficient_names), 0, -1):
		kind = "A" if stage % 2 else "D"
		coefficient = coefficient_names[stage - 1]
		factors.append(f"{kind}_{stage}({coefficient}h)")
	return " ".join(factors)


def main() -> None:
	"""Run all symbolic checks and print the compact exact result."""
	_generic_stage_matrices()
	term_counts = _expanded_block_term_counts()
	print(f"SymPy version: {sp.__version__}")
	print("Direct and adjoint stage identities: verified")
	print("Coupling orthogonality: verified")
	print("J_BM4 =", _ordered_product(_BM4_COEFFICIENT_NAMES))
	print("Expanded monomials per final 2x2 block:", term_counts)
	print("Total expanded block monomials:", sum(term_counts))


if __name__ == "__main__":
	main()
