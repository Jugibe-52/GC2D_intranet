"""Compatibility exports for shared diagnostic output paths."""

from diagnostics.paths import (
	find_project_root,
	next_block_index,
	notebook_output_directory,
	validate_block_name,
)

__all__ = [
	"find_project_root",
	"next_block_index",
	"notebook_output_directory",
	"validate_block_name",
]
