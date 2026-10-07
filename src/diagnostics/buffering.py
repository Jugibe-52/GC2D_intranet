"""Shared buffered writing and exception-preserving observer finalization."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np

from diagnostics._validation import positive_integer
from diagnostics.output import DiagnosticBlockPaths, write_diagnostic_block
from diagnostics.paths import next_block_index, validate_block_name


Sample = TypeVar("Sample")
Block = TypeVar("Block")


class DiagnosticBuffer(Generic[Sample, Block]):
	"""Own pending samples, indexed writes, and one observer's close lifecycle.

	The observer supplies only block-specific serialization. Successful writes
	advance the index and release pending samples; failed writes keep the buffer
	available for explicit retry. Retention of interactive history belongs to
	the observer and is independent of this bounded pending-output buffer.
	"""

	def __init__(
		self,
		*,
		output_directory: Path,
		block_name: str,
		chunk_size: int,
		write_block: Callable[[int, tuple[Sample, ...]], Block],
	) -> None:
		"""Bind one typed serializer without creating files before the first flush."""
		self.output_directory = output_directory
		self.block_name = validate_block_name(block_name)
		self.chunk_size = positive_integer(chunk_size, "chunk_size")
		self._write_block = write_block
		self._pending: list[Sample] = []
		self._blocks: list[Block] = []
		self._next_index = next_block_index(output_directory, self.block_name)
		self.closed = False

	@property
	def blocks(self) -> tuple[Block, ...]:
		"""Return descriptors of successfully written output blocks."""
		return tuple(self._blocks)

	def append(self, sample: Sample) -> None:
		"""Queue a sample; the observer can finish its bookkeeping before flushing."""
		if self.closed:
			raise RuntimeError("This diagnostic output buffer is already closed.")
		self._pending.append(sample)

	def flush_if_full(self) -> Block | None:
		"""Write one chunk when the configured pending-sample limit is reached."""
		return self.flush() if len(self._pending) >= self.chunk_size else None

	def flush(self) -> Block | None:
		"""Commit pending samples and preserve them unchanged if writing fails.

		This also retries pending writes after an exceptional context exit has
		closed the buffer to new samples; it does not reopen observation.
		"""
		if not self._pending:
			return None
		block = self._write_block(self._next_index, tuple(self._pending))
		self._blocks.append(block)
		self._pending.clear()
		self._next_index += 1
		return block

	def write(
		self,
		*,
		block_index: int,
		rows: Sequence[Mapping[str, object]],
		arrays: Mapping[str, np.ndarray],
		metadata: Mapping[str, Any],
		arrays_kind: str = "jacobians",
	) -> DiagnosticBlockPaths:
		"""Write one observer-specific payload through the shared output sink."""
		return write_diagnostic_block(
			output_directory=self.output_directory,
			block_name=self.block_name,
			block_index=block_index,
			rows=rows,
			arrays=arrays,
			metadata=metadata,
			arrays_kind=arrays_kind,
		)

	def close(self, finalize: Callable[[], None] | None = None) -> None:
		"""Retain an optional final completed sample, flush, and close once."""
		if self.closed:
			return
		if finalize is not None:
			finalize()
		self.flush()
		self.closed = True

	def exit(
		self,
		exception_type: type[BaseException] | None,
		exception: BaseException | None,
		finalize: Callable[[], None] | None = None,
	) -> None:
		"""Close a context without replacing an active integration exception."""
		if exception_type is None:
			self.close(finalize)
			return
		try:
			self.close(finalize)
		except Exception as cleanup_error:
			if exception is not None:
				exception.add_note(f"Diagnostic output cleanup also failed: {cleanup_error!r}")
		finally:
			self.closed = True


__all__ = ["DiagnosticBuffer"]
