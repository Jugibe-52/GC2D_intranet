"""Structural contracts consumed by studies.poincare_batch_runtime.cloud_job."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol


class S3Paginator(Protocol):
	"""Object-listing interface used by checkpoint restoration."""

	def paginate(self, **kwargs: str) -> Iterable[dict[str, Any]]: ...


class S3Client(Protocol):
	"""File-transfer interface shared by S3 and offline checkpoint tests."""

	def upload_file(self, file: str, bucket: str, key: str) -> None: ...
	def download_file(self, bucket: str, key: str, file: str) -> None: ...
	def get_paginator(self, name: str) -> S3Paginator: ...


__all__ = ['S3Paginator', 'S3Client']
