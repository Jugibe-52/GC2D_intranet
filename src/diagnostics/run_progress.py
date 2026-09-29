"""Opt-in console/file progress and heartbeats outside numerical kernels."""

from contextlib import contextmanager
from collections.abc import Iterator
import logging
from pathlib import Path
import sys
from threading import Event, Lock, Thread
from time import perf_counter
from uuid import uuid4

import numpy as np


class RunProgress:
    """Report completed units and wall-clock estimates using a private logger."""

    def __init__(self, logger: logging.Logger, total: int, unit: str) -> None:
        self.logger, self.total, self.unit = logger, total, unit
        self.started = perf_counter()
        self.completed = 0
        self.current_phase = 'Preparing'
        self._lock = Lock()

    def phase(self, description: str, *, completed: int | None = None) -> None:
        """Publish an actual completed boundary; estimates never advance it."""
        with self._lock:
            if completed is not None:
                if not self.completed <= completed <= self.total:
                    raise ValueError('Completed work must increase monotonically within the total.')
                self.completed = completed
            self.current_phase = description
        self.report()

    def report(self) -> None:
        """Emit a flushed heartbeat from the latest completed boundary."""
        with self._lock:
            completed, phase = self.completed, self.current_phase
        elapsed = perf_counter() - self.started
        remaining = f'{elapsed * (self.total - completed) / completed:.1f}s' if completed else 'pending'
        self.logger.info('%s | %s=%d/%d | %.1f%% | elapsed=%.1fs | ETA=%s',
                         phase, self.unit, completed, self.total, 100 * completed / self.total,
                         elapsed, remaining)


@contextmanager
def progress_log(path: str | Path, *, total: int, unit: str = 'cycles',
                 heartbeat_seconds: float = 10.) -> Iterator[RunProgress]:
    """Append timestamps, progress and exceptions to a file and notebook output.

    A daemon emits heartbeats even during JIT compilation, field preparation or
    upload. It does not call the solver or enter a compiled JAX computation.
    Both handlers are flushed and removed on success, error or interruption.
    """
    if isinstance(total, bool) or not isinstance(total, int) or total < 1:
        raise ValueError('total must be a positive integer.')
    if not np.isfinite(heartbeat_seconds) or heartbeat_seconds <= 0:
        raise ValueError('heartbeat_seconds must be positive and finite.')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f'gc2d.run.{uuid4().hex}')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handlers: list[logging.Handler] = [logging.FileHandler(path, encoding='utf-8'), logging.StreamHandler(sys.stdout)]
    for handler in handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
        logger.addHandler(handler)
    progress = RunProgress(logger, total, unit)
    stop = Event()

    def heartbeat() -> None:
        while not stop.wait(heartbeat_seconds):
            progress.report()

    worker = Thread(target=heartbeat, daemon=True, name='gc2d-progress')
    try:
        logger.info('START | log=%s', path.resolve())
        worker.start()
        yield progress
    except BaseException:
        logger.exception('FAILED OR INTERRUPTED | no successful completion is recorded')
        raise
    finally:
        stop.set()
        worker.join()
        for handler in handlers:
            handler.flush()
            handler.close()
            logger.removeHandler(handler)


__all__ = ['RunProgress', 'progress_log']
