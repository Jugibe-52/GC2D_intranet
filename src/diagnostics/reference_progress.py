"""Flush reference progress to the notebook and a persistent text log."""

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterator
import math


@contextmanager
def reference_progress_log(
    path: str | Path,
    *,
    t_span: tuple[float, float],
    interval_seconds: float = 10.0,
) -> Iterator[Callable[[str, float, int, str], None]]:
    """Report accepted-time progress and a per-solver wall-time ETA.

    ETA extrapolates the mean accepted-time throughput, so it is approximate
    and excludes the other solver and artifact writing. A log is not a restart
    checkpoint. The context must surround integration and persistence.
    """
    t0, end = t_span
    if not (math.isfinite(t0) and math.isfinite(end) and end > t0):
        raise ValueError("The progress time span must be finite and increasing.")
    if not math.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError("The log interval must be finite and positive.")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        starts: dict[str, float] = {}
        last: dict[str, float] = {}

        def emit(message: str) -> None:
            line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} | {message}"
            print(line, flush=True)
            stream.write(line + "\n")
            stream.flush()

        def report(method: str, time: float, nfev: int, status: str) -> None:
            now = perf_counter()
            starts.setdefault(method, now)
            if status == "running" and now - last.get(method, -math.inf) < interval_seconds:
                return
            last[method] = now
            fraction = min(1.0, max(0.0, (time - t0) / (end - t0)))
            elapsed = now - starts[method]
            eta = f"{elapsed * (1 - fraction) / fraction:.1f}s" if fraction > 0 else "pending"
            phase = "1/2 audit" if method == "Radau" else "2/2 reference"
            emit(f"{phase} {method} {status} | t={time:.8g}/{end:g} | "
                 f"{100*fraction:.2f}% | remaining simulated time={end-time:.8g} | "
                 f"elapsed={elapsed:.1f}s | ETA this solver~{eta} | nfev={nfev}")

        emit(f"Reference run started | t_span={t_span} | Radau then DOP853 | "
             "ETA is approximate and per solver; persistence follows both solves.")
        try:
            yield report
        except BaseException as exc:
            emit(f"Run interrupted or failed: {type(exc).__name__}: {exc}")
            raise
        else:
            emit("Reference computation and artifact persistence completed.")
