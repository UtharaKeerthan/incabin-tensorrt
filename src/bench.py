"""Timing harness used by every notebook.

Two ways to time GPU work, and why both exist:

* Wall-clock (``benchmark``): time.perf_counter() around the call, with a
  device sync before reading the clock. Measures what the application sees,
  including Python overhead, kernel launches and host<->device copies.
* CUDA events (``benchmark_cuda_events``): timestamps recorded *on the GPU
  stream*. Measures GPU execution time only. Python overhead is excluded,
  so this number is usually smaller.

The gap between the two numbers is itself a useful measurement: it shows
how much time is spent outside the GPU (launch overhead, Python, copies).
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np


# --------------------------------------------------------------------------
# Device sync helpers
# --------------------------------------------------------------------------
def cuda_sync_fn() -> Optional[Callable[[], None]]:
    """Return a function that blocks until all queued GPU work is done.

    GPU calls are asynchronous: they return as soon as the work is *queued*,
    not when it finishes. Without a sync before reading the clock, you time
    the queueing, not the work.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.synchronize
    except ImportError:
        pass
    return None


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------
@dataclass
class BenchResult:
    """Latency samples for one configuration, plus summary statistics."""

    name: str
    times_ms: np.ndarray
    batch: int = 1
    method: str = "wall-clock"
    meta: Dict[str, object] = field(default_factory=dict)

    def pct(self, q: float) -> float:
        return float(np.percentile(self.times_ms, q))

    @property
    def p50(self) -> float:
        return self.pct(50)

    @property
    def p90(self) -> float:
        return self.pct(90)

    @property
    def p99(self) -> float:
        return self.pct(99)

    @property
    def mean(self) -> float:
        return float(self.times_ms.mean())

    @property
    def std(self) -> float:
        return float(self.times_ms.std())

    @property
    def fps(self) -> float:
        """Items per second, based on the median latency."""
        return self.batch * 1000.0 / self.p50

    def summary(self) -> Dict[str, object]:
        row = {
            "name": self.name,
            "method": self.method,
            "batch": self.batch,
            "p50_ms": round(self.p50, 4),
            "p90_ms": round(self.p90, 4),
            "p99_ms": round(self.p99, 4),
            "mean_ms": round(self.mean, 4),
            "std_ms": round(self.std, 4),
            "min_ms": round(float(self.times_ms.min()), 4),
            "max_ms": round(float(self.times_ms.max()), 4),
            "jitter_p99_p50": round(self.p99 / self.p50, 3),
            "items_per_s": round(self.fps, 1),
        }
        row.update(self.meta)
        return row

    def __repr__(self) -> str:
        return (
            f"BenchResult({self.name!r}, batch={self.batch}, "
            f"p50={self.p50:.4f} ms, p99={self.p99:.4f} ms, "
            f"{self.fps:.1f} items/s, n={len(self.times_ms)})"
        )


# --------------------------------------------------------------------------
# Benchmark functions
# --------------------------------------------------------------------------
def benchmark(
    fn: Callable[[], object],
    *,
    name: str,
    warmup: int = 50,
    iters: int = 1000,
    batch: int = 1,
    sync: Optional[Callable[[], None]] = "auto",  # type: ignore[assignment]
    meta: Optional[Dict[str, object]] = None,
) -> BenchResult:
    """Wall-clock benchmark of ``fn``.

    Args:
        fn: zero-argument callable that runs one inference (or one step).
        warmup: untimed runs first. They absorb one-off costs: CUDA context
            creation, lazy memory allocation, cuDNN/TensorRT tactic caches,
            CPU frequency ramp-up.
        iters: timed runs. 1000 gives a stable p99 (10 samples above it).
        batch: items per call, used to compute throughput.
        sync: function that waits for the GPU. "auto" picks
            torch.cuda.synchronize when CUDA is available; pass None to
            time without syncing (only correct for CPU work).
    """
    if sync == "auto":
        sync = cuda_sync_fn()

    for _ in range(warmup):
        fn()
    if sync:
        sync()

    times = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        times[i] = (time.perf_counter() - t0) * 1000.0

    return BenchResult(name, times, batch=batch, method="wall-clock", meta=dict(meta or {}))


def benchmark_cuda_events(
    fn: Callable[[], object],
    *,
    name: str,
    warmup: int = 50,
    iters: int = 1000,
    batch: int = 1,
    meta: Optional[Dict[str, object]] = None,
) -> BenchResult:
    """GPU-side benchmark of ``fn`` using CUDA events on the current stream.

    Only correct when ``fn`` enqueues its work on torch's current CUDA
    stream (true for PyTorch ops; for TensorRT we pass that stream handle
    explicitly in later notebooks).
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA events need a CUDA GPU")

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    times = np.array([s.elapsed_time(e) for s, e in zip(starts, ends)], dtype=np.float64)
    return BenchResult(name, times, batch=batch, method="cuda-events", meta=dict(meta or {}))


# --------------------------------------------------------------------------
# Stage timer for multi-stage pipelines (used from notebook 06 onwards)
# --------------------------------------------------------------------------
class StageTimer:
    """Accumulates per-stage latencies across many frames.

    Example:
        st = StageTimer()
        for frame in frames:
            with st("preprocess"):
                x = prep(frame)
            with st("inference"):
                y = infer(x)
        st.table()
    """

    def __init__(self, sync: Optional[Callable[[], None]] = "auto"):  # type: ignore[assignment]
        self.sync = cuda_sync_fn() if sync == "auto" else sync
        self.samples: Dict[str, List[float]] = {}

    @contextmanager
    def __call__(self, stage: str):
        if self.sync:
            self.sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.sync:
                self.sync()
            self.samples.setdefault(stage, []).append((time.perf_counter() - t0) * 1000.0)

    def results(self) -> List[BenchResult]:
        return [BenchResult(k, np.asarray(v)) for k, v in self.samples.items()]

    def table(self):
        import pandas as pd

        rows = []
        total = sum(np.median(v) for v in self.samples.values())
        for k, v in self.samples.items():
            v = np.asarray(v)
            rows.append(
                {
                    "stage": k,
                    "p50_ms": round(float(np.median(v)), 4),
                    "p99_ms": round(float(np.percentile(v, 99)), 4),
                    "share_of_p50_total_%": round(100 * float(np.median(v)) / total, 1),
                    "n": len(v),
                }
            )
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def results_table(results: Iterable[BenchResult], baseline: Optional[str] = None):
    """Summary DataFrame. If ``baseline`` names a result, adds a speedup column."""
    import pandas as pd

    results = list(results)
    df = pd.DataFrame([r.summary() for r in results])
    if baseline is not None:
        base = next(r for r in results if r.name == baseline)
        df["speedup_vs_" + baseline] = [round(base.p50 / r.p50, 2) for r in results]
    return df


def plot_latency(results: Iterable[BenchResult], title: str = "Latency distribution", log: bool = False):
    """Box plot of latency samples. Shows spread and outliers, not just medians."""
    import matplotlib.pyplot as plt

    results = list(results)
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(results)), 4))
    ax.boxplot([r.times_ms for r in results], showfliers=True)
    ax.set_xticks(range(1, len(results) + 1), [r.name for r in results], rotation=20, ha="right")
    ax.set_ylabel("latency (ms)")
    if log:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig
