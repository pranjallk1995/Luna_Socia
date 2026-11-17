"""The models a Luna app runs itself, outside Ollama, and how much they work.

Luna Sentia's model monitor sees every Ollama call, but not these: they run inside
the app. Each one counts its calls here, and ``GET /api/models`` reports them, with
the GPU memory this process holds, so the monitor can show them beside Ollama's.
This module is the same in Luna Visia, Lexia, Vibia, and Socia.
"""

from __future__ import annotations

import ctypes
import sys
import threading
import time
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

# Every model this process has, so the route finds them without being handed each one.
_MODELS: "weakref.WeakSet[ModelUsage]" = weakref.WeakSet()

MB = 1024 * 1024

# Loads measured in this process wait for each other, so each one's figure is its own.
_MEASURE = threading.Lock()
# The CUDA runtime's own share (its context), measured once, before the first model.
_context_mb: float | None = None


def gpu_used_mb() -> float | None:
    """The whole card's used memory, by NVML; None without a GPU."""
    try:
        import pynvml

        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).used / MB
    except Exception:  # noqa: BLE001 - no GPU, or no driver library
        return None


@contextmanager
def measuring(usage: "ModelUsage") -> Iterator[None]:
    """Measure the GPU memory a model's load (and first run) takes, as the card's use after less before.

    Under WSL the driver cannot say what one process holds, and ONNX Runtime and CTranslate2 do not
    report it as torch does, so this is the only measure. It is taken once, at start, when nothing
    else should be loading; a model Ollama loads meanwhile would count too.
    """
    with _MEASURE:
        before = gpu_used_mb()
        yield
        after = gpu_used_mb()
    if usage.vram_mb is None and before is not None and after is not None:
        usage.vram_mb = max(after - before, 0.0)


def measure_cuda_context() -> None:
    """Start the CUDA runtime once and measure its context, which every model of this process shares."""
    global _context_mb
    with _MEASURE:
        if _context_mb is not None:
            return
        before = gpu_used_mb()
        try:
            ctypes.CDLL("libcudart.so.12").cudaFree(None)
        except OSError:
            return
        after = gpu_used_mb()
        if before is not None and after is not None:
            _context_mb = max(after - before, 0.0)


class ModelUsage:
    """One in-process model: what it is, whether it is loaded and where, and its calls so far."""

    def __init__(self, name: str, purpose: str, state: Callable[[], tuple[bool, str]]) -> None:
        self.name = name
        self.purpose = purpose
        # Returns (loaded, device), read when the monitor asks.
        self.state = state
        self.calls = 0
        self.failures = 0
        self.busy_seconds = 0.0
        self.last_used: float | None = None
        self.active = 0
        # Its GPU memory once loaded, as measured at its first load; None when unknown or on the CPU.
        self.vram_mb: float | None = None
        self._lock = threading.Lock()
        _MODELS.add(self)

    @contextmanager
    def record(self) -> Iterator[None]:
        """Time one call of the model; a failure still counts as time spent."""
        started = time.perf_counter()
        with self._lock:
            self.active += 1
        try:
            yield
        except BaseException:
            with self._lock:
                self.failures += 1
            raise
        finally:
            with self._lock:
                self.active -= 1
                self.calls += 1
                self.busy_seconds += time.perf_counter() - started
                self.last_used = time.time()

    def status(self) -> dict[str, Any]:
        loaded, device = self.state()
        with self._lock:
            return {
                "name": self.name,
                "purpose": self.purpose,
                "loaded": loaded,
                "device": device,
                "calls": self.calls,
                "failures": self.failures,
                "busy_seconds": round(self.busy_seconds, 3),
                "last_used": self.last_used,
                "active": self.active,
                "vram_mb": None if self.vram_mb is None else round(self.vram_mb),
            }


def process_gpu() -> dict[str, float] | None:
    """GPU memory this process holds through torch, or None when it has not used the GPU.

    Under WSL the driver cannot say how much each process uses, so this is the only
    way to see these models' share of the card. Torch is never imported just to ask.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        # Models outside torch: what was measured as each loaded, and the CUDA runtime's own share.
        loaded = [m.vram_mb for m in list(_MODELS) if m.vram_mb is not None and m.state()[0]]
        if _context_mb is None and not loaded:
            return None
        context = _context_mb or 0.0
        return {"reserved_mb": context + sum(loaded), "context_mb": context, "measured": True}
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return None
        return {
            "allocated_mb": torch.cuda.memory_allocated() / MB,
            "reserved_mb": torch.cuda.memory_reserved() / MB,
        }
    except Exception:  # noqa: BLE001 - a status report never fails the app
        return None


def report(app: str) -> dict[str, Any]:
    """What ``GET /api/models`` answers: this app's own models and its share of the GPU."""
    return {
        "app": app,
        "models": sorted((model.status() for model in list(_MODELS)), key=lambda row: row["name"]),
        "gpu": process_gpu(),
        "at": time.time(),
    }
