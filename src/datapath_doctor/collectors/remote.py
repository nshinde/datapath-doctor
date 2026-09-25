"""Backend-agnostic telemetry for remote/object-store reads.

This collector does not require a block device. It can wrap reads from fsspec,
S3 clients, WebDataset stream openers, HTTP readers, or custom storage clients
and normalize them into StepSample-compatible fields.
"""

from __future__ import annotations

import time
from typing import Any, BinaryIO, Callable, Optional, TypeVar

T = TypeVar("T")


class RemoteReadCollector:
    def __init__(
        self,
        storage_backend: str,
        filesystem_type: Optional[str] = None,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.storage_backend = storage_backend
        self.filesystem_type = filesystem_type
        self._clock = clock
        self._window_start = clock()
        self._latencies_ms: list[float] = []
        self._bytes = 0
        self._ops = 0
        self._retries = 0
        self._errors = 0

    def record_read(
        self,
        latency_s: float,
        *,
        bytes_read: int = 0,
        retries: int = 0,
        errors: int = 0,
    ) -> None:
        self._latencies_ms.append(max(0.0, latency_s) * 1000.0)
        self._bytes += max(0, int(bytes_read))
        self._ops += 1
        self._retries += max(0, int(retries))
        self._errors += max(0, int(errors))

    def measure_read(
        self,
        read_fn: Callable[..., T],
        *args: Any,
        retries: int = 0,
        **kwargs: Any,
    ) -> T:
        """Execute one read call and record latency/bytes/errors.

        Bytes are inferred for bytes-like/string results. For structured
        results, call ``record_read`` directly when a precise byte count is
        available.
        """
        start = self._clock()
        try:
            result = read_fn(*args, **kwargs)
        except Exception:
            self.record_read(self._clock() - start, retries=retries, errors=1)
            raise

        if isinstance(result, str):
            bytes_read = len(result.encode())
        elif isinstance(result, (bytes, bytearray, memoryview)):
            bytes_read = len(result)
        else:
            bytes_read = 0
        self.record_read(
            self._clock() - start,
            bytes_read=bytes_read,
            retries=retries,
        )
        return result

    def wrap_file(self, fileobj: BinaryIO) -> "InstrumentedFile":
        """Wrap a file-like object; every ``read()`` call is measured."""
        return InstrumentedFile(fileobj, self)

    def sample(self, *, reset: bool = True) -> dict[str, Any]:
        """Return normalized StepSample fields for the current collection window."""
        now = self._clock()
        elapsed = max(now - self._window_start, 1e-9)
        latency = (
            sum(self._latencies_ms) / len(self._latencies_ms)
            if self._latencies_ms
            else None
        )
        result = {
            "is_network_fs": True,
            "storage_backend": self.storage_backend,
            "filesystem_type": self.filesystem_type,
            "remote_read_latency_ms": latency,
            "remote_read_ops_per_s": self._ops / elapsed,
            "remote_read_bytes_per_s": self._bytes / elapsed,
            "remote_retries": self._retries,
            "remote_errors": self._errors,
        }
        if reset:
            self._window_start = now
            self._latencies_ms.clear()
            self._bytes = 0
            self._ops = 0
            self._retries = 0
            self._errors = 0
        return result


class InstrumentedFile:
    """Thin proxy for file-like readers used by fsspec/object-store clients."""

    def __init__(self, fileobj: BinaryIO, collector: RemoteReadCollector) -> None:
        self._fileobj = fileobj
        self._collector = collector

    def read(self, *args: Any, **kwargs: Any) -> Any:
        return self._collector.measure_read(self._fileobj.read, *args, **kwargs)

    def __iter__(self) -> "InstrumentedFile":
        return self

    def __next__(self) -> Any:
        return self._collector.measure_read(next, self._fileobj)

    def __enter__(self) -> "InstrumentedFile":
        if hasattr(self._fileobj, "__enter__"):
            self._fileobj.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if hasattr(self._fileobj, "__exit__"):
            return self._fileobj.__exit__(exc_type, exc, tb)
        self.close()
        return False

    def close(self) -> None:
        self._fileobj.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._fileobj, name)
