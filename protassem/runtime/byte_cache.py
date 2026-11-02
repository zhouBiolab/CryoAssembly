
import sys
from collections import OrderedDict
from dataclasses import dataclass

ENTRY_OVERHEAD_BYTES = 1024


@dataclass
class CacheStats:

    name: str
    capacity_bytes: int
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    rejected_too_large: int = 0
    disabled_puts: int = 0
    entries: int = 0
    bytes: int = 0
    peak_bytes: int = 0

    def as_dict(self):
        data = {"cache": self.name, "capacity_bytes": self.capacity_bytes,
                "hits": self.hits, "misses": self.misses,
                "evictions": self.evictions,
                "rejected_too_large": self.rejected_too_large,
                "disabled_puts": self.disabled_puts,
                "entries": self.entries, "bytes": self.bytes,
                "peak_bytes": self.peak_bytes}
        total = self.hits + self.misses
        data["hit_rate"] = round(float(self.hits) / total, 6) if total else None
        return data


@dataclass
class _Entry:
    value: object
    size: int


class ByteLruCache:

    def __init__(self, capacity_bytes, size_of, name="cache"):
        if capacity_bytes < 0:
            raise ValueError('capacity_bytes must be nonnegative: %r' % (capacity_bytes,))
        if not callable(size_of):
            raise ValueError('size_of must be callable')
        self.capacity_bytes = int(capacity_bytes)
        self.size_of = size_of
        self.name = name
        self.stats = CacheStats(name=name, capacity_bytes=self.capacity_bytes)
        self._entries = OrderedDict()
        self._bytes = 0

    @property
    def enabled(self):
        return self.capacity_bytes > 0

    @property
    def bytes(self):
        return self._bytes

    def __contains__(self, key):
        return key in self._entries

    def get(self, key):
        if key is None or not self.enabled:
            self.stats.misses += 1
            return None
        entry = self._entries.get(key)
        if entry is None:
            self.stats.misses += 1
            return None
        self._entries.move_to_end(key)
        self.stats.hits += 1
        return entry.value

    def put(self, key, value):
        if key is None:
            raise ValueError('Cache key must not be None')
        if not self.enabled:
            self.stats.disabled_puts += 1
            return False
        size = int(self.size_of(value)) + ENTRY_OVERHEAD_BYTES
        if size > self.capacity_bytes:
            self.stats.rejected_too_large += 1
            return False
        if key in self._entries:
            self._bytes -= self._entries[key].size
            del self._entries[key]
        self._entries[key] = _Entry(value=value, size=size)
        self._bytes += size
        while self._bytes > self.capacity_bytes and self._entries:
            _, evicted = self._entries.popitem(last=False)
            self._bytes -= evicted.size
            self.stats.evictions += 1
        self._sync_stats()
        return True

    def clear(self):
        self._entries.clear()
        self._bytes = 0
        self._sync_stats()

    def _sync_stats(self):
        self.stats.entries = len(self._entries)
        self.stats.bytes = self._bytes
        self.stats.peak_bytes = max(self.stats.peak_bytes, self._bytes)

    def snapshot(self):
        self._sync_stats()
        return self.stats.as_dict()


def object_bytes(value):
    if value is None:
        return 0
    if isinstance(value, (bytes, bytearray)):
        return len(value) + 33
    if isinstance(value, str):
        return len(value) * 4 + 49
    numpy_bytes = _numpy_bytes(value)
    if numpy_bytes is not None:
        return numpy_bytes
    torch_bytes = _torch_bytes(value)
    if torch_bytes is not None:
        return torch_bytes
    if isinstance(value, dict):
        return 64 + sum(object_bytes(k) + object_bytes(v) for k, v in value.items())
    if isinstance(value, (list, tuple, set, frozenset)):
        return 56 + sum(object_bytes(item) for item in value)
    return sys.getsizeof(value)


def _numpy_bytes(value):
    try:
        import numpy as np
    except Exception:
        return None
    if isinstance(value, np.ndarray):
        return int(value.nbytes) + 96
    if isinstance(value, np.generic):
        return int(value.nbytes) + 32
    return None


def _torch_bytes(value):
    try:
        import torch
    except Exception:
        return None
    if isinstance(value, torch.Tensor):
        return int(value.element_size() * value.nelement()) + 96
    return None
