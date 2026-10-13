
from protassem.runtime.pool import close_pool, open_pool


class ExecutionContext:
    'Shared process pool and task execution state.'

    def __init__(self, pool_workers=1, start_method=None):
        self.pool_workers = max(1, int(pool_workers or 1))
        self.start_method = start_method
        self._pool = None

    def map(self, func, items):
        items = list(items)
        if not items:
            return []
        if self.pool_workers <= 1:
            return [func(item) for item in items]
        return self._shared_pool().map(func, items)

    def close(self):
        if self._pool is None:
            return
        close_pool(self._pool)
        self._pool = None

    def _shared_pool(self):
        if self._pool is None:
            self._pool = open_pool(self.pool_workers, self.start_method)
        return self._pool

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
