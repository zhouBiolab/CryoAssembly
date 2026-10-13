
import logging
import queue
import threading

log = logging.getLogger(__name__)

_SENTINEL = object()


class TailPipeline:

    def __init__(self, enabled=True, name="dsh-tail"):
        self.enabled = bool(enabled)
        self.name = name
        self.submitted = 0
        self.completed = 0
        self._error = None
        self._closed = False
        if not self.enabled:
            self._queue = None
            self._worker = None
            return
        self._queue = queue.Queue(maxsize=1)
        self._worker = threading.Thread(target=self._run, name=name, daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------
    def submit(self, function):
        if self._error is not None:
            raise self._error
        self.submitted += 1
        if not self.enabled:
            self.completed += 1
            return function()
        self._queue.put(function)
        return None

    def wait(self):
        if self.enabled and self._worker is not None:
            self._queue.join()
        if self._error is not None:
            raise self._error

    def close(self):
        if self._closed:
            return
        self._closed = True
        if not self.enabled or self._worker is None:
            if self._error is not None:
                raise self._error
            return
        self._queue.join()
        self._queue.put(_SENTINEL)
        self._worker.join(timeout=30.0)
        if self._worker.is_alive():
            raise RuntimeError('Tail pipeline worker did not exit before timeout; a task may be stuck')
        if self._error is not None:
            raise self._error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.close()
        except Exception as close_error:
            if exc_type is None:
                raise
            log.error('Tail pipeline shutdown failed: %s', close_error)
        return False

    def snapshot(self):
        return {"enabled": self.enabled, "submitted": self.submitted,
                "completed": self.completed, "failed": self._error is not None}

    # ------------------------------------------------------------------
    def _run(self):
        while True:
            function = self._queue.get()
            try:
                if function is _SENTINEL:
                    return
                function()
                self.completed += 1
            except Exception as error:
                if self._error is None:
                    self._error = error
                log.error('Tail task failed: %s', error)
            finally:
                self._queue.task_done()
