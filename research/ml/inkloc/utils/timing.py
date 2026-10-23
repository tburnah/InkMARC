from __future__ import annotations

import time
from contextlib import contextmanager


@contextmanager
def timer(name: str):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"{name}: {elapsed:.3f}s")
