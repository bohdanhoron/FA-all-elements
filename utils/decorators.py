from functools import wraps
from typing import Callable, Any


def memoize(func: Callable) -> Callable:
    cache = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    wrapper.clear_cache = lambda: cache.clear()
    return wrapper