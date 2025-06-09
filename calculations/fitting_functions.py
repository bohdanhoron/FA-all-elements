from numba import jit

@jit(nopython=True, fastmath=True)
def power_law(x, a, b):
    return a * (x ** b)

