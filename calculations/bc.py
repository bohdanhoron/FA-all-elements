import numpy as np
from numba import jit, njit, prange

def calculate_distance(positions: np.ndarray, L: int, option: str, ngram: str, min_dist: int = 1) -> np.ndarray:
    """
    Розраховує відстані між позиціями елементів з урахуванням граничних умов.
    
    Оптимізована для роботи з великими наборами даних за допомогою паралельної обробки.
    
    Args:
        positions: Масив позицій елементів
        L: Довжина тексту
        option: Тип граничних умов ("no", "ordinary", "periodic")
        ngram: Назва n-грами
        min_dist: Мінімальна відстань (0 або 1)
        
    Returns:
        np.ndarray: Масив відстаней між елементами
    """

#    print(f"calculating distances... option {option}, min_dist={min_dist}")

    # Оптимізуємо обробку масиву позицій
    positions = np.array(positions, dtype=np.int32)
    
    # Переконуємося, що min_dist є цілим числом
    if not isinstance(min_dist, int):
        try:
            min_dist = int(min_dist)
        except (ValueError, TypeError):
            print(f"Warning: min_dist '{min_dist}' is not an integer. Using default min_dist=1")
            min_dist = 1
    
    # Використовуємо оптимізовані функції відповідно до граничних умов
    if option == "no":
        distances = nbc(positions, L, min_dist)
    elif option == "periodic":
        distances = pbc(positions, L, min_dist)
    else:  # "ordinary"
        distances = obc(positions, L, min_dist)
    
    return distances


@njit(parallel=True)
def nbc(pos, L, min_dist=1):
    """
    Обчислює відстані без граничних умов.
    
    Оптимізовано за допомогою Numba JIT з паралельною обробкою.
    
    Args:
        pos: Масив позицій елементів
        L: Довжина послідовності
        min_dist: Мінімальна відстань
        
    Returns:
        np.ndarray: Масив відстаней
    """
    n = len(pos)
    dt = np.zeros(n - 1, dtype=np.int32)
    
    for i in prange(n - 1):
        dt[i] = pos[i + 1] - pos[i]
        if min_dist==0:
            dt[i] -= 1
    
    return dt


@njit(parallel=True)
def pbc(pos, L, min_dist=1):
    """
    Обчислює відстані з періодичними граничними умовами.
    
    Оптимізовано за допомогою Numba JIT з паралельною обробкою.
    
    Args:
        pos: Масив позицій елементів
        L: Довжина послідовності
        min_dist: Мінімальна відстань
        
    Returns:
        np.ndarray: Масив відстаней
    """
    n = len(pos)
    dt = np.zeros(n, dtype=np.int32)
    
    for i in prange(n - 1):
        dt[i] = pos[i + 1] - pos[i]
        if min_dist==0:
            dt[i] -= 1
    
    # Останній елемент обчислюємо окремо через періодичність
    dt[n - 1] = L - pos[n - 1] + pos[0]
    if min_dist==0:
        dt[n - 1] -= 1
    
    return dt


@njit(parallel=True)
def obc(pos, L, min_dist=1):
    """
    Обчислює відстані зі звичайними граничними умовами.
    
    Оптимізовано за допомогою Numba JIT з паралельною обробкою.
    
    Args:
        pos: Масив позицій елементів
        L: Довжина послідовності
        min_dist: Мінімальна відстань
        
    Returns:
        np.ndarray: Масив відстаней
    """
    n = len(pos)
    dt = np.zeros(n, dtype=np.int32)
    
    dt[0] = pos[0]
    if min_dist==0:
        dt[0] -= 1
    
    for i in prange(1, n - 1):
        dt[i] = pos[i + 1] - pos[i]
        if min_dist==0:
            dt[i] -= 1
    
    # Останній елемент обчислюємо окремо
    dt[n - 1] = L - pos[n - 1]
    if min_dist==0:
        dt[n - 1] -= 1
    
    return dt

