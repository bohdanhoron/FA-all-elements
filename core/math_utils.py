import numpy as np
from numba import jit, njit, prange
from typing import List, Tuple, Optional, Dict, Any, Union

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


@jit(nopython=True)
def s(window: np.ndarray) -> int:
    """
    Обчислює суму значень вікна.
    
    Args:
        window: Масив значень
        
    Returns:
        int: Сума значень
    """
    # Використовуємо оптимізовану NumPy функцію
    return np.sum(window)


@njit(fastmath=True)
def calc_non_overlapping_shift(k, min_window, window_expansion):
    """
    Розраховує зміщення для режиму non-overlapping
    k - номер кроку (починаючи з 1)
    """
    # Numba не працює з None значеннями, тому перевірка робиться в make_windows
    if k == 1:
        return min_window
    else:
        return min_window + (k-1) * window_expansion

@njit(fastmath=True)
def make_windows(x: np.ndarray, wi: int, l: int, wsh: int, 
                overlap_mode: str = "overlapping", 
                min_window: Optional[int] = None, 
                window_expansion: Optional[int] = None) -> np.ndarray:
    """
    Створює вікна для аналізу даних.
    
    Args:
        x: Вхідний масив даних
        wi: Розмір вікна
        l: Довжина даних
        wsh: Величина зсуву вікна
        overlap_mode: Режим перекриття вікон ("overlapping" або "non-overlapping")
        min_window: Мінімальний розмір вікна для режиму non-overlapping
        window_expansion: Значення розширення вікна для режиму non-overlapping
        
    Returns:
        np.ndarray: Масив сум у вікнах
    """
    # Використовуємо Numba для оптимізації
    if overlap_mode == "overlapping":
        # Визначаємо кількість вікон заздалегідь для уникнення повторного обчислення
        num_windows = (l - wi) // wsh + 1
        sums = np.zeros(num_windows, dtype=np.float64)
        
        # Використовуємо ефективніший цикл
        for i in range(num_windows):
            start_idx = i * wsh
            end_idx = start_idx + wi
            # Використовуємо вбудовану функцію sum у NumPy
            sums[i] = np.sum(x[start_idx:end_idx])
            
    else:  # non-overlapping режим
        # Використовуємо правильні значення за замовчуванням
        min_win = wi if min_window is None else min_window
        win_exp = wi if window_expansion is None else window_expansion
        
        # Визначаємо кількість вікон
        num_windows = (l - wi) // wi + 1
        sums = np.zeros(num_windows, dtype=np.float64)
        
        # Використовуємо ефективніший цикл для non-overlapping
        for i in range(num_windows):
            start_idx = i * wi
            end_idx = start_idx + wi
            #NOTE виходить, що останнє вікно може бути меншим, ніж інші, цього бажано позбутися
            if end_idx > l:
                end_idx = l
            sums[i] = np.sum(x[start_idx:end_idx])
    
    return sums


@njit(fastmath=True)
def calc_sum(x):
    sums = np.empty(len(x))
    for i, w in enumerate(x):
        sums[i] = np.sum(w)
    return sums
