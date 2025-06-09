import numpy as np
from numba import jit, njit

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
def mse(x: np.ndarray) -> float:
    """
    Обчислює середньоквадратичну похибку (MSE) набору значень.
    
    Args:
        x: Масив значень
        
    Returns:
        float: Значення MSE
    """
    if len(x) == 0:
        return 0.0
        
    # Оптимізоване обчислення MSE
    mean_x = np.mean(x)
    return np.sqrt(np.mean((x - mean_x) ** 2))


@jit(nopython=True, fastmath=True)
def R(x: np.ndarray) -> float:
    """
    Обчислює коефіцієнт варіації.
    
    Args:
        x: Масив значень
        
    Returns:
        float: Значення коефіцієнта варіації
    """
    if len(x) <= 1:
        return 0.0
        
    # Оптимізоване обчислення коефіцієнта варіації
    mean_x = np.mean(x)
    if mean_x == 0:  # Запобігаємо діленню на нуль
        return 0.0
    std_x = np.std(x)
    return std_x / mean_x
