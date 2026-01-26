import numpy as np
from numba import jit, njit, prange

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


@jit(nopython=True, fastmath=True)
def fit(x, a, b):
    return a * (x ** b)

def is_number(s: str) -> bool:
    """
    Перевіряє, чи можна рядок перетворити в число.
    
    Args:
        s: Рядок для перевірки
        
    Returns:
        bool: True, якщо рядок може бути перетворений у число, інакше False
    """
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False
