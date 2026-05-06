from typing import List, Tuple, Optional
import numpy as np
import numpy.polynomial.polynomial as poly

from core.math_utils import calculate_window_positions



def dfa(data: List, args: Tuple[int, int, int],
        overlap_mode: str = "overlapping",
        min_window: Optional[int] = None,
        window_expansion: Optional[int] = None,
        polynom_degree: int = 1,
        raw: bool = False):
    """
    Виконує DETRENDED аналіз флуктуацій (DFA) для даних.

    Args:
        data: Вхідні дані для аналізу
        args: Кортеж (розмір вікна, зсув вікна, довжина даних)
        overlap_mode: Режим перекриття вікон
        min_window: Мінімальний розмір вікна
        window_expansion: Значення розширення вікна
        polynom_degree: Степінь полінома для детрендінгу
        raw: Якщо True — використовує float значення напряму (для float режиму)

    Returns:
        Tuple[np.ndarray, float]: Масив результатів та значення DFA
    """
    wi, wh, l = args

    if raw:
        # Float режим: використовуємо значення напряму
        novelty = np.asarray(data, dtype=np.float64)
    else:
        # Побудова глобального бінарного ряду новизни
        seen = set()
        novelty = []
        for ngram in data:
            is_new = ngram not in seen
            novelty.append(1.0 if is_new else 0.0)
            if is_new:
                seen.add(ngram)
        novelty = np.asarray(novelty, dtype=np.float64)

    # Інтегрований центрований ряд
    y_full = np.cumsum(novelty - np.mean(novelty))

    window_positions = calculate_window_positions(wi, wh, l, overlap_mode, min_window, window_expansion)
    count = np.zeros(len(window_positions), dtype=np.float64)
    scale_ax = np.arange(wi)

    for index, i in enumerate(window_positions):
        y_win = y_full[i:i + wi]
        coefs = poly.polyfit(scale_ax, y_win, polynom_degree)
        xfit = poly.polyval(scale_ax, coefs)
        count[index] = np.sqrt(np.mean((y_win - xfit) ** 2))

    dfa_value = np.sqrt(np.mean(count ** 2))
    return count, dfa_value


def calculate_rmsd(x, wi, polynom_degree):
    """
    Обчислює RMSD (Root Mean Square Deviation) з детрендінгом.
    
    Args:
        x: Вхідний масив даних
        wi: Розмір вікна
        polynom_degree: Степінь полінома для детрендінгу
        
    Returns:
        Tuple[np.ndarray, float]: RMS масив та DFA значення
    """
    i = int(polynom_degree)

    y = np.cumsum(x - np.mean(x))

    shape = (y.shape[0] // wi, wi)
    X = np.lib.stride_tricks.as_strided(y, shape=shape)
    scale_ax = np.arange(wi)
    rms = np.zeros(X.shape[0], dtype=np.float64)
    for e, xcut in enumerate(X):
        coeffs = np.polyfit(scale_ax, xcut, i)
        trend = np.polyval(coeffs, scale_ax)
        resid = xcut - trend
        rms[e] = np.sqrt(np.mean(resid ** 2))

    dfa_value = np.sqrt(np.mean(np.square(rms)))
    return rms, dfa_value