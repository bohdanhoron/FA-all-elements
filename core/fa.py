from typing import List, Tuple, Optional
import numpy as np

from core.math_utils import calculate_window_positions, s, make_windows
from core.statistics import rmse


def fa(data: List, args: Tuple[int, int, int],
       overlap_mode: str = "overlapping",
       min_window: Optional[int] = None,
       window_expansion: Optional[int] = None,
       raw: bool = False):
    """
    Виконує аналіз флуктуацій (FA) для даних.

    Args:
        data: Вхідні дані для аналізу
        args: Кортеж (розмір вікна, зсув вікна, довжина даних)
        overlap_mode: Режим перекриття вікон ("overlapping" або "non-overlapping")
        min_window: Мінімальний розмір вікна для режиму non-overlapping
        window_expansion: Значення розширення вікна для режиму non-overlapping
        raw: Якщо True — використовує float значення напряму (для float режиму)

    Returns:
        Tuple[np.ndarray, float]: Масив результатів та значення FA
    """
    wi, wh, l = args

    window_positions = calculate_window_positions(wi, wh, l, overlap_mode, min_window, window_expansion)

    if raw:
        y_full = np.cumsum(np.asarray(data, dtype=np.float64) - np.mean(data))
        count = np.zeros(len(window_positions), dtype=np.float64)
        for index, i in enumerate(window_positions):
            y_win = y_full[i:i + wi]
            count[index] = np.sqrt(np.mean((y_win - np.mean(y_win)) ** 2))
    else:
        seen = set()
        novelty = []
        for ngram in data:
            is_new = ngram not in seen
            novelty.append(1 if is_new else 0)
            if is_new:
                seen.add(ngram)
        novelty = np.asarray(novelty, dtype=np.uint8)
        count = np.zeros(len(window_positions), dtype=np.uint16)
        for index, i in enumerate(window_positions):
            x = novelty[i:i + wi]
            count[index] = s(x)

    fa_value = float(rmse(count))
    return count, fa_value


def calculate_rms(x, wi, l, wsh, overlap_mode, min_window, window_expansion):
    """
    Обчислює RMS (Root Mean Square) для заданих параметрів вікна.
    
    Args:
        x: Вхідний масив даних
        wi: Розмір вікна
        l: Довжина даних
        wsh: Зсув вікна
        overlap_mode: Режим перекриття
        min_window: Мінімальний розмір вікна
        window_expansion: Розширення вікна
        
    Returns:
        Tuple[np.ndarray, float]: RMS масив та FA значення
    """
    if overlap_mode == "overlapping":
        rms = make_windows(x, wi=wi, l=l, wsh=wsh, overlap_mode=overlap_mode)
    else:
        rms = make_windows(x, wi=wi, l=l, wsh=wsh, overlap_mode=overlap_mode, 
                          min_window=min_window, window_expansion=window_expansion)

    fa_value = rmse(rms)
    return rms, fa_value