import numpy as np
from numba import jit, njit, prange
from typing import List, Tuple, Optional, Dict, Any, Union

from math_utils import calc_non_overlapping_shift, make_windows, s
from statistics import mse


def calculate_window_positions(wi, wh, l, overlap_mode, min_window, window_expansion):

    if overlap_mode == "overlapping":
        window_positions = range(0, l - wi, wh)
    else:
        # Non-overlapping режим
        if min_window is None:
            min_window = wh
        if window_expansion is None:
            window_expansion = wh

        # Оцінюємо кількість і розташування вікон
        k = 1
        i = 0
        window_positions = []
        while i < l - wi:
            window_positions.append(i)
            shift = calc_non_overlapping_shift(k, min_window, window_expansion)
            i += shift
            k += 1

    return list(window_positions)

def calculate_rms(x, wi, l, wsh, overlap_mode, min_window, window_expansion):
    if overlap_mode == "overlapping":
        # print('overlapping')
        # print('  model[ngram].bool, wi=wind, l=L, wsh=w_s_val, overlap_mode=overlap_mode', model[ngram].bool, wind, L, w_s_val, overlap_mode)
        rms = make_windows(x, wi=wi, l=l, wsh=wsh, overlap_mode=overlap_mode)
    else:
        # print('NOT overlapping')
        # print('  model[ngram].bool, wi=wind, l=L, wsh=w_s_val, overlap_mode=overlap_mode min_window=w_s_val, window_expansion=w_e_val', model[ngram].bool, wind, L, w_s_val, overlap_mode, w_s_val, w_e_val)
        rms = make_windows(x, wi=wi, l=l, wsh=wsh, overlap_mode=overlap_mode, min_window=min_window, window_expansion=window_expansion)

    fa = mse(rms)
    return rms, fa


def fa(data: List, args: Tuple[int, int, int],
       overlap_mode: str = "overlapping", 
       min_window: Optional[int] = None, 
       window_expansion: Optional[int] = None):
    """
    Виконує аналіз флуктуацій (DFA) для даних.

    Args:
        data: Вхідні дані для аналізу
        args: Кортеж (розмір вікна, зсув вікна, довжина даних)
        overlap_mode: Режим перекриття вікон ("overlapping" або "non-overlapping")
        min_window: Мінімальний розмір вікна для режиму non-overlapping
        window_expansion: Значення розширення вікна для режиму non-overlapping
        
    Returns:
        np.ndarray: Масив результатів DFA аналізу
    """
    wi, wh, l = args

    window_positions = calculate_window_positions(wi, wh, l, overlap_mode, min_window, window_expansion)
    count = np.zeros(len(window_positions), dtype=np.uint16)

    for index, i in enumerate(window_positions):
        temp_v = []
        x = []
        for ngram in data[i:i + wi]:
            if ngram in temp_v:
                x.append(0)
            else:
                temp_v.append(ngram)
                x.append(1)
        count[index] = s(np.array(x, dtype=np.uint8))

    fa = float(mse(count))

    return count, fa