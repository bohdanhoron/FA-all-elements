import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy.polynomial.polynomial as poly

from fa import calculate_window_positions


def dfa(data: List, args: Tuple[int, int, int],
       overlap_mode: str = "overlapping",
       min_window: Optional[int] = None,
       window_expansion: Optional[int] = None,
       polynom_degree = 1):
    """
    Виконує DETRENDED аналіз флуктуацій (DFAD) для даних.

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

        x = np.array(x)

        ### TODO: find if we need it
        y = np.cumsum(x - np.mean(x))
        #y = x

        scale_ax = np.arange(wi)

        coefs = poly.polyfit(scale_ax, y, polynom_degree)
        xfit = poly.polyval(scale_ax, coefs)

        # count[index] = s(np.array(x, dtype=np.uint8))
        count[index] = np.sqrt(np.mean((y - xfit) ** 2))

    # fa = float(mse(count))
    fa = np.mean(np.sqrt(count ** 2))

    return count, fa

def calculate_rmsd(x, wi, polynom_degree):
    i = int(polynom_degree)

    ### TODO: find if we need it
    y = np.cumsum(x - np.mean(x))
    #y = x

    # making an array with data divided in windows
    shape = (y.shape[0] // wi, wi)
    X = np.lib.stride_tricks.as_strided(y, shape=shape)
    # vector of x-axis points to regression
    scale_ax = np.arange(wi)
    rms = np.zeros(X.shape[0])
    for e, xcut in enumerate(X):
        coeffs = np.polyfit(scale_ax, xcut, i)
        trend = np.polyval(coeffs, scale_ax)
        # detrending and computing RMS of each window
        resid = xcut - trend
        rms[e] = np.sqrt(np.mean(resid ** 2))

    #fa = np.mean(np.sqrt(np.square(rms)))
    fa = np.sqrt(np.mean(np.square(rms)))
    return rms, fa
