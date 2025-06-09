import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

#from bc import calculate_distance
from .stats import mse
from app import make_windows#, fit
from .fitting_functions import power_law

def FA(ngram, L, windows, wh_val, overlap_mode, w_val, we_val):

    # Process windows
    ngram.fa = {}
    ngram.counts = {}
    
    for wind in windows:
        if overlap_mode == "overlapping":
            ngram.counts[wind] = make_windows(ngram.bool, wi=wind, l=L, wsh=wh_val, overlap_mode=overlap_mode)
        else:
            ngram.counts[wind] = make_windows(ngram.bool, wi=wind, l=L, wsh=wh_val,
                                              overlap_mode=overlap_mode, min_window=w_val, window_expansion=we_val)
        ngram.fa[wind] = mse(ngram.counts[wind])
    
    ff = [ngram.fa[wind] for wind in windows]
    
    try:
        c, _ = curve_fit(power_law, windows, ff, method='lm', maxfev=5000)
        
        a = round(c[0], 8)
        gamma = round(c[1], 8)
        fa = [power_law(w_val, c[0], c[1]) for w_val in windows]
        err = round(r2_score(ff, fa), 5)
    except Exception as e:
        print(f"{e}")
        # Handle curve fitting errors
        a = 0
        gamma = 0
        err = 0

    return a, gamma, fa, err
