import numpy as np
import pandas as pd
from numba import jit, njit


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

    mean_x = np.mean(x)
    if mean_x == 0: 
        return 0.0
    std_x = np.std(x)
    return std_x / mean_x

def add_batch_statistics(results):
    """Adds mean and standard deviation rows to batch results."""
    if not results:
        return
    
    data_for_stats = []
    numeric_fields = ["length", "vocabulary", "time", "r_avg", "dr", "rw_avg", "drw", 
                    "gamma_avg", "dgamma", "gammaw_avg", "dgammaw"]
    
    for item in results:
        if item["filename"] in ["MEAN", "STDDEV"]:
            continue
        
        data_point = {}
        for field in numeric_fields:
            if field in item:
                data_point[field] = item[field]
        
        data_for_stats.append(data_point)
    
    if not data_for_stats:
        return
        
    df_stats = pd.DataFrame(data_for_stats)
    
    means = {
        "no": len(results) + 1,
        "filename": "MEAN",
        "f_min": "-",
    }
    
    stddevs = {
        "no": len(results) + 2,
        "filename": "STDDEV",
        "f_min": "-",
    }
    
    for field in numeric_fields:
        if field in df_stats.columns:
            means[field] = round(df_stats[field].mean(), 
                               8 if field in ["r_avg", "dr", "rw_avg", "drw", "gamma_avg", "dgamma", "gammaw_avg", "dgammaw"] else 
                               3 if field == "time" else 0)
            
            stddevs[field] = round(df_stats[field].std(), 
                                 8 if field in ["r_avg", "dr", "rw_avg", "drw", "gamma_avg", "dgamma", "gammaw_avg", "dgammaw"] else 
                                 3 if field == "time" else 0)
    
    results[:] = [r for r in results if r["filename"] not in ["MEAN", "STDDEV"]]
    
    results.append(means)
    results.append(stddevs)

