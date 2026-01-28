import  numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
import pandas as pd

from models.ngram import Ngram
from utils.decorators import memoize

def make_dataframe(model, fmin=3):
    """
    Створює DataFrame для відображення результатів аналізу.
    
    Args:
        model: Словник моделі з n-грамами
        fmin: Мінімальна частота для включення n-грами в аналіз
        
    Returns:
        pd.DataFrame: DataFrame з результатами
    """
    # Фільтруємо n-грами за мінімальною частотою
    filtered_data = list(
        filter(lambda x: sum(value for value in model[x].values() if isinstance(value, int)) >= fmin, model))
    
    # Додаємо new_ngram, якщо вона існує в моделі
    if 'new_ngram' not in filtered_data and 'new_ngram' in model:
        filtered_data.append("new_ngram")
        
    # Створюємо структуру даних для DataFrame
    data = {"ngram": [],
            "F": np.empty(len(filtered_data), dtype=np.dtype(int))}

    # Заповнюємо дані
    for i, ngram in enumerate(filtered_data):
        data["ngram"].append(ngram)

        if ngram == "new_ngram" and hasattr(model[ngram], 'bool'):
            #data['F'][i] = sum(model[ngram].bool)
            data['F'][i] = np.sum(model[ngram].bool)
        elif ngram == "new_ngram":
            # Якщо атрибут bool відсутній, встановлюємо значення за замовчуванням
            data['F'][i] = 0
        elif hasattr(model[ngram], 'pos'):
            data["F"][i] = len(model[ngram].pos)
        else:
            data["F"][i] = 0

    # Створюємо DataFrame з даних
    dffff = pd.DataFrame(data=data)
    return dffff


@memoize
def build_static_index(data: List, order: int = 1) -> Tuple[Dict[Any, Ngram], int, int]:
    local_model = dict()
    l_val = len(data) - order
    
    local_model['new_ngram'] = Ngram()
    local_model['new_ngram'].bool = np.zeros(l_val, dtype=np.uint8)
    local_model['new_ngram'].pos = []
    
    if order > 1:
        for i in range(l_val - 1):
            window = tuple(data[i: i + order])
            
            if window not in local_model:
                local_model[window] = Ngram()
                local_model[window].pos = []
                local_model[window].bool = np.zeros(l_val, dtype=np.uint8)
                
                local_model['new_ngram'].bool[i] = 1
                local_model['new_ngram'].pos.append(i + 1)
            
            local_model[window].update([data[i + order]])
            local_model[window].pos.append(i + 1)
            local_model[window].bool[i] = 1
    else:
        for i in range(l_val):
            item = data[i]
            next_item = data[i + order]

            if item not in local_model:
                local_model[item] = Ngram()
                local_model[item].pos = []
                local_model[item].bool = np.zeros(l_val, dtype=np.uint8)

                local_model['new_ngram'].pos.append(i + order)
                local_model['new_ngram'].bool[i] = 1

            local_model[item].update([next_item])
            local_model[item].pos.append(i + order)
            local_model[item].bool[i] = 1
        
        if data[l_val] in local_model:
            local_model[data[l_val]].update({data[0]: 1})
        else:
            local_model[data[l_val]] = {data[0]: 1}

        if data[0] in local_model:
            local_model[data[0]].update({data[l_val]: 1})
        else:
            local_model[data[0]] = {data[l_val]: 1}
        
    v_val = len(local_model)
    
    return local_model, l_val, v_val