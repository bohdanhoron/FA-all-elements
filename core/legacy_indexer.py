from typing import Dict, Tuple
import numpy as np
import pandas as pd

from models.state_manager import state
from models.ngram import Ngram
from utils.decorators import memoize


@memoize
def make_markov_chain(data: Tuple, order: int = 1) -> Dict[str, Ngram]:
    """
    Створює ланцюг Маркова з вхідних даних.
    
    Args:
        data: Tuple елементів для побудови ланцюга Маркова
        order: Порядок ланцюга Маркова (кількість попередніх елементів для прогнозу)
        
    Returns:
        Dict[str, Ngram]: Модель ланцюга Маркова у вигляді словника n-грам
    """
    model = dict()
    L = len(data) - order
    
    model['new_ngram'] = Ngram()
    model['new_ngram'].bool = np.zeros(L, dtype=np.uint8)
    model['new_ngram'].pos = []
    
    if order > 1:
        for i in range(L - 1):
            window = tuple(data[i: i + order])
            next_item = data[i + order]
            
            if window in model:
                if next_item in model[window]:
                    model[window][next_item] += 1
                else:
                    model[window][next_item] = 1
                model[window].pos.append(i + 1)
                model[window].bool[i] = 1
            else:
                model[window] = Ngram()
                model[window][next_item] = 1
                model[window].pos = []
                model[window].pos.append(i + 1)
                model[window].bool = np.zeros(L, dtype=np.uint8)
                model[window].bool[i] = 1
                model['new_ngram'].bool[i] = 1
                model['new_ngram'].pos.append(i + 1)
    else:
        for i in range(L):
            item = data[i]
            next_item = data[i + order]

            if item not in model:
                model[item] = Ngram()
                model[item].pos = []
                model[item].bool = np.zeros(L, dtype=np.uint8)
                
                model[item][next_item] = 1
                model[item].pos.append(i + order)
                model[item].bool[i] = 1

                model['new_ngram'].pos.append(i + order)
                model['new_ngram'].bool[i] = 1
            else:
                if next_item in model[item]:
                    model[item][next_item] += 1
                else:
                    model[item][next_item] = 1
                model[item].pos.append(i + order)
                model[item].bool[i] = 1
        
        last_item = data[L]
        first_item = data[0]
        
        if last_item in model:
            if first_item in model[last_item]:
                model[last_item][first_item] += 1
            else:
                model[last_item][first_item] = 1
        else:
            model[last_item] = Ngram()
            model[last_item][first_item] = 1
            model[last_item].pos = []
            model[last_item].bool = np.zeros(L, dtype=np.uint8)

        if first_item in model:
            if last_item in model[first_item]:
                model[first_item][last_item] += 1
            else:
                model[first_item][last_item] = 1

    V = len(model)
    
    state.model = model
    state.L = L
    state.V = V
    
    return model


def make_dataframe(model: Dict, fmin: int = 3) -> pd.DataFrame:
    """
    Створює DataFrame для відображення результатів аналізу.
    
    Args:
        model: Словник моделі з n-грамами
        fmin: Мінімальна частота для включення n-грами в аналіз
        
    Returns:
        pd.DataFrame: DataFrame з результатами
    """
    filtered_data = []
    for ngram in model:
        if ngram == 'new_ngram':
            continue
        total_count = sum(value for value in model[ngram].values() if isinstance(value, int))
        if total_count >= fmin:
            filtered_data.append(ngram)
    
    if 'new_ngram' in model:
        filtered_data.append("new_ngram")
        
    data = {"ngram": [],
            "F": np.empty(len(filtered_data), dtype=np.dtype(int))}

    for i, ngram in enumerate(filtered_data):
        data["ngram"].append(ngram)

        if ngram == "new_ngram":
            if hasattr(model[ngram], 'bool') and model[ngram].bool is not None:
                data['F'][i] = int(np.sum(model[ngram].bool))
            elif hasattr(model[ngram], 'pos'):
                data['F'][i] = len(model[ngram].pos)
            else:
                data['F'][i] = 0
        elif hasattr(model[ngram], 'pos'):
            data["F"][i] = len(model[ngram].pos)
        else:
            data["F"][i] = 0

    df = pd.DataFrame(data=data)
    return df