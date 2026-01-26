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
def build_static_index(data: List, order: int = 1) -> Dict[str, Ngram]:
    """
    Створює ланцюг Маркова з вхідних даних.
    
    Args:
        data: Список елементів для побудови ланцюга Маркова
        order: Порядок ланцюга Маркова (кількість попередніх елементів для прогнозу)
        
    Returns:
        Dict[str, Ngram]: Модель ланцюга Маркова у вигляді словника n-грам
    """
    global model, L, V
    
    # Створюємо новий словник моделі
    model = dict()
    L = len(data) - order
    
    # Ініціалізуємо спеціальну n-граму для нових елементів
    model['new_ngram'] = Ngram()
    model['new_ngram'].bool = np.zeros(L, dtype=np.uint8)  # використовуємо uint8 для зменшення пам'яті
    model['new_ngram'].pos = []
    
    # Використовуємо більш ефективний алгоритм для побудови ланцюга Маркова
    if order > 1:
        for i in range(L - 1):
            window = tuple(data[i: i + order])  # Додаємо в словник
            
            if window in model:  # Приєднуємо до вже існуючого розподілу
                model[window].update([data[i + order]])
                model[window].pos.append(i + 1)
                model[window].bool[i] = 1
            else:
                model[window] = Ngram([data[i + order]])
                model[window].pos = []
                model[window].pos.append(i + 1)
                model[window].bool = np.zeros(L, dtype=np.uint8)
                model[window].bool[i] = 1
                model['new_ngram'].bool[i] = 1
                model['new_ngram'].pos.append(i + 1)
    else:
        # Попередньо визначаємо множину унікальних елементів для оптимізації
        # unique_items = set(data)
        
        # Ініціалізуємо модель для кожного унікального елемента
        # for item in unique_items:
        #     model[item] = Ngram()
        #     model[item].pos = []
        #     model[item].bool = np.zeros(L, dtype=np.uint8)
        
        # Заповнюємо модель
        # for i in range(L):
        #     item = data[i]
        #     next_item = data[i + order]
            
        #     model[item].update([next_item])
        #     model[item].pos.append(i + order)
        #     model[item].bool[i] = 1

        for i in range(L):
            item = data[i]
            next_item = data[i + order]

            if item not in model:
                model[item] = Ngram()
                model[item].pos = []
                model[item].bool = np.zeros(L, dtype=np.uint8)

                model[item].update([next_item])
                model[item].pos.append(i + order)
                model[item].bool[i] = 1

                model['new_ngram'].pos.append(i + order)
                model['new_ngram'].bool[i] = 1

            else:
                model[item].update([next_item])
                model[item].pos.append(i + order)
                model[item].bool[i] = 1
        
        if data[L] in model:
            model[data[L]].update({data[0]: 1})
        else:
            model[data[L]] = {data[0]: 1}

        # Connect the first word with the last one
        if data[0] in model:
            model[data[0]].update({data[L]: 1})
        else:
            model[data[0]] = {data[L]: 1}

        # З'єднуємо останнє слово з першим та перше з останнім
        
        """if data[L] not in model:
            #model[data[L]] = Ngram()
            #model[data[L]].pos = []
            #model[data[L]].pos.append(L + order)
            #model[data[L]].bool = np.zeros(L, dtype=np.uint8)
            #model[data[L]].bool[L-1] = 1
            model[data[L]].update([data[0]])
            model['new_ngram'].pos.append(L + order)
            model['new_ngram'].bool[L-1] = 1
        else:
            model[data[L]].pos.append(L + order)
            model[data[L]].bool = np.zeros(L, dtype=np.uint8)
            model[data[L]].bool[L-1] = 1
        
        model[data[0]].update([data[L]])"""

    #print(sum(model['new_ngram'].bool))
        
    V = len(model)
    return model