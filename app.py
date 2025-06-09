from typing import List, Tuple, Optional, Dict, Any, Union
import gc  # Garbage Collector для кращого управління пам'яттю

import numpy as np
from numba import jit, njit, prange
import pandas as pd

# Обробка даних і тексту
import re
from string import punctuation
from time import time

# Dash і візуалізація
import dash
import dash_bootstrap_components as dbc

# Системні і допоміжні бібліотеки
import webbrowser

# tkinter for selecting browsing folder
import tkinter as tk
from tkinter import filedialog

#Code Tokenizer
from processing.ngrams import Ngram, newNgram

from callbacks import *

# Функція для очищення пам'яті
def clear_memory(keep: List[str] = []):
    """
    Очищує пам'ять від великих структур даних, які більше не потрібні.
    
    Args:
        keep: Список назв змінних, які потрібно зберегти
    """
    global model, df, new_ngram, data, uploaded_files, file_lengths, batch_results
    
    # Зберігаємо лише необхідні дані для таблиці
    variables_to_keep = keep + ['uploaded_files', 'file_lengths', 'batch_results']
    
    # Очищення великих глобальних структур даних
    if 'model' not in variables_to_keep and 'model' in globals():
        if isinstance(model, dict):
            model.clear()
        model = {}
    
    # Очищення DataFrame
    if 'df' not in variables_to_keep and 'df' in globals() and df is not None:
        df = None
    
    # Очищення даних тексту
    if 'data' not in variables_to_keep and 'data' in globals() and data is not None:
        data = None
    
    # Очищення об'єкта newNgram
    if 'new_ngram' not in variables_to_keep and 'new_ngram' in globals() and new_ngram is not None:
        new_ngram = None
    
    # Очищення кешу мемоізованих функцій
    if hasattr(prepare_data, 'clear_cache') and 'prepare_data_cache' not in variables_to_keep:
        prepare_data.clear_cache()
    
    if hasattr(make_markov_chain, 'clear_cache') and 'make_markov_chain_cache' not in variables_to_keep:
        make_markov_chain.clear_cache()
    
    # Додаємо агресивне очищення пам'яті за допомогою Python gc
    import gc
    gc.collect(generation=2) # Запуск повного збирання сміття
    gc.collect(generation=1)
    gc.collect(generation=0)

# Кешування для покращення продуктивності
def memoize(func):
    """
    Декоратор для кешування результатів функцій, щоб уникнути повторних обчислень.
    """
    cache = {}
    
    def wrapper(*args, **kwargs):
        # Створюємо унікальний ключ на основі аргументів
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    
    # Додаємо функцію для очищення кешу
    wrapper.clear_cache = lambda: cache.clear()
    return wrapper


#def remove_punctuation_for_words(data):
    """
    Розбиває текст на слова та видаляє знаки пунктуації.
    
    Args:
        data: Вхідний текст
        
    Returns:
        List[str]: Список оброблених слів
    """
    # Використовуємо ефективніший регулярний вираз один раз
    # words = re.findall(r'\b[a-zA-Z0-9]+(?:[-\'][a-zA-Z0-9]+)*\b', data.lower())
    
    # Обробляємо слова з дефісами та апострофами
    """result = []
    for word in words:
        if '-' in word or '\'' in word:
            # Розділяємо слово на підчастини за спеціальними символами
            parts = re.split(r'[-\']', word)
            # Додаємо лише непорожні частини
            result.extend([part for part in parts if part])
        else:
            result.append(word)"""
    
    #return result
    #return words


def remove_punctuation(data):
    """
    Видаляє знаки пунктуації з тексту.
    """
    temp = []
    for i in range(len(data)):
        if data[i] in punctuation:
            continue
        else:
            temp.append(data[i].lower())
    return "".join(temp)




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
def make_markov_chain(data: List, order: int = 1) -> Dict[str, Ngram]:
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


#@njit(fastmath=True)
#def calc_non_overlapping_shift(k, min_window, window_expansion):
#    """
#    Розраховує зміщення для режиму non-overlapping
#    k - номер кроку (починаючи з 1)
#    """
    # Numba не працює з None значеннями, тому перевірка робиться в make_windows
#    if k == 1:
#        return min_window
#    else:
#        return min_window + (k-1) * window_expansion

@njit(fastmath=True)
def make_windows(x: np.ndarray, wi: int, l: int, wsh: int, 
                overlap_mode: str = "overlapping", 
                min_window: Optional[int] = None, 
                window_expansion: Optional[int] = None) -> np.ndarray:
    """
    Створює вікна для аналізу даних.
    
    Args:
        x: Вхідний масив даних
        wi: Розмір вікна
        l: Довжина даних
        wsh: Величина зсуву вікна
        overlap_mode: Режим перекриття вікон ("overlapping" або "non-overlapping")
        min_window: Мінімальний розмір вікна для режиму non-overlapping
        window_expansion: Значення розширення вікна для режиму non-overlapping
        
    Returns:
        np.ndarray: Масив сум у вікнах
    """
    # Використовуємо Numba для оптимізації
    if overlap_mode == "overlapping":
        # Визначаємо кількість вікон заздалегідь для уникнення повторного обчислення
        num_windows = (l - wi) // wsh + 1
        sums = np.zeros(num_windows, dtype=np.float64)
        
        # Використовуємо ефективніший цикл
        for i in range(num_windows):
            start_idx = i * wsh
            end_idx = start_idx + wi
            # Використовуємо вбудовану функцію sum у NumPy
            sums[i] = np.sum(x[start_idx:end_idx])
            
    else:  # non-overlapping режим
        # Використовуємо правильні значення за замовчуванням
        min_win = wi if min_window is None else min_window
        win_exp = wi if window_expansion is None else window_expansion
        
        # Визначаємо кількість вікон
        num_windows = (l - wi) // wi + 1
        sums = np.zeros(num_windows, dtype=np.float64)
        
        # Використовуємо ефективніший цикл для non-overlapping
        for i in range(num_windows):
            start_idx = i * wi
            end_idx = start_idx + wi
            #NOTE виходить, що останнє вікно може бути меншим, ніж інші, цього бажано позбутися
            if end_idx > l:
                end_idx = l
            sums[i] = np.sum(x[start_idx:end_idx])
    
    return sums


"""@njit(fastmath=True)
def calc_sum(x):
    sums = np.empty(len(x))
    for i, w in enumerate(x):
        sums[i] = np.sum(w)
    return sums"""



@memoize
def prepare_data(data: str, n: int, split: str, filename: str, computer_code: bool = False, ignore_comments: bool = False) -> List:
    """
    Підготовка даних для аналізу, розбиття на n-грами залежно від вказаних параметрів.
    
    Args:
        data: Вхідний текст для обробки
        n: Розмір n-грами
        split: Метод розбиття тексту ("word", "letter", "symbol")
        
    Returns:
        List: Список підготовлених даних
    """
    global L
    if n is None:
        return dash.no_update
    
    # Використовуємо спільний код попередньої обробки для всіх типів
    if not computer_code:
        data = re.sub(r'\n+', '\n', data)
        data = re.sub(r'\n\s\s', '\n', data)
        data = re.sub(r'﻿', '', data)
    
    # Для n=1 (одиничні елементи)
    if n == 1:
        if split == "word":
            # Обробка тексту для слів
            if not computer_code:
                data = re.sub(r'--', ' -', data)
            processor = NgrammProcessor(computer_code=computer_code, ignore_comments=ignore_comments)
            processor.preprocess(data, file_name=filename)
            result = processor.get_words()
            L = len(result)
            return result
            
        elif split == 'letter':
            # Обробка для літер і чисел
            temp = []
            data = remove_punctuation(data)
            for word in data:
                for i in word:
                    if i == ' ':
                        continue
                    temp.append(i)
            L = len(temp)
            return temp
            
        elif split == 'symbol':
            # Обробка для символів
            result = []
            for char in data:
                if char == " " or char == "\n" or char == "\ufeff":
                    result.append("space")
                else:
                    result.append(char.lower())
            L = len(result)
            return result
    
    # Для n>1 (n-грами)
    else:
        if split == "word":
            # Обробка для n-грам слів
            data = re.sub(r'--', ' -', data)
            processor = NgrammProcessor()
            processor.preprocess(data)
            words = processor.get_words()
            L = len(words)
            
            # Створюємо n-грами з слів
            result = []
            for i in range(L - n + 1):
                window = tuple(words[i:i + n])
                result.append(window)
            
            return result
                
        elif split == "letter":
            # Обробка для n-грам літер і чисел
            temp = []
            data = remove_punctuation(data)
            for word in data:
                for i in word:
                    if i == ' ':
                        continue
                    temp.append(i)
            L = len(temp)
            data = temp
            temp = []
            for i in range(L - n + 1):
                window = tuple(data[i:i + n])
                temp.append(window)
            return temp
                
        elif split == 'symbol':
            # Обробка для n-грам символів
            temp = []
            for char in data:
                if char == " " or char == "\n" or char == "\ufeff":
                    temp.append("space")
                else:
                    temp.append(char.lower())
            data = temp
            L = len(data)
            temp = []
            for i in range(L - n + 1):
                window = tuple(data[i:i + n])
                temp.append(window)
            return temp

    return []




app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], use_pages=True)

# Dictionary to store uploaded files
uploaded_files = {}
# Dictionary to store file lengths with structure: {filename: {'word': length, 'symbol': length, 'letter': length}}
file_lengths = {}
# List to store batch processing results
batch_results = []

# Removing the corpuses list since we're using file upload now
colors = {
    "background": "#a1a1a1",
    "text": "#a1a1a1"}




#app.layout = layout1
df = None
g = None

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



def is_valid_letter(char: str) -> bool:
    """
    Check if a character should be skipped.
    Returns True if character should be skipped.
    """
    invalid_characters = [' ', '\n', '\ufeff', '°', '"', '„', '–']
    return char in invalid_characters


length_updated = False



def remove_empty_strings(arr: List[str]) -> List[str]:
    """
    Видаляє порожні рядки та спеціальні символи з списку.
    
    Args:
        arr: Список рядків для обробки
        
    Returns:
        List[str]: Список без порожніх рядків та спеціальних символів
    """
    return [item for item in arr if item and item != '\ufeff']

new_ngram = None


    
def add_batch_statistics(results):
    """
    Adds mean and standard deviation rows to batch results
    
    Args:
        results: List of batch results to add statistics to
    """
    if not results:
        return
    
    # Extract only numerical data for statistics
    data_for_stats = []
    numeric_fields = ["length", "vocabulary", "time", "r_avg", "dr", "rw_avg", "drw", 
                    "gamma_avg", "dgamma", "gammaw_avg", "dgammaw"]
    
    for item in results:
        # Skip statistics rows (if this function is called multiple times)
        if item["filename"] in ["MEAN", "STDDEV"]:
            continue
        
        data_point = {}
        for field in numeric_fields:
            if field in item:
                data_point[field] = item[field]
        
        data_for_stats.append(data_point)
    
    # Calculate means
    if not data_for_stats:
        return
        
    df_stats = pd.DataFrame(data_for_stats)
    
    # Calculate means
    means = {
        "no": len(results) + 1,
        "filename": "MEAN",
        "f_min": "-",
    }
    
    # Calculate standard deviations
    stddevs = {
        "no": len(results) + 2,
        "filename": "STDDEV",
        "f_min": "-",
    }
    
    # Fill in statistics for all numeric fields
    for field in numeric_fields:
        if field in df_stats.columns:
            means[field] = round(df_stats[field].mean(), 
                               8 if field in ["r_avg", "dr", "rw_avg", "drw", "gamma_avg", "dgamma", "gammaw_avg", "dgammaw"] else 
                               3 if field == "time" else 0)
            
            stddevs[field] = round(df_stats[field].std(), 
                                 8 if field in ["r_avg", "dr", "rw_avg", "drw", "gamma_avg", "dgamma", "gammaw_avg", "dgammaw"] else 
                                 3 if field == "time" else 0)
    
    # Remove old statistics rows if present
    results[:] = [r for r in results if r["filename"] not in ["MEAN", "STDDEV"]]
    
    # Add statistics to results
    results.append(means)
    results.append(stddevs)


clikced_ngram = None



save_folder = None
def pick_folder():
    global save_folder
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    new_folder = filedialog.askdirectory()
    if new_folder and new_folder is not None and new_folder != "":
        save_folder = new_folder
    root.destroy() 



# import webbrowser # Commented out as it might cause issues if run non-interactively

if __name__ == "__main__":
    #webbrowser.open_new("http://127.0.0.1:8050/") # Автоматично відкриває браузер
    # Replace app.run() with the older style Flask server run for Dash < 2.0
    app.server.run(host='0.0.0.0', port=8050, debug=True)

