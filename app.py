import numbers
from typing import List, Tuple, Optional, Dict, Any, Union
import gc  # Garbage Collector для кращого управління пам'яттю

import numpy as np
from numba import jit, njit, prange
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Обробка даних і тексту
import re
from string import punctuation
from time import time
import openpyxl

# Dash і візуалізація
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

# Системні і допоміжні бібліотеки
import base64
import io
from os import listdir
import webbrowser
from dash.dependencies import Input, Output, State
import plotly.express as px
from sklearn.metrics import r2_score
import networkx as nx
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import numba
import os
import chardet

# tkinter for selecting browsing folder
import tkinter as tk
from tkinter import filedialog

#Code Tokenizer
from processing.CodeTokenizer import CodeTokenizer
from processing.ngrams import newNgram

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

toast_visible = False
error_visible = False
analyze_visible = False



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




@jit(nopython=True)
def s(window: np.ndarray) -> int:
    """
    Обчислює суму значень вікна.
    
    Args:
        window: Масив значень
        
    Returns:
        int: Сума значень
    """
    # Використовуємо оптимізовану NumPy функцію
    return np.sum(window)


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
        
    # Оптимізоване обчислення MSE
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
        
    # Оптимізоване обчислення коефіцієнта варіації
    mean_x = np.mean(x)
    if mean_x == 0:  # Запобігаємо діленню на нуль
        return 0.0
    std_x = np.std(x)
    return std_x / mean_x


@njit(fastmath=True)
def calc_non_overlapping_shift(k, min_window, window_expansion):
    """
    Розраховує зміщення для режиму non-overlapping
    k - номер кроку (починаючи з 1)
    """
    # Numba не працює з None значеннями, тому перевірка робиться в make_windows
    if k == 1:
        return min_window
    else:
        return min_window + (k-1) * window_expansion

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


@njit(fastmath=True)
def calc_sum(x):
    sums = np.empty(len(x))
    for i, w in enumerate(x):
        sums[i] = np.sum(w)
    return sums


@jit(nopython=True, fastmath=True)
def fit(x, a, b):
    return a * (x ** b)


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

def dfa(data: List, args: Tuple[int, int, int], 
       overlap_mode: str = "overlapping", 
       min_window: Optional[int] = None, 
       window_expansion: Optional[int] = None) -> np.ndarray:
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
    
    if overlap_mode == "overlapping":
        # Стандартний режим з фіксованим зміщенням
        window_count = len(range(0, l - wi, wh))
        #count = np.zeros(window_count, dtype=np.uint8)
        count = np.zeros(window_count, dtype=np.uint16)
        
        for index, i in enumerate(range(0, l - wi, wh)):
            temp_v = []
            x = []
            for ngram in data[i:i + wi]:
                if ngram in temp_v:
                    x.append(0)
                else:
                    temp_v.append(ngram)
                    x.append(1)
            count[index] = s(np.array(x, dtype=np.uint8))
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
            
        count = np.zeros(len(window_positions), dtype=np.uint16)
        #count = np.zeros(len(window_positions), dtype=np.uint8)
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
    
    return count



app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Dictionary to store uploaded files
uploaded_files = {}
# Dictionary to store file lengths with structure: {filename: {'word': length, 'symbol': length, 'letter': length}}
file_lengths = {}
# List to store batch processing results
batch_results = []

# Removing the corpuses list since we're using file upload now
# corpuses = listdir("corpus/")
colors = {
    "background": "#a1a1a1",
    "text": "#a1a1a1"}

import dash_bootstrap_components as dbc

layout2 = html.Div()

layout1 = html.Div([
    dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("Configuration:", style={"background-color": "#e9f5fe", "fontWeight": "bold"}),
                        dbc.CardBody(
                            [
                                #MODE SELECT
                                html.Div([
                                    html.H6("Text type", 
                                            className="text-primary text-center mb-2", 
                                            style={"background": "#f8f9fa", "padding": "6px", "border-radius": "5px"}),
                                    html.Div([
                                        dcc.RadioItems(
                                            id='mode-selector',
                                            options=[
                                                {'label': 'Natural Text', 'value': 'natural_text'},
                                                {'label': 'Computer Code', 'value': 'computer_code'},
                                            ],
                                            value='natural_text', 
                                            labelStyle={'display': 'inline-block', 'margin': '0 10px 0 10px'}
                                        ),
                                    ],                                 
                                    style={
                                        'display': 'flex',
                                        'justifyContent': 'center',
                                        'alignItems': 'center',
                                    }),
                                    html.Hr(),
                                ]),
                                #IGNORE COMMENTS
                                html.Div([
                                    html.H6("Ignore comments", 
                                            className="text-primary text-center mb-2", 
                                            style={"background": "#f8f9fa", "padding": "6px", "border-radius": "5px"}),
                                    html.Div([
                                        dcc.RadioItems(
                                            id='comments-selector',
                                            options=[
                                                {'label': 'Yes', 'value': True},
                                                {'label': 'No', 'value': False},
                                            ],
                                            value=False, 
                                            labelStyle={'display': 'inline-block', 'margin': '0 10px 0 10px'}
                                        ),
                                    ],                                 
                                    style={
                                        'display': 'flex',
                                        'justifyContent': 'center',
                                        'alignItems': 'center',
                                    }),
                                    html.Hr(),
                                ], id="ignore-comments-container"),
                                # FILE SECTION
                                html.Div([
                                    html.H6("File Selection", 
                                           className="text-primary text-center mb-2", 
                                           style={"background": "#f8f9fa", "padding": "6px", "border-radius": "5px"}),
                                
                                    html.Label("Upload file:"),
                                    html.Div(
                                        [
                                            # Replace dropdown with Upload component
                                            dcc.Upload(
                                                id='upload-data',
                                                children=html.Div([
                                                    'Drag and Drop or ',
                                                    html.A('Select Files', style={'fontWeight': 'bold', 'color': '#007bff'})
                                                ]),
                                                style={
                                                    'width': '100%',
                                                    'height': '60px',
                                                    'lineHeight': '60px',
                                                    'borderWidth': '1px',
                                                    'borderStyle': 'dashed',
                                                    'borderRadius': '5px',
                                                    'textAlign': 'center',
                                                    'margin': '10px 0',
                                                    'background': '#fafafa',
                                                    'borderColor': '#007bff'
                                                },
                                                multiple=True
                                            ),
                                            html.Div(id='upload-status'),
                                            # Add dropdown for selecting files
                                            dbc.InputGroup(
                                                [
                                                    dbc.InputGroupText("Select file"),
                                                    dcc.Dropdown(
                                                        id='file-selector',
                                                        options=[],
                                                        placeholder="Select file to analyze",
                                                        style={"minWidth": "250px", "maxWidth": "100%", "whiteSpace": "nowrap", "textOverflow": "ellipsis"}
                                                    )
                                                ], 
                                                size="md", 
                                                className="mb-3",
                                                style={"marginBottom": "10px"}
                                            ),
                                        ]),
                                ], style={"marginBottom": "15px", "borderBottom": "1px solid #eee", "paddingBottom": "10px"}),
                                
                                # ANALYSIS PARAMETERS SECTION
                                html.Div([
                                    html.H6("Analysis Parameters", 
                                           className="text-primary text-center mb-2", 
                                           style={"background": "#f8f9fa", "padding": "6px", "border-radius": "5px"}),
                                    
                                    dbc.InputGroup(
                                        [
                                            dbc.InputGroupText("Order of n-grams"),
                                            dbc.Input(id="n_size", type="number", value=1, style={"font-weight": "bold"})
                                        ], 
                                        size="md", 
                                        className="mb-2"
                                    ),
                                    dbc.InputGroup(
                                        [
                                            dbc.InputGroupText("Split by"),
                                            dbc.Select(
                                                id="split",
                                                options=[
                                                    {"label": "word", "value": "word"},
                                                    {"label": "letter&number", "value": "letter"},
                                                    {"label": "symbol", "value": "symbol"},
                                                ],
                                                value="word"
                                            )
                                        ], 
                                        size="md", 
                                        className="mb-2"
                                    ),
                                    dbc.InputGroup(
                                        [
                                            dbc.Select(
                                                id="condition",
                                                options=[
                                                    {"label": "no", "value": "no"},
                                                    {"label": "periodic", "value": "periodic"},
                                                    {"label": "ordinary", "value": "ordinary"}
                                                ],
                                                value="periodic",
                                                style={"font-weight": "bold"}
                                            ),
                                    dbc.InputGroupText("Boundary Condition:")
                                ], 
                                size="md", 
                                className="mb-2"
                                    ),
                                    dbc.InputGroup([
                                        dbc.InputGroupText("Min tau:"),
                                        dbc.Select(
                                            id="min_dist_option",
                                            options=[
                                                {"label": "0", "value": "0"},
                                                {"label": "1", "value": "1"}
                                            ],
                                            value="1",
                                            style={"font-weight": "bold"}
                                        )
                                    ], className="mb-1"),
                                    dbc.InputGroup(
                                        [
                                            dbc.InputGroupText("filter"),
                                            dbc.Input(id="f_min", type="number", value=3, min=1, style={"font-weight": "bold"})
                                        ],
                                        className="mb-3"
                                    ),
                                ], style={"marginBottom": "15px", "borderBottom": "1px solid #eee", "paddingBottom": "10px"}),
                                
                                # WINDOW SETTINGS SECTION
                                html.Div([
                                    html.H6("Sliding Window Settings",
                                            className="text-primary text-center mb-2",
                                            style={"background": "#f8f9fa", "padding": "6px", "border-radius": "5px"}),
                                    
                                    dbc.InputGroup(
                                        [
                                            dbc.Select(
                                                id="overlap_mode",
                                                options=[
                                                    {"label": "overlapping", "value": "overlapping"},
                                                    {"label": "non-overlapping", "value": "non-overlapping"}
                                                ],
                                                value="overlapping"
                                            ),
                                            dbc.InputGroupText("Window Mode"),
                                        ], size="md", className="mb-2"
                                    ),
                                    
                                    dbc.InputGroup(
                                        [
                                            dbc.Select(
                                                id="def",
                                                options=[
                                                    {"label": "static", "value": "static"},
                                                    {"label": "dynamic", "value": "dynamic"}
                                                ],
                                                value="static"
                                            ),
                                            dbc.InputGroupText("Definition", style={"background-color": "#e9f5fe"}),
                                            dbc.Tooltip(
                                                "Static: Manual window parameters. Dynamic: Auto-calculated based on data size",
                                                target="def",
                                            ),
                                        ], size="md", className="mb-3"
                                    ),

                                    html.Div([
                                        html.Small([
                                            html.Span("w_min = Min Window", style={"fontWeight": "bold"}), " | ",
                                            html.Span("w_s = Window Shift", style={"fontWeight": "bold"}), " | ",
                                            html.Span("w_e = Window Expansion", style={"fontWeight": "bold"}), " | ",
                                            html.Span("w_max = Max Window", style={"fontWeight": "bold"})
                                        ], className="text-muted mb-2 d-block text-center"),
                                    ], style={"background": "#f0f8ff", "padding": "6px", "borderRadius": "5px", "marginBottom": "10px"}),

                                    dbc.InputGroup([
                                        dbc.InputGroupText(html.Span(["Min", html.Br(), "Window"], style={"lineHeight": "1.2", "textAlign": "center"}),
                                                         style={"width": "90px", "background-color": "#e9f5fe"}),
                                        dbc.Input(id="w_min", type="number", style={"font-weight": "bold"}),
                                    ], className="mb-2"),
                                    
                                    dbc.InputGroup([
                                        dbc.InputGroupText(html.Span(["Window", html.Br(), "Shift"], style={"lineHeight": "1.2", "textAlign": "center"}),
                                                         style={"width": "90px", "background-color": "#e9f5fe"}),
                                        dbc.Input(id="w_s", type="number", style={"font-weight": "bold"}),
                                    ], className="mb-2"),
                                    
                                    dbc.InputGroup([
                                        dbc.InputGroupText(html.Span(["Window", html.Br(), "Expansion"], style={"lineHeight": "1.2", "textAlign": "center"}),
                                                         style={"width": "90px", "background-color": "#e9f5fe"}),
                                        dbc.Input(id="w_e", type="number", style={"font-weight": "bold"}),
                                    ], className="mb-2"),
                                    
                                    dbc.InputGroup([
                                        dbc.InputGroupText(html.Span(["Max", html.Br(), "Window"], style={"lineHeight": "1.2", "textAlign": "center"}),
                                                         style={"width": "90px", "background-color": "#e9f5fe"}),
                                        dbc.Input(id="w_max", type="number", style={"font-weight": "bold"}),
                                    ], className="mb-3"),
                                ], style={"marginBottom": "15px", "borderBottom": "1px solid #eee", "paddingBottom": "10px"}),

                                # ACTION BUTTONS SECTION
                                html.Div([
                                    html.H6("Actions", 
                                           className="text-primary text-center mb-2", 
                                           style={"background": "#f8f9fa", "padding": "6px", "border-radius": "5px"}),
                                    
                                    dbc.Button("Analyze", id="chain_button", color="primary", 
                                              className="w-100 mb-2", 
                                              style={"fontWeight": "bold", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}, 
                                              disabled=analyze_visible),
                                    dbc.Button("Save data", id="save", color="danger", 
                                              className="w-100",
                                              style={"fontWeight": "bold", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}),
                                    html.Div(id="temp_seve",
                                             children=[]
                                             ),
                                ], style={"marginBottom": "15px", "borderBottom": "1px solid #eee", "paddingBottom": "10px"}),

                                # BATCH PROCESSING SECTION
                                html.Div([
                                    html.H6("Batch Processing", 
                                           className="text-primary text-center mb-2", 
                                           style={"background": "#f8f9fa", "padding": "6px", "border-radius": "5px"}),
                                    # Add the min-max info Div here
                                    html.Div(id='min-max-length-info', style={"marginTop": "5px", "fontSize": "small", "textAlign": "center", "marginBottom": "10px"}),
                                    
                                    dbc.InputGroup(
                                        [
                                            dbc.InputGroupText("Lmin: Fmin1"),
                                            dbc.Input(id="fmin1", type="number", value=3, min=1, style={"font-weight": "bold"})
                                        ],
                                        style={'marginBottom': '5px'}
                                    ),
                                    dbc.InputGroup(
                                        [
                                            dbc.InputGroupText("Lmax: Fmin2"),
                                            dbc.Input(id="fmin2", type="number", value=5, min=1, style={"font-weight": "bold"})
                                        ],
                                        style={'marginBottom': '5px'}
                                    ),
                                    # Add batch window settings options
                                    dbc.Collapse(
                                        [
                                            html.H6("Batch Window Settings", style={'marginTop': '10px', 'fontSize': '14px'}),
                                            dbc.InputGroup(
                                                [
                                                    dbc.Select(
                                                        id="batch_window_mode",
                                                        options=[
                                                            {"label": "Use UI settings", "value": "ui"},
                                                            {"label": "Auto per file", "value": "auto"},
                                                        ],
                                                        value="auto"
                                                    ),
                                                    dbc.InputGroupText("Window Mode")
                                                ],
                                                style={'marginBottom': '5px'}
                                            ),
                                        ],
                                        id="batch_window_controls",
                                        is_open=True
                                    ),
                                    dbc.Button("Process All Files", id="batch_process", color="success", 
                                              className="w-100", 
                                              style={'marginBottom': '10px', "fontWeight": "bold", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}),
                                    dbc.Button("Save Batch Results", id="save_batch", color="primary",
                                              className="w-100",
                                              style={'marginBottom': '10px', "fontWeight": "bold", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}),
                                    html.Div(id="temp_seve_batch", style={'marginBottom': '10px'}),
                                ], style={"marginBottom": "15px", "borderBottom": "1px solid #eee", "paddingBottom": "10px"}),
                                
                                html.Div(id="alert", children=[])
                                # html.H6("Boundary Condition:"),
                                # dcc.RadioItems(id='condition',options=[{"label":"no","value":"no"},{"label":"periodic","value":"periodic"},{"label":"ordinary","value":"ordinary"}],value="words"),
                            ]

                        ),

                    ], color="light", style={"margin-left": "0px", "margin-top": "10px", }
                ),
                width={"size": 3, "offset": 0}
            ),
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                dbc.Tabs(
                                    [
                                        dbc.Tab(label="DataTable", tab_id="data_table", label_style={"font-weight": "bold"})
                                    ],
                                    id="dataframe",
                                    active_tab="data_table"
                                )

                            ),
                            dbc.CardBody(
                                [
                                    # here table
                                    html.Div(id="box_tab",
                                             style={"display": "none", "height": "400px", "minHeight": "400px"},
                                             children=[dbc.Spinner(dash_table.DataTable(
                                                 id="table",
                                                 columns=[{"name": i, "id": i} for i in
                                                          ['rank', "ngram", "F", "R", "a", "gamma", "goodness"]],
                                                 style_data={'whiteSpace': 'auto', 'height': 'auto'},
                                                 editable=False,
                                                 filter_action="native",
                                                 sort_action="native",
                                                 page_size=50,
                                                 fixed_rows={'headers': True},
                                                 fixed_columns={'headers': True},
                                                 style_cell={'whiteSpace': 'normal',
                                                             'height': 'auto',
                                                             "widht": "auto",
                                                             'textAlign': 'right',
                                                             "fontSize": 15,
                                                             "font-family": "sans-serif"},
                                                 # 'minWidth': 40, 'width': 95, 'maxWidth': 95},
                                                 style_table={"height": "400px", "minWidth": "500px",
                                                              'overflowY': 'auto', "overflowX": "none",
                                                              "minHeight": "400px"},
                                                 style_header={
                                                     'backgroundColor': '#e9f5fe',
                                                     'fontWeight': 'bold',
                                                     'textAlign': 'center'
                                                 },
                                                 style_data_conditional=[
                                                     {
                                                         'if': {'row_index': 'odd'},
                                                         'backgroundColor': '#f9f9f9'
                                                     },
                                                     {
                                                         'if': {'state': 'selected'},
                                                         'backgroundColor': '#deeaff',
                                                         'border': '1px solid #aaa'
                                                     }
                                                 ]
                                             ))]),
                                    html.Div(id="box_chain",
                                             style={"display": "none"},
                                             children=[dbc.Spinner(dcc.Graph(id="chain", style={"height": "400px"}))]),

                                    dbc.CardHeader("Characteristics", style={"padding": "5px 20px", "background-color": "#f0f8ff", "font-weight": "bold"}),
                                    # here add chars
                                    dbc.CardBody(
                                        dbc.Row([
                                            # NOTE додала вивід 8-ми значень з екселю а також кнопку для копіювання всього
                                            dbc.Col([
                                                html.Div(["Length: "], id="l", style={"whiteSpace": "nowrap", "width": "100%", "overflow": "hidden", "textOverflow": "ellipsis", "fontWeight": "bold", "padding": "3px"}),
                                                html.Div(["Vocabulary: "], id="v", style={"fontWeight": "bold", "padding": "3px"}),
                                                html.Div(["Time: "], id="t", style={"fontWeight": "bold", "padding": "3px"})

                                            ], width={"size": 5}),
                                            dbc.Col([
                                                html.Div([""], id="new_output1", n_clicks=0, style={"padding": "3px"}),
                                                html.Div([""], id="new_output2", n_clicks=0, style={"padding": "3px"}),
                                            ], width={"size": 2}),
                                            dbc.Col([
                                                html.Div([""], id="new_output3", n_clicks=0, style={"padding": "3px"}),
                                                html.Div([""], id="new_output4", n_clicks=0, style={"padding": "3px"}),
                                            ], width={"size": 2}),
                                            dbc.Col([
                                                html.Div([""], id="new_output5", n_clicks=0, style={"padding": "3px"}),
                                                html.Div([""], id="new_output6", n_clicks=0, style={"padding": "3px"}),
                                            ], width={"size": 2}),
                                            dbc.Col([
                                                html.Div([""], id="new_output7", n_clicks=0, style={"padding": "3px"}),
                                                html.Div([""], id="new_output8", n_clicks=0, style={"padding": "3px"}),
                                                html.Div([""], id="copy_all", n_clicks=0, style={"fontWeight": "bold", "color": "#007bff", "cursor": "pointer", "textDecoration": "underline", "padding": "3px"})
                                            ], width={"size": 1}),
                                        ])
                                    ),
                                    # Add batch results table
                                    html.Div([
                                        html.H5("Batch Processing Results", style={'marginTop': '20px'}),
                                        dbc.Spinner(dash_table.DataTable(
                                            id="batch_table",
                                            columns=[
                                                {"name": "No.", "id": "no"},
                                                {"name": "Filename", "id": "filename"},
                                                {"name": "F_min", "id": "f_min"},
                                                {"name": "Length (L)", "id": "length"},
                                                {"name": "Vocabulary (V)", "id": "vocabulary"},
                                                {"name": "Time (s)", "id": "time"},
                                                {"name": "R_avg", "id": "r_avg"},
                                                {"name": "dR", "id": "dr"},
                                                {"name": "Rw_avg", "id": "rw_avg"},
                                                {"name": "dRw", "id": "drw"},
                                                {"name": "gamma_avg", "id": "gamma_avg"},
                                                {"name": "dgamma", "id": "dgamma"},
                                                {"name": "gammaw_avg", "id": "gammaw_avg"},
                                                {"name": "dgammaw", "id": "dgammaw"}
                                            ],
                                            style_data={'whiteSpace': 'normal', 'height': 'auto'},
                                            style_cell={'textAlign': 'center'},
                                            style_header={'fontWeight': 'bold', 'backgroundColor': '#e9f5fe'},
                                            style_table={"overflowX": "auto"},
                                            style_data_conditional=[
                                                {
                                                    'if': {'row_index': 'odd'},
                                                    'backgroundColor': '#f9f9f9'
                                                },
                                                {
                                                    'if': {'row_index': -1},
                                                    'fontWeight': 'bold',
                                                    'backgroundColor': 'lightyellow'
                                                },
                                                {
                                                    'if': {'row_index': -2},
                                                    'fontWeight': 'bold',
                                                    'backgroundColor': 'lightblue'
                                                }
                                            ]
                                        )),
                                        # Remove the old batch save button
                                    ], id="batch_results_container", style={"display": "none"})
                                ]
                            )
                        ], style={"padding": "0", "margin-right": "0px", "margin-top": "10px", "height": "auto", "minHeight": "650px"}),
                ],
                width={"size": 9, "padding": 0}
            ),
        ]
    ),
    dbc.Row([
        dbc.Col(
            width={"size": 6, "offset": 0},
            children=[
                dbc.Card(
                    [
                        dbc.CardHeader(
                            dbc.Tabs(
                                [
                                    dbc.Tab(label="n-gram distribution in text", tab_id="tab1", label_style={"font-weight": "bold"}),
                                ],
                                id='card-tabs1',
                                active_tab="tab1"
                                #active_tab="tab1",
                                #card=True
                            )
                        ),
                        dbc.CardBody([
                            dcc.Graph(id="graphs", config={'displayModeBar': True, 'displaylogo': False})

                        ], style={"background-color": "#fcfcfc"})
                    ], style={"height": "100%", "widht": "100%", "margin-right": "0%", "margin-top": "10px",
                              "margin-left": "0%"}
                )
            ]),
        dbc.Col(
            width={"size": 6},
            children=[

                dbc.Card(
                    [
                        dbc.CardHeader(
                            dbc.Tabs(
                                [
                                    dbc.Tab(label="fluctuations", tab_id="tab2", label_style={"font-weight": "bold"}),
                                    dbc.Tab(label="gamma vs. R", tab_id="tab3", label_style={"font-weight": "bold"})
                                ],
                                id='card-tabs',
                                active_tab="tab2"
                                #active_tab="tab2",
                                #card=True
                            )
                        ),
                        dbc.CardBody([
                            dcc.RadioItems(
                                id="scale",
                                options=[
                                    {"label": "linear fit", "value": "linear"},
                                    {"label": "log-log fit", "value": "log"}
                                ],
                                value="linear",
                                labelStyle={"marginRight": "15px", "fontWeight": "bold"},
                                inputStyle={"marginRight": "5px"},
                                style={"marginBottom": "10px", "backgroundColor": "#f8f9fa", "padding": "8px", "borderRadius": "5px"}
                            ),
                            dcc.Graph(id="fa", config={'displayModeBar': True, 'displaylogo': False})

                        ], style={"background-color": "#fcfcfc"})

                    ], style={"height": "100%", "widht": "100%", "padding": "0", "margin-right": "0%",
                              "margin-top": "10px", "margin-left": "0%"}
                )

            ]
        )
    ]

    ),
    dbc.Row(
        children=[
            html.Br(),
            html.Br()
        ]
    ),
    dcc.Store(id='stored-data'),
    html.Div(id='output-message'),
    dbc.Toast(
        id="click-toast",
        header="Attention",
        icon="danger",
        is_open=error_visible,
        dismissable=True,
        duration=6000,
        children="Length has not been calculated yet!",
        style={"position": "fixed", "top": "40%", "right": "40%", "width": 500, "zIndex": 9999}
    ),
])
from dash.dependencies import Input, Output, State

app.layout = layout1
df = None
g = None
import plotly.express as px
from sklearn.metrics import r2_score
import networkx as nx
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import numba
import os

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

