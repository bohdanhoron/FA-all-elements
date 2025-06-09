import numbers
from typing import List, Tuple, Optional, Dict, Any, Union
import gc  # Garbage Collector для кращого управління пам'яттю

import numpy as np
import pandas as pd
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
from processing.NgrammProcessor import NgrammProcessor
from processing.ngrams import Ngram, newNgram

from calculations.bc import calculate_distance
from calculations.filtering import highpass_threshold
from calculations.main import FA
from calculations.stats import R
from calculations.fitting_functions import power_law
from app import * #prepare_data, make_markov_chain, clear_memory, remove_punctuation

uploaded_files = {}
file_lengths = {}
model = {}
#df = None
new_ngram = None
batch_results = []

@dash.callback(
    [Output('upload-status', 'children'),
     Output('file-selector', 'options'),
     Output('min-max-length-info', 'children')], # Added new output
    [Input('upload-data', 'contents')],
    [State('mode-selector', 'value'),
     State('comments-selector', 'value'),
     State('upload-data', 'filename'),
     State('n_size', 'value'),
     State('split', 'value')] # Added split state
)
def update_upload_status(contents,processor_mode, ignore_comments, filenames, n_size, split_mode):
    global uploaded_files, file_lengths

    min_max_info = ""
    options = [{'label': filename, 'value': filename, 'title': filename} for filename in list(uploaded_files.keys())]
    
    if contents is None:
        # Calculate min/max even if no new files are uploaded, but existing ones are present
        if file_lengths and split_mode:
            lengths = [file_lengths[filename].get(split_mode, 0) for filename in file_lengths]
            if lengths:
                min_len = min(lengths)
                max_len = max(lengths)
                split_label = "letters&numbers" if split_mode == 'letter' else f"{split_mode}s"
                min_max_info = f"Min/Max Length ({split_label}): {min_len} / {max_len}"
        return html.Div(["No new files uploaded"]), options, html.Div(min_max_info)
    
    success_count = 0
    error_count = 0

    print("guessing encoding...")
    for i, (content, filename) in enumerate(zip(contents, filenames)):
        try:
            content_type, content_string = content.split(',')
            decoded = base64.b64decode(content_string)
            detection = chardet.detect(decoded)
            encoding = detection['encoding'] or 'windows-1251'
            print(f"Detected encoding: {encoding}")

            try:
                file_content = decoded.decode(encoding)
                uploaded_files[filename] = file_content
                file_lengths[filename] = {}
                
                # Word length
                computer_code = True if processor_mode == 'computer_code' else False
                text_word = file_content
                if not computer_code:
                    text_word = re.sub(r'\n+', '\n', file_content)
                    text_word = re.sub(r'\n\s\s', '\n', text_word)
                    text_word = re.sub(r'﻿', '', text_word)
                    text_word = re.sub(r'--', ' -', text_word)
                processor = NgrammProcessor(computer_code=computer_code, ignore_comments=ignore_comments)
                processor.preprocess(text_word, file_name=filename)
                words = processor.get_words()
                file_lengths[filename]['word'] = len(words)
                
                # Symbol length
                """symbols = []
                for char in file_content:
                    if char == " " or char == "\n" or char == "\ufeff":
                        symbols.append("space")
                    else:
                        symbols.append(char.lower())
                file_lengths[filename]['symbol'] = len(symbols)
                print(len(symbols))"""
                #print(len(file_content)

                file_lengths[filename]['symbol'] = len(file_content)
                
                # Letter length
                #text_letter = remove_punctuation(file_content)
                """letters = []
                for word in text_letter:
                    for i in word:
                        if i == ' ':
                            continue
                        letters.append(i)"""

                #print(len(letters))
                #print(len(''.join(file_content.split(' '))))
                #print(len(re.sub(' ', '', file_content)))
                
                #file_lengths[filename]['letter'] = len(letters)
                file_lengths[filename]['letter'] = len(re.sub(' ', '',  file_content))

                print(f"✓ Uploaded: {filename}")
                print(f"  Words: {file_lengths[filename]['word']} | Symbols: {file_lengths[filename]['symbol']} | Letters: {file_lengths[filename]['letter']}")
                
                success_count += 1
            except UnicodeDecodeError:
                print(f"✗ Error: {filename} is not a valid text file")
                error_count += 1
        except Exception as e:
            print(f"✗ Error processing {filename}: {str(e)}")
            error_count += 1
    
    summary_message = html.Div([
        html.H5(f"Upload Summary:"),
        html.P(f"Successfully uploaded: {success_count} file(s)", style={'color': 'green'}),
        html.P(f"Files with errors: {error_count}", style={'color': 'red' if error_count > 0 else 'green'})
    ])
    
    options = [{'label': filename, 'value': filename, 'title': filename} for filename in list(uploaded_files.keys())]

    # Calculate Min/Max length based on the current split mode
    if file_lengths and split_mode:
        lengths = [file_lengths[filename].get(split_mode, 0) for filename in file_lengths]
        if lengths:
            min_len = min(lengths)
            max_len = max(lengths)
            split_label = "letters&numbers" if split_mode == 'letter' else f"{split_mode}s"
            min_max_info = f"Min/Max Length ({split_label}): {min_len} / {max_len}"
    
    return summary_message, options, html.Div(min_max_info)


# Add callback to handle file selection
@dash.callback(
    [Output('l', 'children'),
     Output('w_min', 'value'),
     Output('w_s', 'value'),
     Output('w_e', 'value'),
     Output('w_max', 'value')],
    [Input('file-selector', 'value'),
     Input('split', 'value'),
     Input('mode-selector', 'value'),
     Input('comments-selector', 'value')],
    [State('def', 'value'),
     State('n_size', 'value')]
)
def process_selected_file(selected_filename, split, processor_mode, ignore_comments, definition, n):
    """
    метод рахує розмір тексту і пропонує розміри вікон, оброблений тут текст далі нікуди не передається

    Returns:
    ---
    lengths_elements: List of html objects
        contain message about length of selected file in different units (letters&numbers, symbols, words)
    w_min: int
        minimal window size
    w_s: int
        step window size (how to move window over sequence)
    w_e: int
        expansion window size (how to increase window size)
    w_max: int
        maximal window size
    """

    global L, data, length_updated
    
    if selected_filename is None or selected_filename not in uploaded_files:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    file = uploaded_files[selected_filename]
    length_updated = False
    computer_code = True if processor_mode == 'computer_code' else False

    # Calculate L based on split type (dynamic or static handles data differently)
    if definition == "dynamic":
        data = prepare_data(file, n, split, selected_filename, computer_code, ignore_comments)
        L = len(data)
        w_max = int(L / 10)
        w_min = int(w_max / 10)
    else:
        # Static mode calculation based on selected split
        if split == "letter":
        # letter preprocessing
            temp = []
            data = remove_punctuation(file)
            for word in data:
                for i in word:
                    if i == ' ':
                        continue
                    temp.append(i)
            data = temp
            L = len(data)
        elif split == "symbol":
        # symbol preprocessing
            temp = []
            for char in file:
                if char == " " or char == "\n" or char == "\ufeff":
                    temp.append("space")
                else:
                    temp.append(char.lower())
            data = temp
            L = len(data)
        elif split == "word":
        # word preprocessing
            """if not computer_code:
                file = re.sub(r'\n+', '\n', file)
                file = re.sub(r'\n\s\s', '\n', file)
                file = re.sub(r'﻿', '', file)
                file = re.sub(r'--', ' -', file)"""

            processor = NgrammProcessor(computer_code=computer_code, ignore_comments=ignore_comments)
            processor.preprocess(file, file_name=selected_filename)
            data = processor.get_words()
            L = len(data)

        file_lengths[selected_filename][split] = L
        w_max = int(L / 20)
        w_min = int(w_max / 20)
        length_updated = True
    
    # Format the lengths into a multi-line Div
    length_elements = [html.Strong("Length:")]
    
    # Get all lengths from the stored dictionary
    lengths = file_lengths[selected_filename]
    
    # Add each length type on a new line
    if 'word' in lengths:
        length_elements.append(html.Div(f"words: {lengths['word']}"))
    if 'symbol' in lengths:
        length_elements.append(html.Div(f"symbols: {lengths['symbol']}"))
    if 'letter' in lengths:
        length_elements.append(html.Div(f"letters&numbers: {lengths['letter']}"))
        
    return length_elements, w_min, w_min, w_min, w_max

# Add callback for batch processing
@dash.callback(
    [Output("batch_table", "data"),
     Output("batch_results_container", "style")],
    [Input("batch_process", "n_clicks")],
    [State('mode-selector', 'value'),
     State('comments-selector', 'value'),
     State("fmin1", "value"),
     State("fmin2", "value"),
     State("split", "value"),
     State("n_size", "value"),
     State("condition", "value"),
     State("def", "value"),
     State("min_dist_option", "value"),
     State("overlap_mode", "value"),
     State("w_min", "value"),
     State("w_s", "value"),
     State("w_e", "value"),
     State("w_max", "value"),
     State("batch_window_mode", "value")]
)
def process_all_files(n_clicks, processor_mode, ignore_comments, fmin1, fmin2, split, n_size, condition, definition, min_dist_option, 
                      overlap_mode, w_min, w_s, w_e, w_max, batch_window_mode):
    global batch_results, uploaded_files, file_lengths
    
    if n_clicks is None or not uploaded_files:
        return [], {"display": "none"}
    
    # Find Lmin and Lmax for the current split method
    lengths = [file_lengths[filename][split] for filename in list(uploaded_files.keys())]
    if not lengths:
        return [], {"display": "none"}
        
    lmin = min(lengths)
    lmax = max(lengths)
    
    # Initialize batch results list
    batch_results = []
    
    # Process each file sequentially and clear memory after each
    file_list = list(uploaded_files.items())
    computer_code = True if processor_mode == 'computer_code' else False
    for idx, (filename, file_content) in enumerate(file_list, 1):
        # Force garbage collection before starting new file
        gc.collect()

        print(f"{filename}, {definition}")
        
        # Calculate F_min based on file length
        file_length = file_lengths[filename][split]

        f_min = highpass_threshold(file_length, lmin, lmax, fmin1, fmin2)
        
        # Process the file
        start_time = time()
        
        # Initialize all variables locally to avoid memory leaks
        data = None
        local_model = {}
        
        # Process data based on definition mode
        if definition == "dynamic":
            data = prepare_data(file_content, n_size, split, filename, computer_code, ignore_comments)
        else:
            if split == "letter":
                file_text = re.sub(r'	', '', file_content)
                processed_data = remove_punctuation(file_text)
                temp = []
                current_number = ""
                
                for char in processed_data:
                    if char.isspace() or char == '\n' or char == '\ufeff':
                        if current_number:
                            temp.append(current_number)
                            current_number = ""
                        continue
                    if char.isdigit() or char.isalpha():
                        temp.append(char)
                
                data = temp
                # Free memory immediately
                del processed_data
                del temp
                del file_text
                gc.collect()
            elif split == "symbol":
                # Optimize processing
                clean_text = re.sub(r'	', '', file_content)
                clean_text = re.sub(r'\n+', '\n', clean_text)
                clean_text = re.sub(r'\n\s\s', '\n', clean_text)
                clean_text = re.sub(r'﻿', '', clean_text)
                temp = []
                for i in clean_text:
                    if i == " " or i == "\n" or i == "\ufeff":
                        temp.append("space")
                    elif i == '﻿' or is_valid_letter(i):
                        continue
                    else:
                        temp.append(i.lower())
                data = temp
                del temp
                del clean_text
                gc.collect()
            elif split == "word":
                file_text = file_content
                """if not computer_code:
                    file_text = re.sub(r'\n+', '\n', file_content)
                    file_text = re.sub(r'\n\s\s', '\n', file_text)
                    file_text = re.sub(r'﻿', '', file_text)
                    file_text = re.sub(r'--', ' -', file_text)"""

                processor = NgrammProcessor(computer_code=computer_code, ignore_comments=ignore_comments)
                processor.preprocess(file_text, file_name=filename)
                data = processor.get_words()
                del processor
                del file_text
                gc.collect()

        L = len(data)
        
        # Calculate window parameters based on batch settings
        # NOTE виправити ці автоматичні призначення!!!
        if batch_window_mode == "ui":
            # Use the values from the UI
            wm_val = int(w_max) if w_max is not None else int(L / 20)
            w_val = int(w_s) if w_s is not None else int(wm_val / 20)
            wh_val = int(w_s) if w_s is not None else w_val
            we_val = int(w_e) if w_e is not None else w_val
        else:  # auto
            # Calculate based on file length
            # NOTE ці рядки умови при "dynamic" не працюють
            if definition == "dynamic":
                wm_val = int(L / 20)
                w_val = int(wm_val / 20)
            else:
                wm_val = int(L / 20)
                w_val = int(wm_val / 20)
            wh_val = w_val
            we_val = w_val
            
        # Ensure we have valid non-zero values
        wm_val = max(10, wm_val)
        w_val = max(5, w_val)
        wh_val = max(1, wh_val)
        we_val = max(1, we_val)
        
        # Make Markov chain (use local variables)
        for i in range(L - n_size + 1):
            if i + n_size <= len(data):
                if n_size == 1:
                    ngram = data[i]
                else:
                    ngram = tuple(data[i:i + n_size])
                
                if ngram not in local_model:
                    local_model[ngram] = Ngram()
                    local_model[ngram].pos = []
                
                local_model[ngram].pos.append(i)
        
        # Build a vocabulary count for statistics
        V = len(local_model)

        if definition=="static":
        
            # Make DataFrame locally instead of using global function to save memory
            filtered_data = list(filter(lambda x: len(local_model[x].pos) >= f_min, local_model))
            
            data_df = {"ngram": [], "F": np.empty(len(filtered_data), dtype=np.int32)}
            for i, ngram in enumerate(filtered_data):
                data_df["ngram"].append(ngram)
                data_df["F"][i] = len(local_model[ngram].pos)
            
            current_df = pd.DataFrame(data=data_df)
            
            # Process positions and calculate parameters
            temp_gamma = []
            temp_R = []
            temp_error = []
            temp_a = []
           
            print(f"windows: w_val {w_val}, wm_val {wm_val}, we_val {we_val}.")
            #windows = list(range(w_val, wm_val, we_val))
            windows = list(range(w_val, wm_val+1, we_val))
            
            # Process each ngram
            for i, row in current_df.iterrows():
                ngram = row['ngram']
                
                
                #r = round(R(np.array(local_model[ngram].dt)), 8)
                #temp_R.append(r)
                min_dist_int = int(min_dist_option) if isinstance(min_dist_option, (str, float)) else min_dist_option
                
                local_model[ngram].dt = calculate_distance(np.array(local_model[ngram].pos, dtype=np.uint32), L, condition, ngram, min_dist_int)
                r = round(R(np.array(local_model[ngram].dt)), 8)

                # get ngram binary sequence with positions    
                ngram.bool = np.zeros(L, dtype=np.int8)
                for pos in ngram.pos:
                    ngram.bool[pos] = 1
                
                a, gamma, fa, error = FA(local_model[ngram], L, windows, wh_val, overlap_mode, w_val, we_val)

                temp_R.append(r)
                temp_a.append(a)
                temp_gamma.append(gamma)
                #temp_fa.append(fa)
                temp_error.append(error)
            
            # Handle n-grams formatting if needed
            if n_size > 1:
                temp_ngram = []
                for ng in current_df['ngram']:
                    if isinstance(ng, tuple):
                        temp_ngram.append(" ".join(ng))
                    else:
                        temp_ngram.append(ng)
                current_df["ngram"] = temp_ngram
            
            # Add calculated parameters to DataFrame
            current_df['R'] = temp_R
            current_df['gamma'] = temp_gamma
            current_df['a'] = temp_a
            current_df['goodness'] = temp_error
            current_df = current_df.sort_values(by="F", ascending=False)
            current_df['rank'] = range(1, len(current_df) + 1)
            current_df = current_df.set_index(pd.Index(np.arange(len(current_df))))
            
            # Calculate the 8 parameters
            df_filtered = current_df.copy()
            if len(df_filtered) > 0:
                df_filtered['w'] = df_filtered['F'] / df_filtered['F'].sum()
                
                R_avg = df_filtered['R'].mean()
                dR = df_filtered['R'].std()
                Rw_avg = (df_filtered['R'] * df_filtered['w']).sum()
                dRw = np.sqrt((((df_filtered['R'] - Rw_avg) ** 2) * df_filtered['w']).sum())
                
                gamma_avg = df_filtered['gamma'].mean()
                dgamma = df_filtered['gamma'].std()
                gammaw_avg = (df_filtered['gamma'] * df_filtered['w']).sum()
                dgammaw = np.sqrt((((df_filtered['gamma'] - gammaw_avg) ** 2) * df_filtered['w']).sum())
            else:
                # Default values if no data
                R_avg = dR = Rw_avg = dRw = gamma_avg = dgamma = gammaw_avg = dgammaw = 0

            """del data
            del local_model"""
            del current_df
            del df_filtered
            del temp_gamma
            del temp_R
            del temp_error
            del temp_a

        elif definition=="dynamic":

            # Додаємо перевірку на None для безпеки
            w_max_val = int(L / 20)
            #w_max_val = int(L / 10)
            #w_max_val = int(w_max) if w_max is not None else int(L / 10)
            #w_s_val = int(w_s) if w_s is not None else 5
            #w_max_val = int(w_max) if w_max is not None else 100
            #w_s_val = int(w_max_val / 10)
            w_s_val = int(w_max_val / 20)
            #w_s_val = int(w_s) if w_s is not None else int(w_max_val / 10)
            #w_e_val = int(w_e) if w_e is not None else 5
            #w_e_val = int(w_e) if w_e is not None else w_s_val
            w_e_val = w_s_val
            
            # Запобігання ValueError: range() arg 3 must not be zero
            if w_e_val == 0:
                w_e_val = 5
                print("Warning: Window expansion (w_e) was 0, set to default value 5")
            
            print(f"LENGTH {L}")
            print(f"windows: w_s_val {w_s_val}, w_max_val {w_max_val}, w_e_val {w_e_val}.")
            #windows = list(range(w_s_val, w_max_val, w_e_val))
            windows = list(range(w_s_val, w_max_val+1, w_e_val))

            #print(w_s_val, w_max_val, w_e_val)
            
            # Створення нового n-граму та його обробка
            new_ngram = newNgram(data, w_s_val, L)
            
            # Визначаємо функцію для паралельної обробки вікон
            def process_window(w):
                if overlap_mode == "overlapping":
                    return new_ngram.func(w)
                else:
                    return new_ngram.func(w, overlap_mode=overlap_mode, min_window=w_s_val, window_expansion=w_e_val)
            
            # Паралельна обробка вікон (якщо їх достатньо багато)
            if len(windows) > 4:  # Паралелізуємо лише якщо є достатня кількість вікон
                with ThreadPoolExecutor(max_workers=min(4, len(windows))) as executor:
                    list(executor.map(process_window, windows))
            else:
                # Послідовна обробка для малої кількості вікон
                for w in windows:
                    process_window(w)
           
            ### СТАТИЧНА ЧАСТИНА для розрахунку R ###
            # Оптимізоване створення списків для елементів та їх позицій
            temp_v = []
            temp_pos = []
            unique_items = set()  # Використовуємо множину для швидшого пошуку
            
            for i, ngram in enumerate(data):
                if ngram not in unique_items:
                    unique_items.add(ngram)
                    temp_v.append(ngram)
                    temp_pos.append(i)
            
            # Використовуємо numpy масиви для ефективнішої обробки
            temp_pos_array = np.array(temp_pos, dtype=np.uint32)
            # Використовуємо перший елемент або "new_ngram" для розрахунку відстаней
            ngram_for_calc = temp_v[0] if temp_v else "new_ngram"
            new_ngram.dt = calculate_distance(temp_pos_array, L, condition, ngram_for_calc, min_dist_option)
            new_ngram.R = round(R(new_ngram.dt), 8)

            ##########################################


            ### ДИНАМІЧНА ЧАСТИНА для розрахунку gamma ###

            new_ngram.fit()

            V = len(temp_v)
            
            end_time = time()
            execution_time = end_time - start_time
            
            # Підготовка даних для відображення
            #df_table = df.to_dict("records")
            
            # Додаємо інформацію про розмір словника і час виконання
            vocab_info = f"Vocabulary: {V}"
            time_info = f"Time: {execution_time:.4f} s"
            

            R_avg = new_ngram.R
            dR = 0
            Rw_avg = 0
            dRw = 0

            gamma_avg = new_ngram.gamma
            dgamma = 0
            gammaw_avg = 0
            dgammaw = 0
            
            # Звільняємо пам'ять від тимчасових змінних
            del temp_v, temp_pos, unique_items, temp_pos_array
            gc.collect()

        
        # Calculate execution time
        end_time = time()
        execution_time = end_time - start_time
        
        # Create batch result (store only what's needed for the table)
        batch_result = {
            "no": idx,
            "filename": filename,
            "f_min": f_min,
            "length": L,
            "vocabulary": V,
            "time": round(execution_time, 3),
            "r_avg": round(R_avg, 8),
            "dr": round(dR, 8),
            "rw_avg": round(Rw_avg, 8),
            "drw": round(dRw, 8),
            "gamma_avg": round(gamma_avg, 8),
            "dgamma": round(dgamma, 8),
            "gammaw_avg": round(gammaw_avg, 8),
            "dgammaw": round(dgammaw, 8)
        }

        #print(batch_result)
        
        batch_results.append(batch_result)
        
        # Explicitly clean up all local variables
        del data
        del local_model
        """del current_df
        del df_filtered
        del temp_gamma
        del temp_R
        del temp_error
        del temp_a"""
        
        # Force garbage collection multiple times
        gc.collect()
        gc.collect()
    
    # Calculate statistics if we have results
    if batch_results:
        # Add the mean and standard deviation rows after all files processed
        add_batch_statistics(batch_results)
    
    return batch_results, {"display": "block"}

# Update the batch results table to show window parameters too
@dash.callback(
    Output("batch_table", "columns"),
    [Input("batch_process", "n_clicks")]
)
def update_batch_table_columns(n_clicks):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    
    columns = [
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
    ]
    
    return columns

# Add callback to save batch results
@dash.callback(
    Output("temp_seve_batch", "children"),  # Changed output ID to avoid conflicts
    [Input("save_batch", "n_clicks")],
    [State("n_size", "value"),
     State("split", "value"),
     State("condition", "value"),
     State("def", "value"),
     State("min_dist_option", "value"),
     State("overlap_mode", "value"),
     State("batch_window_mode", "value")]
)
def save_batch_results(n_clicks, n_size, split, condition, definition, min_dist_option, overlap_mode, batch_window_mode):
    global save_folder
    if n_clicks is None:
        return dash.no_update
    if not batch_results:
        return html.Div(["No batch results to save"])
    
    try:
        if save_folder is None or save_folder == "":
            pick_folder()
            if save_folder is None or save_folder == "":
                return dash.no_update
        df_batch = pd.DataFrame(batch_results)
    
        # Ensure column names match the display columns for consistency
        # This ensures the saved file has the same data structure as what's shown in the UI
        column_mapping = {}
        
        # Create filename with parameters
        output_filename = "{}/batch_results_n={},split={},condition={},definition={},min_dist={},overlap={},window_mode={}.xlsx".format(
            save_folder, n_size, split, condition, definition, min_dist_option, overlap_mode, batch_window_mode)
                
        # Ensure directory exists
        os.makedirs(save_folder, exist_ok=True)
        
        # Save to Excel - modify to use older pandas style
        with pd.ExcelWriter(output_filename) as writer:
                df_batch.to_excel(writer, index=False)
        
        return html.Div(["Saved batch results to {}".format(output_filename)])
    except Exception as e:
        print(e)
        return html.Div(["Error saving batch results: {}".format(str(e))])

@dash.callback([Output("table", "data"), Output("chain", "figure"),
               Output("box_tab", "style"),
               Output("box_chain", "style"),
               Output("alert", "children"),
               Output("v", "children"),
               Output("t", "children"),
                Output('click-toast', 'is_open'),
               ],
              [Input("chain_button", "n_clicks"),
               Input("dataframe", "active_tab")],
              [State("f_min", "value"),
               State("w_min", "value"),
               State("w_s", "value"),
               State("w_e", "value"),
               State("w_max", "value"),
               State("def", "value"),
               State("min_dist_option", "value"),
               State("overlap_mode", "value"),
               State("n_size", "value"),
               State("split", "value"),
               State("condition", "value")
               ])
def update_table(n, dataframe, f_min, w_min, w_s, w_e, w_max, definition, min_dist_option, overlap_mode, n_size, split, condition):
    """
    Оновлює таблицю та графік на основі вибраних параметрів.
    
    Використовує паралельну обробку для інтенсивних обчислень і оптимізоване управління пам'яттю
    для зменшення навантаження.
    """
    global model, L, V, df, new_ngram
    
    # Очищуємо кеш для мемоізованих функцій
    if hasattr(prepare_data, 'clear_cache'):
        prepare_data.clear_cache()
    if hasattr(make_markov_chain, 'clear_cache'):
        make_markov_chain.clear_cache()
    
    # Викликаємо збирач сміття для звільнення пам'яті
    clear_memory(keep=['data', 'uploaded_files', 'file_lengths'])
    
    if n is None or dataframe is None:
        return (dash.no_update, dash.no_update, {"display": "none"}, {"display": "none"},
                dash.no_update, dash.no_update, dash.no_update,
                dash.no_update)
                
    # Вже нема вкладки MarkovChain, тому використовуємо тільки data_table
    if definition == "dynamic":
        start = time()
        
        # Додаємо перевірку на None для безпеки
        w_max_val = int(w_max) if w_max is not None else int(L / 20)
        w_s_val = int(w_s) if w_s is not None else int(w_max_val / 20)
        w_e_val = int(w_e) if w_e is not None else w_s_val
        
        # Запобігання ValueError: range() arg 3 must not be zero
        if w_e_val == 0:
            w_e_val = 5
            print("Warning: Window expansion (w_e) was 0, set to default value 5")
        
        windows = list(range(w_s_val, w_max_val+1, w_e_val))
        print(w_s_val, w_max_val, w_e_val)
        
        # Створення нового n-граму та його обробка
        new_ngram = newNgram(data, w_s_val, L)
        
        # Визначаємо функцію для паралельної обробки вікон
        def process_window(w):
            if overlap_mode == "overlapping":
                return new_ngram.func(w)
            else:
                return new_ngram.func(w, overlap_mode=overlap_mode, min_window=w_s_val, window_expansion=w_e_val)
        
        # Паралельна обробка вікон (якщо їх достатньо багато)
        if len(windows) > 4:  # Паралелізуємо лише якщо є достатня кількість вікон
            with ThreadPoolExecutor(max_workers=min(4, len(windows))) as executor:
                list(executor.map(process_window, windows))
        else:
            # Послідовна обробка для малої кількості вікон
            for w in windows:
                process_window(w)
        
        ### СТАТИЧНА ЧАСТИНА для розрахунку R ###
        # Оптимізоване створення списків для елементів та їх позицій
        temp_v = []
        temp_pos = []
        unique_items = set()  # Використовуємо множину для швидшого пошуку
        
        for i, ngram in enumerate(data):
            if ngram not in unique_items:
                unique_items.add(ngram)
                temp_v.append(ngram)
                temp_pos.append(i)
        
        # Використовуємо numpy масиви для ефективнішої обробки
        temp_pos_array = np.array(temp_pos, dtype=np.uint32)
        # Використовуємо перший елемент або "new_ngram" для розрахунку відстаней
        ngram_for_calc = temp_v[0] if temp_v else "new_ngram"
        new_ngram.dt = calculate_distance(temp_pos_array, L, condition, ngram_for_calc, min_dist_option)
        new_ngram.R = round(R(new_ngram.dt), 8)

        #############################################

        new_ngram.fit()

        # Створення DataFrame для представлення результатів
        df = pd.DataFrame({
            'rank': [1],
            'ngram': ['new_ngram'],
            'F': [len(temp_pos)],
            'R': [new_ngram.R],
            'a': [new_ngram.a],
            'gamma': [new_ngram.gamma],
            'goodness': [new_ngram.goodness]
        })
        
        V = len(temp_v)
        
        end_time = time()
        execution_time = end_time - start
        
        # Підготовка даних для відображення
        df_table = df.to_dict("records")
        
        # Додаємо інформацію про розмір словника і час виконання
        vocab_info = f"Vocabulary: {V}"
        time_info = f"Time: {execution_time:.4f} s"
        
        # Звільняємо пам'ять від тимчасових змінних
        del temp_v, temp_pos, unique_items, temp_pos_array
        gc.collect()
        
        return (df_table, dash.no_update, {"display": "inline"}, {"display": "none"},
                dash.no_update, vocab_info, time_info, False)
    else:
        # Markov Chain обробка
        start = time()

        # Створення ланцюга Маркова та DataFrame
        model = make_markov_chain(data, order=n_size)
        df = make_dataframe(model, f_min)
        
        # Перевірка безпеки для None значень
        w_max_val = int(w_max) if w_max is not None else int(L / 20)
        w_s_val = int(w_s) if w_s is not None else int(w_max_val / 20)
        w_e_val = int(w_e) if w_e is not None else w_s_val

        print("Proceed carefully, window sizes could be automatically assigned!")
        
        # Запобігання ValueError: range() arg 3 must not be zero
        if w_e_val == 0:
            w_e_val = 5
            print("Warning: Window expansion (w_e) was 0, set to default value 5")
        
        windows = list(range(w_s_val, w_max_val+1, w_e_val))
        
        # Функція для обробки окремого n-грама
        def process_ngram(ngram_data):
            ngram, index = ngram_data
            
            # Розрахунок відстаней
            dt = calculate_distance(np.array(model[ngram].pos, dtype=np.uint32), L, condition, ngram, min_dist_option)
            model[ngram].dt = dt
            
            r = round(R(dt), 8)
            model[ngram].R = r

            a, gamma, fa, err = FA(model[ngram], L, windows, w_s_val, overlap_mode, w_s_val, w_e_val)

            model[ngram].a = a
            model[ngram].gamma = gamma
            model[ngram].temp_fa = fa

            return {
                'ngram': ngram,
                'a': round(a, 8),
                'gamma': round(gamma, 8),
                'error': round(err, 5),
                'R': r
            }

        # Підготовка даних для паралельної обробки
        ngram_items = [(ngram, i) for i, ngram in enumerate(df["ngram"])]
        
        # Визначаємо кількість робітників на основі кількості n-грамів
        max_workers = min(4, len(ngram_items))
        
        # Паралельна обробка для великої кількості n-грамів, інакше послідовна
        results = []
        if len(ngram_items) >= 4:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(process_ngram, ngram_items))
        else:
            results = [process_ngram(item) for item in ngram_items]
        
        # Витягуємо результати
        temp_a = [result['a'] for result in results]
        temp_gamma = [result['gamma'] for result in results]
        temp_error = [result['error'] for result in results]
        temp_R = [result['R'] for result in results]
        
        # Обробка n-грамів для відображення
        if n_size > 1:
            temp_ngram = []
            for ng in df['ngram']:
                if isinstance(ng, tuple):
                    temp_ngram.append(" ".join(ng))
                else:
                    temp_ngram.append(ng)
            df["ngram"] = temp_ngram
        
        # Оновлення DataFrame результатами
        df['R'] = temp_R
        df['gamma'] = temp_gamma
        df['a'] = temp_a
        df['goodness'] = temp_error
        df = df.sort_values(by="F", ascending=False)
        df['rank'] = range(1, len(temp_R) + 1)
        df = df.set_index(pd.Index(np.arange(len(df))))
        
        end_time = time()
        execution_time = end_time - start
        
        # Підготовка даних для відображення
        df_table = df.to_dict("records")

        V = len(set(data))
        
        # Додаємо інформацію про розмір словника і час виконання
        vocab_info = f"Vocabulary: {V}"
        time_info = f"Time: {execution_time:.4f} s"
        
        # Звільняємо пам'ять від тимчасових змінних
        del temp_gamma, temp_R, temp_error, temp_a, results, ngram_items
        gc.collect()
        
        return (df_table, dash.no_update, {"display": "inline"}, {"display": "none"},
                dash.no_update, vocab_info, time_info, False)

@dash.callback([Output("graphs", "figure"), Output("fa", "figure"), ],
              [Input("dataframe", "active_tab"),
               Input("card-tabs", "active_tab"),
               Input("table", "active_cell"),
                # NOTE додала параметр page_current та використала його для показу правильної інформації
               Input("table", "page_current"),
               Input("table", "derived_virtual_selected_rows"),
               Input("table", "derived_virtual_indices"),
               Input("chain", "clickData"),
               Input("scale", "value"),
               Input("fa", "clickData"),
               Input("graphs", "clickData"),
               Input("w_max", "value")],
              [State("n_size", "value"),
               State("def", "value"), ])
def tab_content(active_tab2, active_tab1, active_cell, page_current, row_ids, ids, clicked_data, scale, fa_click,
                graph_click, w_max, n,
                definition):
    # Тільки для вкладки DataTable, оскільки MarkovChain було видалено
    if active_tab2 == "data_table":
        fig = go.Figure()
        fig1 = go.Figure()
        if active_tab1 == "tab2":
            if active_cell:
                if definition == "dynamic":
                    ## add bar
                    if fa_click:
                        if overlap_mode == "overlapping":
                            fig.add_trace(go.Bar(x=np.arange(w_s, L, w_s), y=new_ngram.count[fa_click["points"][0]["x"]],
                                                name="∑∆w"))
                        else:
                            # Для non-overlapping режиму потрібно розрахувати положення барів
                            bar_positions = []
                            k = 1
                            i = 0
                            ww = fa_click["points"][0]["x"]
                            while i < L - ww:
                                bar_positions.append(i)
                                shift = calc_non_overlapping_shift(k, w_s, w_e)
                                i += shift
                                k += 1
                            fig.add_trace(go.Bar(x=bar_positions, y=new_ngram.count[ww], name="∑∆w"))

                    # Перевірка, чи existує new_ngram та його атрибути
                    if new_ngram is not None and hasattr(new_ngram, 'dfa') and new_ngram.dfa:
                        fig1.add_trace(
                            go.Scatter(x=[*new_ngram.dfa.keys()], y=[*new_ngram.dfa.values()], mode='markers', name="∆F"))
                        
                        if hasattr(new_ngram, 'temp_dfa') and new_ngram.temp_dfa:
                            #fig1.add_trace(go.Scatter(x=[*new_ngram.dfa.keys()], y=[*new_ngram.temp_dfa], name="fit=aw^b"))
                            fig1.add_trace(go.Scatter(x=[*sorted(new_ngram.dfa.keys())], y=[*new_ngram.temp_dfa], name="fit=aw^b"))
                        
                        fig1.update_xaxes(type=scale)
                        fig1.update_yaxes(type=scale)
                        fig1.update_layout(hovermode="x unified")

                    return fig, fig1

                if n > 1:
                    ngram = tuple(df['ngram'][ids[active_cell['row']]].split())
                    if ngram[0] == 'new_ngram':
                        ngram = 'new_ngram'
                else:
                    ngram = df['ngram'][ids[active_cell['row']]]
                fig.add_trace(go.Scatter(x=np.arange(L), y=model[ngram].bool, name="positions"))

                if fa_click:
                    if overlap_mode == "overlapping":
                        fig.add_trace(go.Bar(x=np.arange(w_s, L, w_s), y=model[ngram].counts[fa_click["points"][0]["x"]],
                                             name="∑∆w"))
                    else:
                        # Для non-overlapping режиму потрібно розрахувати положення барів
                        bar_positions = []
                        k = 1
                        i = 0
                        while i < L - ww:
                            bar_positions.append(i)
                            shift = calc_non_overlapping_shift(k, w_s, w_e)
                            i += shift
                            k += 1
                        fig.add_trace(go.Bar(x=bar_positions, y=model[ngram].counts[ww], name="∑∆w"))
                if graph_click:
                    www = graph_click['points'][0]['x']
                graph_click = None
                fa_click = None

                temp_ww = [*model[ngram].fa.keys()]
                fig1.add_trace(
                    go.Scatter(x=temp_ww,
                               y=[*model[ngram].fa.values()],
                               mode='markers',
                               name="∆F"))
                fig1.add_trace(go.Scatter(
                    x=temp_ww,
                    y=model[ngram].temp_fa,
                    name="fit=aw^b"))
                fig1.update_xaxes(type=scale)
                fig1.update_yaxes(type=scale)
                fig1.update_layout(hovermode="x unified")
                active_cell = None
                return fig, fig1
            else:
                active_cell = None
                return fig, fig1
        else:
            hover_data = []
            if active_cell:
                if definition == "dynamic":
                    if fa_click:
                        fig.add_trace(
                            go.Bar(x=np.arange(w_s, L, w_s), y=new_ngram.count[fa_click["points"][0]["x"]], name="∑∆w"))

                    # Перевірка наявності new_ngram та його атрибутів
                    if new_ngram is not None and hasattr(new_ngram, 'R') and hasattr(new_ngram, 'gamma'):
                        fig1.add_trace(go.Scatter(x=new_ngram.R, y=new_ngram.gamma, mode='markers', hover_data=["new_ngram"]))
                        fig1.update_xaxes(type=scale)
                        fig1.update_yaxes(type=scale)
                        fig1.update_layout(hovermode="x unified")

                    return fig, fig1

                if n > 1:
                    ngram = tuple(df['ngram'][ids[active_cell['row']]].split())
                    if ngram[0] == 'new_ngram':
                        ngram = 'new_ngram'
                else:
                    ngram = df['ngram'][ids[active_cell['row']]]

                for data in df['ngram']:
                    # HERE ADDED to skip random float entities
                    if not isinstance(data, numbers.Number):
                        hover_data.append("".join(data))
                fig.add_trace(go.Scatter(x=np.arange(L), y=model[ngram].bool, name="positions"))
                if fa_click:
                    ww = fa_click['points'][0]["x"]
                    # HERE ww-1
                    if overlap_mode == "overlapping":
                        fig.add_trace(go.Bar(x=np.arange(ww, L, w_s), y=model[ngram].counts[ww], name="∑∆w"))
                    else:
                        # Для non-overlapping режиму потрібно розрахувати положення барів
                        bar_positions = []
                        k = 1
                        i = 0
                        while i < L - ww:
                            bar_positions.append(i)
                            shift = calc_non_overlapping_shift(k, w_s, w_e)
                            i += shift
                            k += 1
                        fig.add_trace(go.Bar(x=bar_positions, y=model[ngram].counts[ww], name="∑∆w"))

                fa_click = None
                if graph_click:
                    print(model[ngram].sums.keys())

                graph_click = None

                fig1.add_trace(go.Scatter(x=df["R"], y=df["gamma"], mode="markers", text=hover_data))
                # fig1.add_trace(go.Scatter(x=[df['R'][active_cell['row']]],
                fig1.add_trace(go.Scatter(x=[df['R'][ids[active_cell['row']]]],
                                          # y=[df["b"][active_cell['row']]],
                                          y=[df["gamma"][ids[active_cell['row']]]],
                                          mode="markers",
                                          text=' '.join(ngram),
                                          marker=dict(
                                              size=20,
                                              color="red"
                                          )))
                fig1.update_layout(showlegend=False)
                fig1.update_yaxes(type=scale)
                fig1.update_xaxes(type=scale)
                fig1.update_layout(hovermode="x unified")
                active_cell = None

            return fig, fig1

    return dash.no_update, dash.no_update

@dash.callback(
    Output('ignore-comments-container', 'style'),
    Input('mode-selector', 'value')
)
def toggle_comment_container_visibility(value):
    if value == 'natural_text':
        return {'display': 'none'}
    return {'display': 'block'}



@dash.callback([Output("temp_seve", "children")],
              [Input("save", "n_clicks"),
               Input("table", "active_cell"),
               Input("table", "page_current"),
               Input("table", "derived_virtual_indices")],
              [State("file-selector", "value"),
               State("n_size", "value"),
               State("w_min", "value"),
               State("w_s", "value"),
               State("w_e", "value"),
               State("w_max", "value"),
               State("f_min", "value"),
               State("condition", "value"),
               State("def", "value"),
               State("min_dist_option", "value"),
               State("overlap_mode", "value")])
def save(n, active_cell, page_current, ids, filename, n_size, w_min, w_s, w_e, w_max, fmin, opt, definition, min_dist_option, overlap_mode):
    global save_folder
    if n is None:
        return dash.no_update
    if filename is None:
        return [html.Div(["No file selected to save"])]

    #print(active_cell)
    
    try:
        if save_folder is None or save_folder == "":
            pick_folder()
            if save_folder is None or save_folder == "":
                return dash.no_update

        file = filename
        global df, model, new_ngram

        # For dynamic mode, we want to save exactly what's shown in the table
        if definition == "dynamic":
            # Create DataFrame with the new_ngram row
            df_to_save = df.copy()  # This will include the new_ngram row
            
            output_filename = "{11}/{0} condition={7},fmin={1},n={2},w=({3},{4},{5},{6}),definition={8},min_dist={9},overlap={10}.xlsx".format(
                file, fmin, n_size, w_min, w_s, w_e, w_max, opt, definition, min_dist_option, overlap_mode, save_folder)
            
            # Ensure save directory exists
            os.makedirs(save_folder, exist_ok=True)
            
            # Save the main file with new_ngram data
            writer = pd.ExcelWriter(output_filename)
            #df_to_save.to_excel(writer, index=False)
            #writer.save()
            with pd.ExcelWriter(output_filename) as writer:
                    df_to_save.to_excel(writer, index=False)
            #writer.close()

            # If new_ngram exists and we have its details, save them too
            if active_cell:
                if new_ngram and hasattr(new_ngram, 'dfa'):
                    details_filename = "{}/{} new_ngram_details.xlsx".format(save_folder, file)
                    writer_details = pd.ExcelWriter(details_filename)
                    df_details = pd.DataFrame()
                    df_details["w"] = sorted(list(new_ngram.dfa.keys()))
                    #df_details['∆F'] = list(new_ngram.dfa.values())
                    df_details['∆F'] = [new_ngram.dfa[key] for key in sorted(list(new_ngram.dfa.keys()))]
                    df_details['fit=a*w^b'] = new_ngram.temp_dfa
                    #df_details.to_excel(writer_details, index=False)
                    
                    with pd.ExcelWriter(details_filename) as writer:
                            df_details.to_excel(writer, index=False)
                    #writer_details.save()
                    #writer_details.close()
                    return [html.Div([
                        "Saved main data to {}".format(output_filename),
                        html.Br(),
                        "Saved new_ngram details to {}".format(details_filename)
                    ])]
            
            return [html.Div(["Saved data to {}".format(output_filename)])]
        
        else:  # For non-dynamic mode, keep existing logic
            df_copy = df.copy()
            df_copy = df_copy[df_copy.ngram != 'new_ngram']
            df_copy['rank'] = range(1, len(df_copy) + 1)

            if len(df_copy) > 0:
                df_copy['w'] = (df_copy['F']) / (df_copy['F'].sum())

                R_avg = df_copy['R'].mean()
                dR = df_copy['R'].std()
                Rw_avg = (df_copy['R'] * df_copy['w']).sum()
                dRw = np.sqrt((((df_copy['R'] - Rw_avg) ** 2) * df_copy['w']).sum())

                gamma_avg = df_copy['gamma'].mean()
                dgamma = df_copy['gamma'].std()
                gammaw_avg = (df_copy['gamma'] * df_copy['w']).sum()
                dgammaw = np.sqrt((((df_copy['gamma'] - gammaw_avg) ** 2) * df_copy['w']).sum())

                df_copy.loc[:, 'R_avg'] = None
                df_copy.loc[df_copy.index[0], 'R_avg'] = R_avg
                df_copy.loc[:, 'dR'] = None
                df_copy.loc[df_copy.index[0], 'dR'] = dR
                df_copy.loc[:, 'Rw_avg'] = None
                df_copy.loc[df_copy.index[0], 'Rw_avg'] = Rw_avg
                df_copy.loc[:, 'dRw'] = None
                df_copy.loc[df_copy.index[0], 'dRw'] = dRw

                df_copy.loc[:, 'gamma_avg'] = None
                df_copy.loc[df_copy.index[0], 'gamma_avg'] = gamma_avg
                df_copy.loc[:, 'dgamma'] = None
                df_copy.loc[df_copy.index[0], 'dgamma'] = dgamma
                df_copy.loc[:, 'gammaw_avg'] = None
                df_copy.loc[df_copy.index[0], 'gammaw_avg'] = gammaw_avg
                df_copy.loc[:, 'dgammaw'] = None
                df_copy.loc[df_copy.index[0], 'dgammaw'] = dgammaw

                df_copy = df_copy.drop(columns=['w'])

            output_filename = "{11}/{0} condition={7},fmin={1},n={2},w=({3},{4},{5},{6}),definition={8},min_dist={9},overlap={10}.xlsx".format(
                file, fmin, n_size, w_min, w_s, w_e, w_max, opt, definition, min_dist_option, overlap_mode, save_folder)
            
            os.makedirs(save_folder, exist_ok=True)
            
            writer = pd.ExcelWriter(output_filename)
            #df_copy.to_excel(writer, index=False)
            with pd.ExcelWriter(output_filename) as writer:
                    df_copy.to_excel(writer, index=False)
            #writer.save()
            #writer.close()

            if active_cell:
                try:
                    row_index = active_cell['row']
                    if page_current is not None and page_current > 0:
                        row_index += page_current * 50

                    if ids is not None and row_index < len(ids):
                        selected_index = ids[row_index]
                        if selected_index < len(df):
                            ngram = df.iloc[selected_index]['ngram']
                            #if ngram != 'new_ngram' and ngram in model:
                            """if ngram == 'new_ngram' and ngram in model:
                                details_filename = "saved_data/{} {}_details.xlsx".format(file, ngram)
                                #writer_details = pd.ExcelWriter(details_filename)
                                df1 = pd.DataFrame()
                                df1["w"] = sorted(list(model[ngram].fa.keys()))
                                #df1['∆F'] = list(model[ngram].fa.values())
                                df1['∆F'] = [model[ngram].fa[key] for key in sorted(list(model[ngram].fa.keys()))]
                                df1['fit=a*w^b'] = model[ngram].temp_fa
                                #df1.to_excel(writer_details, index=False)
                                
                                with pd.ExcelWriter(details_filename) as writer:
                                        df1.to_excel(writer, index=False)
                                #writer_details.save()
                                #writer_details.close()
                                return [html.Div([
                                    "Saved main data to {}".format(output_filename),
                                    html.Br(),
                                    "Saved details to {}".format(details_filename)
                                ])]"""

                            details_filename = "{}/{} {}_details.xlsx".format(save_folder, file, ngram)
                            df1 = pd.DataFrame()
                            df1["w"] = sorted(list(model[ngram].fa.keys()))
                            df1['∆F'] = [model[ngram].fa[key] for key in sorted(list(model[ngram].fa.keys()))]
                            df1['fit=a*w^b'] = model[ngram].temp_fa
                            
                            with pd.ExcelWriter(details_filename) as writer:
                                    df1.to_excel(writer, index=False)
                           
                            return [html.Div([
                               "Saved main data to {}".format(output_filename),
                               html.Br(),
                               "Saved details to {}".format(details_filename)
                               ])]
                except Exception as e:
                    print("Error saving ngram details: {}".format(e))

            return [html.Div(["Saved data to {}".format(output_filename)])]
            
    except Exception as e:
        return [html.Div(["Error saving data: {}".format(str(e))])]


@dash.callback(
    [Output('output_folder_label', 'children')],
    [Input('pick_output_folder', 'n_clicks')]
)
def pick_output_folder(n):
    global save_folder
    if n is None:
        return dash.no_update
    pick_folder()
    if save_folder and save_folder is not None:
        return [html.Div(["Selected output folder as {} ".format(save_folder)]) ] 
    else:
        return [html.Div(["No output folder selected"])] 

