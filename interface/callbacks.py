import gc
import os
import re
import base64
from time import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import dash
from dash import no_update
from dash.dependencies import Input, Output, State
import dash_html_components as html
import plotly.graph_objs as go
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import chardet


from models.state_manager import state
from models.ngram import Ngram, newNgram
from core.math_utils import calculate_distance, fit, calc_non_overlapping_shift
from core.statistics import R, add_batch_statistics
from core.fa import calculate_rms
from core.dfa import calculate_rmsd
from core.legacy_indexer import make_markov_chain, make_dataframe
from utils.memory import clear_memory
from data.text_processing import prepare_data, remove_punctuation, is_valid_letter, NgrammProcessor
from data.loader import  pick_folder

def register_callbacks(app):
    """Реєструє всі callback-и для Dash додатку."""
    @app.callback(
        [Output('upload-status', 'children'),
         Output('file-selector', 'options'),
         Output('min-max-length-info', 'children')],
        [Input('upload-data', 'contents')],
        [State('mode-selector', 'value'),
         State('comments-selector', 'value'),
         State('upload-data', 'filename'),
         State('n_size', 'value'),
         State('split', 'value')]
    )
    def update_upload_status(contents, processor_mode, ignore_comments, filenames, n_size, split_mode):
        min_max_info = ""
        options = [{'label': filename, 'value': filename, 'title': filename} 
                   for filename in list(state.uploaded_files.keys())]
        
        if contents is None:
            if state.file_lengths and split_mode:
                lengths = [state.file_lengths[filename].get(split_mode, 0) for filename in state.file_lengths]
                if lengths:
                    min_len = min(lengths)
                    max_len = max(lengths)
                    split_label = "letters&numbers" if split_mode == 'letter' else f"{split_mode}s"
                    min_max_info = f"Min/Max Length ({split_label}): {min_len} / {max_len}"
            return html.Div(["No new files uploaded"]), options, html.Div(min_max_info)
        
        success_count = 0
        error_count = 0
        
        for i, (content, filename) in enumerate(zip(contents, filenames)):
            try:
                content_type, content_string = content.split(',')
                decoded = base64.b64decode(content_string)
                detection = chardet.detect(decoded)
                encoding = detection['encoding'] or 'windows-1251'

                try:
                    file_content = decoded.decode(encoding)
                    state.uploaded_files[filename] = file_content
                    state.file_lengths[filename] = {}
                    
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
                    state.file_lengths[filename]['word'] = len(words)
                    
                    # Symbol length
                    symbols = []
                    for char in file_content:
                        if char == " " or char == "\n" or char == "\ufeff":
                            symbols.append("space")
                        else:
                            symbols.append(char.lower())
                    state.file_lengths[filename]['symbol'] = len(symbols)
                    
                    # Letter length
                    text_letter = remove_punctuation(file_content)
                    letters = []
                    for word in text_letter:
                        for i in word:
                            if i == ' ':
                                continue
                            letters.append(i)
                    state.file_lengths[filename]['letter'] = len(letters)
                    
                    success_count += 1
                except UnicodeDecodeError:
                    error_count += 1
            except Exception as e:
                error_count += 1
        
        summary_message = html.Div([
            html.H5(f"Upload Summary:"),
            html.P(f"Successfully uploaded: {success_count} file(s)", style={'color': 'green'}),
            html.P(f"Files with errors: {error_count}", style={'color': 'red' if error_count > 0 else 'green'})
        ])
        
        options = [{'label': filename, 'value': filename, 'title': filename} 
                   for filename in list(state.uploaded_files.keys())]

        if state.file_lengths and split_mode:
            lengths = [state.file_lengths[filename].get(split_mode, 0) for filename in state.file_lengths]
            if lengths:
                min_len = min(lengths)
                max_len = max(lengths)
                split_label = "letters&numbers" if split_mode == 'letter' else f"{split_mode}s"
                min_max_info = f"Min/Max Length ({split_label}): {min_len} / {max_len}"
        
        return summary_message, options, html.Div(min_max_info)

    @app.callback(
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
        if selected_filename is None or selected_filename not in state.uploaded_files:
            return no_update, no_update, no_update, no_update, no_update
        
        file = state.uploaded_files[selected_filename]
        computer_code = True if processor_mode == 'computer_code' else False

        if definition == "dynamic":
            data = prepare_data(file, n, split, selected_filename, computer_code, ignore_comments)
            state.data = data
            state.L = len(data)
            w_max = int(state.L / 10)
            w_min = int(w_max / 10)
        else:
            if split == "letter":
                temp = []
                data = remove_punctuation(file)
                for word in data:
                    for i in word:
                        if i == ' ':
                            continue
                        temp.append(i)
                data = temp
                state.L = len(data)
            elif split == "symbol":
                temp = []
                for char in file:
                    if char == " " or char == "\n" or char == "\ufeff":
                        temp.append("space")
                    else:
                        temp.append(char.lower())
                data = temp
                state.L = len(data)
            elif split == "word":
                if not computer_code:
                    file = re.sub(r'\n+', '\n', file)
                    file = re.sub(r'\n\s\s', '\n', file)
                    file = re.sub(r'﻿', '', file)
                    file = re.sub(r'--', ' -', file)

                processor = NgrammProcessor(computer_code=computer_code, ignore_comments=ignore_comments)
                processor.preprocess(file, file_name=selected_filename)
                data = processor.get_words()
                state.L = len(data)

            state.data = data
            state.file_lengths[selected_filename][split] = state.L
            w_max = int(state.L / 20)
            w_min = int(w_max / 20)
        
        length_elements = [html.Strong("Length:")]
        lengths = state.file_lengths[selected_filename]
        
        if 'word' in lengths:
            length_elements.append(html.Div(f"words: {lengths['word']}"))
        if 'symbol' in lengths:
            length_elements.append(html.Div(f"symbols: {lengths['symbol']}"))
        if 'letter' in lengths:
            length_elements.append(html.Div(f"letters&numbers: {lengths['letter']}"))
            
        return length_elements, w_min, w_min, w_min, w_max

    @app.callback(
        [Output("table", "data"),
         Output("chain", "figure"),
         Output("box_tab", "style"),
         Output("box_chain", "style"),
         Output("alert", "children"),
         Output("v", "children"),
         Output("t", "children"),
         Output('click-toast', 'is_open')],
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
         State("condition", "value"),
         State("algo-selector", "value"),
         State("polynom_degree", "value")]
    )
    def update_table(n, dataframe, f_min, w_min, w_s, w_e, w_max, definition, 
                     min_dist_option, overlap_mode, n_size, split, condition, 
                     algo_selector, polynom_degree):
        
        if hasattr(prepare_data, 'clear_cache'):
            prepare_data.clear_cache()
        
        clear_memory()
        
        if n is None or dataframe is None:
            return (no_update, no_update, {"display": "none"}, {"display": "none"},
                    no_update, no_update, no_update, no_update)
        
        data = state.data
        L = state.L
        
        if data is None or L == 0:
            return (no_update, no_update, {"display": "none"}, {"display": "none"},
                    no_update, no_update, no_update, True)
                    
        if definition == "dynamic":
            start = time()
            
            w_max_val = int(w_max) if w_max is not None else int(L / 20)
            w_s_val = int(w_s) if w_s is not None else int(w_max_val / 20)
            w_e_val = int(w_e) if w_e is not None else w_s_val
            
            if w_e_val == 0:
                w_e_val = 5
            
            windows = list(range(w_s_val, w_max_val + 1, w_e_val))
            
            # Створення нового n-граму
            state.new_ngram = newNgram(data, w_s_val, L)
            new_ngram = state.new_ngram

            def process_window(w):
                if overlap_mode == "overlapping":
                    return new_ngram.func(w, algo_selector=algo_selector, polynom_degree=polynom_degree)
                else:
                    return new_ngram.func(w, overlap_mode=overlap_mode, min_window=w_s_val, 
                                         window_expansion=w_e_val, algo_selector=algo_selector, 
                                         polynom_degree=polynom_degree)
            
            if len(windows) > 4:
                with ThreadPoolExecutor(max_workers=min(4, len(windows))) as executor:
                    list(executor.map(process_window, windows))
            else:
                for w in windows:
                    process_window(w)
            
            temp_v = []
            temp_pos = []
            unique_items = set()
            
            for i, ngram in enumerate(data):
                if ngram not in unique_items:
                    unique_items.add(ngram)
                    temp_v.append(ngram)
                    temp_pos.append(i)
            
            temp_pos_array = np.array(temp_pos, dtype=np.uint32)
            ngram_for_calc = temp_v[0] if temp_v else "new_ngram"
            new_ngram.dt = calculate_distance(temp_pos_array, L, condition, ngram_for_calc, min_dist_option)
            new_ngram.R = round(R(new_ngram.dt), 8)

            try:
                dfa_keys = sorted(list(new_ngram.dfa.keys()))
                dfa_values = [new_ngram.dfa[key] for key in dfa_keys]
                
                if len(dfa_keys) < 5 or len(dfa_values) < 5:
                    new_ngram.a = 1.0
                    new_ngram.gamma = 0.5
                    new_ngram.temp_dfa = [1.0] * (len(dfa_keys) if dfa_keys else 1)
                    new_ngram.goodness = 0.0
                else:
                    c, _ = curve_fit(fit, dfa_keys, dfa_values, method='lm', maxfev=5000)
                    new_ngram.a = round(c[0], 8)
                    new_ngram.gamma = round(c[1], 8)
                    new_ngram.temp_dfa = [fit(w, new_ngram.a, new_ngram.gamma) for w in dfa_keys]
                    new_ngram.goodness = round(r2_score(dfa_values, new_ngram.temp_dfa), 8)

                del dfa_keys, dfa_values
            except Exception as e:
                new_ngram.a = 1.0
                new_ngram.gamma = 0.5
                new_ngram.temp_dfa = []
                new_ngram.goodness = 0.0

            df = pd.DataFrame({
                'rank': [1],
                'ngram': ['new_ngram'],
                'F': [len(temp_pos)],
                'R': [new_ngram.R],
                'a': [new_ngram.a],
                'gamma': [new_ngram.gamma],
                'goodness': [new_ngram.goodness]
            })

            state.df = df
            state.V = len(temp_v)
            
            end_time = time()
            execution_time = end_time - start
            
            df_table = df.to_dict("records")
            vocab_info = f"Vocabulary: {state.V}"
            time_info = f"Time: {execution_time:.4f} s"
            
            del temp_v, temp_pos, unique_items, temp_pos_array
            gc.collect()
            
            return (df_table, no_update, {"display": "inline"}, {"display": "none"},
                    no_update, vocab_info, time_info, False)
        else:
            # Static mode - Markov Chain
            start = time()
            
            model = make_markov_chain(tuple(data), order=n_size)
            state.model = model
            df = make_dataframe(model, f_min)

            w_max_val = int(w_max) if w_max is not None else int(L / 20)
            w_s_val = int(w_s) if w_s is not None else int(w_max_val / 20)
            w_e_val = int(w_e) if w_e is not None else w_s_val
            
            if w_e_val == 0:
                w_e_val = 5
            
            windows = list(range(w_s_val, w_max_val + 1, w_e_val))

            def process_ngram(ngram_data):
                ngram, index = ngram_data

                dt = calculate_distance(np.array(model[ngram].pos, dtype=np.uint32), L, condition, ngram, min_dist_option)
                model[ngram].dt = dt

                for wind in windows:
                    if algo_selector == '2':
                        rms, fa_val = calculate_rmsd(model[ngram].bool, wi=wind, polynom_degree=polynom_degree)
                    else:
                        rms, fa_val = calculate_rms(model[ngram].bool, wi=wind, l=L, wsh=w_s_val, 
                                                   overlap_mode=overlap_mode, min_window=w_s_val, 
                                                   window_expansion=w_e_val)

                    model[ngram].counts[wind] = rms
                    model[ngram].fa[wind] = fa_val

                try:
                    ff = [*model[ngram].fa.values()]
                    c, _ = curve_fit(fit, windows, ff, method='lm', maxfev=5000)
                    
                    a_val = c[0]
                    gamma_val = c[1]
                    temp_fa = [fit(w_val, a_val, gamma_val) for w_val in windows]
                    
                    model[ngram].a = a_val
                    model[ngram].gamma = gamma_val
                    model[ngram].temp_fa = temp_fa
                    
                    r_val = round(R(dt), 8)
                    model[ngram].R = r_val
                    
                    return {
                        'ngram': ngram,
                        'a': round(a_val, 8),
                        'gamma': round(gamma_val, 8),
                        'error': round(r2_score(ff, temp_fa), 5),
                        'R': r_val
                    }
                except Exception as e:
                    model[ngram].a = 0
                    model[ngram].gamma = 0
                    model[ngram].temp_fa = [0] * len(windows)
                    r_val = round(R(dt), 8)
                    model[ngram].R = r_val

                    return {
                        'ngram': ngram,
                        'a': 0,
                        'gamma': 0,
                        'error': 0,
                        'R': r_val
                    }
            
            ngram_items = [(ngram, i) for i, ngram in enumerate(df["ngram"])]
            max_workers = min(4, len(ngram_items))

            results = []
            if len(ngram_items) >= 4:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    results = list(executor.map(process_ngram, ngram_items))
            else:
                results = [process_ngram(item) for item in ngram_items]
            
            temp_a = [result['a'] for result in results]
            temp_gamma = [result['gamma'] for result in results]
            temp_error = [result['error'] for result in results]
            temp_R = [result['R'] for result in results]

            if n_size > 1:
                temp_ngram = []
                for ng in df['ngram']:
                    if isinstance(ng, tuple):
                        temp_ngram.append(" ".join(ng))
                    else:
                        temp_ngram.append(ng)
                df["ngram"] = temp_ngram
            
            df['R'] = temp_R
            df['gamma'] = temp_gamma
            df['a'] = temp_a
            df['goodness'] = temp_error
            df = df.sort_values(by="F", ascending=False)
            df['rank'] = range(1, len(temp_R) + 1)
            df = df.set_index(pd.Index(np.arange(len(df))))
            
            state.df = df
            state.V = len(model)
            
            end_time = time()
            execution_time = end_time - start
            
            df_table = df.to_dict("records")
            vocab_info = f"Vocabulary: {state.V}"
            time_info = f"Time: {execution_time:.4f} s"
            
            del temp_gamma, temp_R, temp_error, temp_a, results, ngram_items
            gc.collect()
            
            return (df_table, no_update, {"display": "inline"}, {"display": "none"},
                    no_update, vocab_info, time_info, False)

    @app.callback(
        [Output("graphs", "figure"), Output("fa", "figure")],
        [Input("dataframe", "active_tab"),
         Input("card-tabs", "active_tab"),
         Input("table", "active_cell"),
         Input("table", "page_current"),
         Input("table", "derived_virtual_selected_rows"),
         Input("table", "derived_virtual_indices"),
         Input("chain", "clickData"),
         Input("scale", "value"),
         Input("fa", "clickData"),
         Input("graphs", "clickData"),
         Input("w_max", "value")],
        [State("n_size", "value"),
         State("def", "value"),
         State("w_s", "value"),
         State("w_e", "value"),
         State("overlap_mode", "value")]
    )
    def tab_content(active_tab2, active_tab1, active_cell, page_current, row_ids, ids, 
                    clicked_data, scale, fa_click, graph_click, w_max, n, definition, w_s, w_e, overlap_mode):
        
        if active_tab2 == "data_table":
            fig = go.Figure()
            fig1 = go.Figure()
            
            model = state.model
            df = state.df
            new_ngram = state.new_ngram
            L = state.L
            
            if active_tab1 == "tab2":
                if active_cell:
                    if definition == "dynamic":
                        if fa_click and new_ngram is not None:
                            if overlap_mode == "overlapping":
                                fig.add_trace(go.Bar(x=np.arange(w_s, L, w_s), 
                                                    y=new_ngram.count[fa_click["points"][0]["x"]],
                                                    name="∑∆w"))
                            else:
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

                        if new_ngram is not None and hasattr(new_ngram, 'dfa') and new_ngram.dfa:
                            fig1.add_trace(
                                go.Scatter(x=[*new_ngram.dfa.keys()], y=[*new_ngram.dfa.values()], 
                                          mode='markers', name="∆F"))
                            
                            if hasattr(new_ngram, 'temp_dfa') and new_ngram.temp_dfa:
                                fig1.add_trace(go.Scatter(x=[*sorted(new_ngram.dfa.keys())], 
                                                         y=[*new_ngram.temp_dfa], name="fit=aw^b"))
                            
                            fig1.update_xaxes(type=scale)
                            fig1.update_yaxes(type=scale)
                            fig1.update_layout(hovermode="x unified")

                        return fig, fig1

                    if model is None or df is None:
                        return fig, fig1
                        
                    if n > 1:
                        ngram = tuple(df['ngram'][ids[active_cell['row']]].split())
                        if ngram[0] == 'new_ngram':
                            ngram = 'new_ngram'
                    else:
                        ngram = df['ngram'][ids[active_cell['row']]]
                    
                    if ngram in model:
                        fig.add_trace(go.Scatter(x=np.arange(L), y=model[ngram].bool, name="positions"))

                        if fa_click:
                            ww = fa_click["points"][0]["x"]
                            if overlap_mode == "overlapping":
                                fig.add_trace(go.Bar(x=np.arange(w_s, L, w_s), 
                                                    y=model[ngram].counts[ww], name="∑∆w"))
                            else:
                                bar_positions = []
                                k = 1
                                i = 0
                                while i < L - ww:
                                    bar_positions.append(i)
                                    shift = calc_non_overlapping_shift(k, w_s, w_e)
                                    i += shift
                                    k += 1
                                fig.add_trace(go.Bar(x=bar_positions, y=model[ngram].counts[ww], name="∑∆w"))

                        temp_ww = [*model[ngram].fa.keys()]
                        fig1.add_trace(
                            go.Scatter(x=temp_ww, y=[*model[ngram].fa.values()],
                                      mode='markers', name="∆F"))
                        fig1.add_trace(go.Scatter(x=temp_ww, y=model[ngram].temp_fa, name="fit=aw^b"))
                        fig1.update_xaxes(type=scale)
                        fig1.update_yaxes(type=scale)
                        fig1.update_layout(hovermode="x unified")
                    
                    return fig, fig1
                else:
                    return fig, fig1
            else:
                # tab3 - gamma vs R
                import numbers
                hover_data = []
                if active_cell:
                    if definition == "dynamic":
                        if new_ngram is not None and hasattr(new_ngram, 'R') and hasattr(new_ngram, 'gamma'):
                            fig1.add_trace(go.Scatter(x=[new_ngram.R], y=[new_ngram.gamma], 
                                                     mode='markers', text=["new_ngram"]))
                            fig1.update_xaxes(type=scale)
                            fig1.update_yaxes(type=scale)
                            fig1.update_layout(hovermode="x unified")
                        return fig, fig1

                    if model is None or df is None:
                        return fig, fig1
                        
                    if n > 1:
                        ngram = tuple(df['ngram'][ids[active_cell['row']]].split())
                        if ngram[0] == 'new_ngram':
                            ngram = 'new_ngram'
                    else:
                        ngram = df['ngram'][ids[active_cell['row']]]

                    for data_item in df['ngram']:
                        if not isinstance(data_item, numbers.Number):
                            hover_data.append("".join(str(data_item)))
                    
                    if ngram in model:
                        fig.add_trace(go.Scatter(x=np.arange(L), y=model[ngram].bool, name="positions"))
                        
                        if fa_click:
                            ww = fa_click['points'][0]["x"]
                            if overlap_mode == "overlapping":
                                fig.add_trace(go.Bar(x=np.arange(ww, L, w_s), 
                                                    y=model[ngram].counts[ww], name="∑∆w"))
                            else:
                                bar_positions = []
                                k = 1
                                i = 0
                                while i < L - ww:
                                    bar_positions.append(i)
                                    shift = calc_non_overlapping_shift(k, w_s, w_e)
                                    i += shift
                                    k += 1
                                fig.add_trace(go.Bar(x=bar_positions, y=model[ngram].counts[ww], name="∑∆w"))

                    fig1.add_trace(go.Scatter(x=df["R"], y=df["gamma"], mode="markers", text=hover_data))
                    fig1.add_trace(go.Scatter(x=[df['R'][ids[active_cell['row']]]],
                                              y=[df["gamma"][ids[active_cell['row']]]],
                                              mode="markers",
                                              text=' '.join(str(ngram)) if isinstance(ngram, tuple) else str(ngram),
                                              marker=dict(size=20, color="red")))
                    fig1.update_layout(showlegend=False)
                    fig1.update_yaxes(type=scale)
                    fig1.update_xaxes(type=scale)
                    fig1.update_layout(hovermode="x unified")

                return fig, fig1

        return no_update, no_update

    @app.callback(
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
         State("batch_window_mode", "value"),
         State("algo-selector", "value"),
         State("polynom_degree", "value")]
    )
    def process_all_files(n_clicks, processor_mode, ignore_comments, fmin1, fmin2, split, n_size, 
                          condition, definition, min_dist_option, overlap_mode, w_min, w_s, w_e, 
                          w_max, batch_window_mode, algo_selector, polynom_degree):
        
        if n_clicks is None or not state.uploaded_files:
            return [], {"display": "none"}
        
        lengths = [state.file_lengths[filename][split] for filename in list(state.uploaded_files.keys())]
        if not lengths:
            return [], {"display": "none"}
            
        lmin = min(lengths)
        lmax = max(lengths)
        
        batch_results = []
        
        file_list = list(state.uploaded_files.items())
        computer_code = True if processor_mode == 'computer_code' else False
        
        for idx, (filename, file_content) in enumerate(file_list, 1):
            gc.collect()
            
            file_length = state.file_lengths[filename][split]
            if lmin == lmax:
                f_min = fmin1
            else:
                f_min = fmin1 + (fmin2 - fmin1) * (file_length - lmin) / (lmax - lmin)
                f_min = round(f_min)
            
            start_time = time()
            
            data = None
            local_model = {}
            
            if definition == "dynamic":
                data = prepare_data(file_content, n_size, split, filename, computer_code, ignore_comments)
            else:
                if split == "letter":
                    file_text = re.sub(r'	', '', file_content)
                    processed_data = remove_punctuation(file_text)
                    temp = []
                    
                    for char in processed_data:
                        if char.isspace() or char == '\n' or char == '\ufeff':
                            continue
                        if char.isdigit() or char.isalpha():
                            temp.append(char)
                    
                    data = temp
                    del processed_data, temp, file_text
                    gc.collect()
                elif split == "symbol":
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
                    del temp, clean_text
                    gc.collect()
                elif split == "word":
                    file_text = file_content
                    if not computer_code:
                        file_text = re.sub(r'\n+', '\n', file_content)
                        file_text = re.sub(r'\n\s\s', '\n', file_text)
                        file_text = re.sub(r'﻿', '', file_text)
                        file_text = re.sub(r'--', ' -', file_text)

                    processor = NgrammProcessor(computer_code=computer_code, ignore_comments=ignore_comments)
                    processor.preprocess(file_text, file_name=filename)
                    data = processor.get_words()
                    del processor, file_text
                    gc.collect()

            L = len(data)
            
            if batch_window_mode == "ui":
                wm_val = int(w_max) if w_max is not None else int(L / 20)
                w_val = int(w_s) if w_s is not None else int(wm_val / 20)
                wh_val = int(w_s) if w_s is not None else w_val
                we_val = int(w_e) if w_e is not None else w_val
            else:
                if definition == "dynamic":
                    wm_val = int(L / 20)
                    w_val = int(wm_val / 20)
                else:
                    wm_val = int(L / 20)
                    w_val = int(wm_val / 20)
                wh_val = w_val
                we_val = w_val
                
            wm_val = max(10, wm_val)
            w_val = max(5, w_val)
            wh_val = max(1, wh_val)
            we_val = max(1, we_val)
            
            # Build local model
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
            
            V = len(local_model)

            if definition == "static":
                filtered_data = list(filter(lambda x: len(local_model[x].pos) >= f_min, local_model))
                
                data_df = {"ngram": [], "F": np.empty(len(filtered_data), dtype=np.int32)}
                for i, ngram in enumerate(filtered_data):
                    data_df["ngram"].append(ngram)
                    data_df["F"][i] = len(local_model[ngram].pos)
                
                current_df = pd.DataFrame(data=data_df)
                
                temp_gamma = []
                temp_R = []
                temp_error = []
                temp_a = []

                windows = list(range(w_val, wm_val + 1, we_val))
                
                for i, row in current_df.iterrows():
                    ngram = row['ngram']
                    
                    local_model[ngram].bool = np.zeros(L, dtype=np.int8)
                    for pos in local_model[ngram].pos:
                        local_model[ngram].bool[pos] = 1
                    
                    min_dist_int = int(min_dist_option) if isinstance(min_dist_option, (str, float)) else min_dist_option
                    local_model[ngram].dt = calculate_distance(np.array(local_model[ngram].pos, dtype=np.uint32), 
                                                               L, condition, ngram, min_dist_int)
                    
                    local_model[ngram].fa = {}
                    local_model[ngram].counts = {}
                    
                    for wind in windows:
                        if algo_selector == '2':
                            rms, fa_val = calculate_rmsd(local_model[ngram].bool, wi=wind, polynom_degree=polynom_degree)
                        else:
                            rms, fa_val = calculate_rms(local_model[ngram].bool, wi=wind, l=L, wsh=wh_val, 
                                                       overlap_mode=overlap_mode, min_window=w_val, 
                                                       window_expansion=we_val)
                        local_model[ngram].counts[wind] = rms
                        local_model[ngram].fa[wind] = fa_val

                    ff = [*local_model[ngram].fa.values()]

                    try:
                        c, _ = curve_fit(fit, windows, ff, method='lm', maxfev=5000)
                        a_val = c[0]
                        gamma_val = c[1]
                        temp_fa = [fit(w_val, c[0], c[1]) for w_val in windows]
                        temp_error.append(round(r2_score(ff, temp_fa), 5))
                        temp_gamma.append(round(gamma_val, 8))
                        temp_a.append(round(a_val, 8))
                    except:
                        temp_error.append(0)
                        temp_gamma.append(0)
                        temp_a.append(0)
                    
                    r = round(R(np.array(local_model[ngram].dt)), 8)
                    temp_R.append(r)
                
                if n_size > 1:
                    temp_ngram = []
                    for ng in current_df['ngram']:
                        if isinstance(ng, tuple):
                            temp_ngram.append(" ".join(ng))
                        else:
                            temp_ngram.append(ng)
                    current_df["ngram"] = temp_ngram
                
                current_df['R'] = temp_R
                current_df['gamma'] = temp_gamma
                current_df['a'] = temp_a
                current_df['goodness'] = temp_error
                current_df = current_df.sort_values(by="F", ascending=False)
                current_df['rank'] = range(1, len(current_df) + 1)
                current_df = current_df.set_index(pd.Index(np.arange(len(current_df))))
                
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
                    R_avg = dR = Rw_avg = dRw = gamma_avg = dgamma = gammaw_avg = dgammaw = 0

                del current_df, df_filtered, temp_gamma, temp_R, temp_error, temp_a

            elif definition == "dynamic":
                w_max_val = int(L / 20)
                w_s_val = int(w_max_val / 20)
                w_e_val = w_s_val
                
                if w_e_val == 0:
                    w_e_val = 5
                
                windows = list(range(w_s_val, w_max_val + 1, w_e_val))

                local_new_ngram = newNgram(data, w_s_val, L)
                
                def process_window(w):
                    if overlap_mode == "overlapping":
                        return local_new_ngram.func(w, algo_selector=algo_selector, polynom_degree=polynom_degree)
                    else:
                        return local_new_ngram.func(w, overlap_mode=overlap_mode, min_window=w_s_val, 
                                                   window_expansion=w_e_val, algo_selector=algo_selector, 
                                                   polynom_degree=polynom_degree)

                if len(windows) > 4:
                    with ThreadPoolExecutor(max_workers=min(4, len(windows))) as executor:
                        list(executor.map(process_window, windows))
                else:
                    for w in windows:
                        process_window(w)
                
                temp_v = []
                temp_pos = []
                unique_items = set()
                
                for i, ngram in enumerate(data):
                    if ngram not in unique_items:
                        unique_items.add(ngram)
                        temp_v.append(ngram)
                        temp_pos.append(i)
                
                temp_pos_array = np.array(temp_pos, dtype=np.uint32)
                ngram_for_calc = temp_v[0] if temp_v else "new_ngram"
                local_new_ngram.dt = calculate_distance(temp_pos_array, L, condition, ngram_for_calc, min_dist_option)
                local_new_ngram.R = round(R(local_new_ngram.dt), 8)

                try:
                    dfa_keys = sorted(list(local_new_ngram.dfa.keys()))
                    dfa_values = [local_new_ngram.dfa[key] for key in dfa_keys]
                    
                    if len(dfa_keys) < 2 or len(dfa_values) < 2:
                        local_new_ngram.a = 1.0
                        local_new_ngram.gamma = 0.5
                        local_new_ngram.temp_dfa = [1.0] * (len(dfa_keys) if dfa_keys else 1)
                        local_new_ngram.goodness = 0.0
                    else:
                        c, _ = curve_fit(fit, dfa_keys, dfa_values, method='lm', maxfev=5000)
                        local_new_ngram.a = round(c[0], 8)
                        local_new_ngram.gamma = round(c[1], 8)
                        local_new_ngram.temp_dfa = [fit(w, local_new_ngram.a, local_new_ngram.gamma) for w in dfa_keys]
                        local_new_ngram.goodness = round(r2_score(dfa_values, local_new_ngram.temp_dfa), 8)
                    
                    del dfa_keys, dfa_values
                except Exception as e:
                    local_new_ngram.a = 1.0
                    local_new_ngram.gamma = 0.5
                    local_new_ngram.temp_dfa = []
                    local_new_ngram.goodness = 0.0
                
                V = len(temp_v)

                R_avg = local_new_ngram.R
                dR = 0
                Rw_avg = 0
                dRw = 0

                gamma_avg = local_new_ngram.gamma
                dgamma = 0
                gammaw_avg = 0
                dgammaw = 0
                
                del temp_v, temp_pos, unique_items, temp_pos_array
                gc.collect()

            end_time = time()
            execution_time = end_time - start_time
            
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
            
            batch_results.append(batch_result)
            
            del data, local_model
            gc.collect()
        
        if batch_results:
            add_batch_statistics(batch_results)
        
        state.batch_results = batch_results
        
        return batch_results, {"display": "block"}

    @app.callback(
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

    @app.callback(
        Output("temp_seve_batch", "children"),
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
        if n_clicks is None:
            return no_update
        if not state.batch_results:
            return html.Div(["No batch results to save"])
        
        try:
            if state.save_folder is None or state.save_folder == "":
                pick_folder()
                if state.save_folder is None or state.save_folder == "":
                    return no_update
            
            df_batch = pd.DataFrame(state.batch_results)
            
            output_filename = "{}/batch_results_n={},split={},condition={},definition={},min_dist={},overlap={},window_mode={}.xlsx".format(
                state.save_folder, n_size, split, condition, definition, min_dist_option, overlap_mode, batch_window_mode)
                    
            os.makedirs(state.save_folder, exist_ok=True)
            
            with pd.ExcelWriter(output_filename) as writer:
                df_batch.to_excel(writer, index=False)
            
            return html.Div(["Saved batch results to {}".format(output_filename)])
        except Exception as e:
            return html.Div(["Error saving batch results: {}".format(str(e))])

    @app.callback(
        [Output("temp_seve", "children")],
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
         State("overlap_mode", "value")]
    )
    def save(n, active_cell, page_current, ids, filename, n_size, w_min, w_s, w_e, w_max, fmin, opt, definition, min_dist_option, overlap_mode):
        if n is None:
            return no_update
        if filename is None:
            return [html.Div(["No file selected to save"])]
        
        try:
            if state.save_folder is None or state.save_folder == "":
                pick_folder()
                if state.save_folder is None or state.save_folder == "":
                    return no_update

            df = state.df
            model = state.model
            new_ngram = state.new_ngram

            if definition == "dynamic":
                df_to_save = df.copy()
                
                output_filename = "{11}/{0} condition={7},fmin={1},n={2},w=({3},{4},{5},{6}),definition={8},min_dist={9},overlap={10}.xlsx".format(
                    filename, fmin, n_size, w_min, w_s, w_e, w_max, opt, definition, min_dist_option, overlap_mode, state.save_folder)
                
                os.makedirs(state.save_folder, exist_ok=True)
                
                with pd.ExcelWriter(output_filename) as writer:
                    df_to_save.to_excel(writer, index=False)

                if active_cell:
                    if new_ngram and hasattr(new_ngram, 'dfa'):
                        details_filename = "{}/{} new_ngram_details.xlsx".format(state.save_folder, filename)
                        df_details = pd.DataFrame()
                        df_details["w"] = sorted(list(new_ngram.dfa.keys()))
                        df_details['∆F'] = [new_ngram.dfa[key] for key in sorted(list(new_ngram.dfa.keys()))]
                        df_details['fit=a*w^b'] = new_ngram.temp_dfa
                        
                        with pd.ExcelWriter(details_filename) as writer:
                            df_details.to_excel(writer, index=False)
                        
                        return [html.Div([
                            "Saved main data to {}".format(output_filename),
                            html.Br(),
                            "Saved new_ngram details to {}".format(details_filename)
                        ])]
                
                return [html.Div(["Saved data to {}".format(output_filename)])]
            
            else:
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
                    filename, fmin, n_size, w_min, w_s, w_e, w_max, opt, definition, min_dist_option, overlap_mode, state.save_folder)
                
                os.makedirs(state.save_folder, exist_ok=True)
                
                with pd.ExcelWriter(output_filename) as writer:
                    df_copy.to_excel(writer, index=False)

                if active_cell:
                    try:
                        row_index = active_cell['row']
                        if page_current is not None and page_current > 0:
                            row_index += page_current * 50

                        if ids is not None and row_index < len(ids):
                            selected_index = ids[row_index]
                            if selected_index < len(df):
                                ngram = df.iloc[selected_index]['ngram']

                                details_filename = "{}/{} {}_details.xlsx".format(state.save_folder, filename, ngram)
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

    @app.callback(
        Output('ignore-comments-container', 'style'),
        Input('mode-selector', 'value')
    )
    def toggle_comment_container_visibility(value):
        if value == 'natural_text':
            return {'display': 'none'}
        return {'display': 'block'}

    @app.callback(
        Output('wrapper_polynom_degree', 'style'),
        Input('algo-selector', 'value')
    )
    def toggle_wrapper_polynom_degree_visibility(value):
        if value == '2':
            return {'display': 'block'}
        return {'display': 'none'}

    @app.callback(
        Output("batch_custom_controls", "is_open"),
        [Input("batch_window_mode", "value")]
    )
    def toggle_batch_window_controls(mode):
        return mode in ["ui", "auto"]