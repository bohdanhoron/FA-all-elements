import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import pandas as pd
import numpy as np
from time import time
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

from models.state_manager import state_manager
from data.loader import load_files_to_state, get_upload_summary_layout
from data.text_processing import prepare_data, NgrammProcessor, remove_punctuation
from core.legacy_indexer import build_static_index
from core.math_utils import calculate_distance, calculate_rms
from core.statistics import R, fit
from interface.plots import (create_positions_plot, add_window_sums_to_fig, 
                             create_fluctuation_plot, create_gamma_r_plot)
from utils.memory import clear_memory

def register_callbacks(app):

    @app.callback(
        [Output('upload-status', 'children'),
         Output('file-selector', 'options'),
         Output('min-max-length-info', 'children')],
        [Input('upload-data', 'contents')],
        [State('mode-selector', 'value'),
         State('comments-selector', 'value'),
         State('upload-data', 'filename'),
         State('split', 'value')]
    )
    def update_upload_status(contents, processor_mode, ignore_comments, filenames, split_mode):
        if contents is None:
            options = [{'label': f, 'value': f} for f in state_manager.uploaded_files.keys()]
            return html.Div("No files"), options, ""

        success, errors = load_files_to_state(contents, filenames, processor_mode, ignore_comments)
        
        options = [{'label': f, 'value': f} for f in state_manager.uploaded_files.keys()]
        lengths = [state_manager.file_lengths[f].get(split_mode, 0) for f in state_manager.file_lengths]
        min_max = f"Min/Max: {min(lengths)} / {max(lengths)}" if lengths else ""
        
        return get_upload_summary_layout(success, errors), options, html.Div(min_max)

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
        if not selected_filename or selected_filename not in state_manager.uploaded_files:
            return dash.no_update
        
        content = state_manager.uploaded_files[selected_filename]
        comp_code = (processor_mode == 'computer_code')
        
        state_manager.data = prepare_data(content, n, split, selected_filename, comp_code, ignore_comments)
        state_manager.L = len(state_manager.data)
        
        w_max = int(state_manager.L / 10) if definition == "dynamic" else int(state_manager.L / 20)
        w_min = int(w_max / 10) if definition == "dynamic" else int(w_max / 20)
        
        lengths = state_manager.file_lengths.get(selected_filename, {})
        res_html = [html.Strong("Length: ")] + [html.Div(f"{k}: {v}") for k, v in lengths.items()]
        
        return res_html, w_min, w_min, w_min, w_max

    @app.callback(
        [Output("table", "data"),
         Output("v", "children"),
         Output("t", "children"),
         Output("box_tab", "style")],
        [Input("chain_button", "n_clicks")],
        [State("f_min", "value"),
         State("n_size", "value"),
         State("w_min", "value"),
         State("w_s", "value"),
         State("w_e", "value"),
         State("w_max", "value"),
         State("overlap_mode", "value"),
         State("condition", "value"),
         State("min_dist_option", "value")]
    )
    def update_table(n_clicks, f_min, n_size, w_min, w_s, w_e, w_max, overlap, cond, m_dist):
        if not n_clicks: return dash.no_update
        
        start_t = time()
        clear_memory(keep=['data', 'uploaded_files', 'file_lengths'])
        
        state_manager.model, state_manager.L, state_manager.V = build_static_index(state_manager.data, n_size)
        
        windows = list(range(w_min, w_max + 1, w_s))
        rows = []
        
        for ngram, obj in state_manager.model.items():
            if ngram == 'new_ngram' or len(obj.pos) < f_min: continue
            
            obj.dt = calculate_distance(np.array(obj.pos, dtype=np.int32), state_manager.L, cond, str(ngram), int(m_dist))
            obj.fa = {}
            for w in windows:
                _, f_val = calculate_rms(obj.bool, w, state_manager.L, w_s, overlap, w_min, w_e)
                obj.fa[w] = f_val
            
            y_vals = list(obj.fa.values())
            c, _ = curve_fit(fit, windows, y_vals, maxfev=5000)
            obj.a, obj.gamma = c[0], c[1]
            obj.temp_fa = [fit(w, c[0], c[1]) for w in windows]
            obj.R = R(obj.dt)
            
            rows.append({
                "ngram": " ".join(ngram) if isinstance(ngram, tuple) else ngram,
                "F": len(obj.pos),
                "R": round(obj.R, 6),
                "gamma": round(obj.gamma, 6),
                "a": round(obj.a, 6),
                "goodness": round(r2_score(y_vals, obj.temp_fa), 4)
            })
            
        state_manager.df = pd.DataFrame(rows).sort_values("F", ascending=False)
        state_manager.df['rank'] = range(1, len(state_manager.df) + 1)
        
        return state_manager.df.to_dict("records"), f"Vocabulary: {state_manager.V}", f"Time: {time()-start_t:.2f}s", {"display":"block"}

    @app.callback(
        [Output("graphs", "figure"), Output("fa", "figure")],
        [Input("table", "active_cell"), Input("scale", "value")],
        [State("n_size", "value"), State("w_s", "value")]
    )
    def update_plots(cell, scale, n_size, w_s):
        if not cell or state_manager.df is None: return go.Figure(), go.Figure()
        
        idx = cell['row']
        ngram_str = state_manager.df.iloc[idx]['ngram']
        ngram_key = tuple(ngram_str.split()) if n_size > 1 else ngram_str
        obj = state_manager.model.get(ngram_key)
        
        fig_pos = create_positions_plot(state_manager.L, obj.bool, ngram_str)
        
        wins = list(obj.fa.keys())
        fas = list(obj.fa.values())
        fig_fa = create_fluctuation_plot(wins, fas, obj.temp_fa, scale)
        
        return fig_pos, fig_fa