import plotly.graph_objs as go
import numpy as np
import pandas as pd
from typing import Optional, List, Union

def create_positions_plot(l_val: int, bool_array: np.ndarray, ngram_name: str) -> go.Figure:
    """Генерує графік позицій n-грами в тексті."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(l_val), 
        y=bool_array, 
        name=f"positions: {ngram_name}",
        line=dict(color='#007bff')
    ))
    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        height=400,
        xaxis_title="Position in text",
        yaxis=dict(range=[-0.1, 1.1], tickvals=[0, 1])
    )
    return fig

def add_window_sums_to_fig(fig: go.Figure, x_positions: np.ndarray, counts: np.ndarray) -> go.Figure:
    """Додає бари сум у вікнах на існуючий графік позицій."""
    fig.add_trace(go.Bar(
        x=x_positions, 
        y=counts, 
        name="∑∆w",
        marker_color='rgba(255, 0, 0, 0.5)',
        yaxis="y2"
    ))
    fig.update_layout(
        yaxis2=dict(
            title="Window Sum",
            overlaying="y",
            side="right"
        ),
        showlegend=True
    )
    return fig

def create_fluctuation_plot(
    windows: List[int], 
    fa_values: List[float], 
    fit_values: Optional[List[float]] = None, 
    scale: str = "linear"
) -> go.Figure:
    """Генерує графік флуктуацій (∆F vs w)."""
    fig = go.Figure()
    
    # Експериментальні дані
    fig.add_trace(go.Scatter(
        x=windows, 
        y=fa_values, 
        mode='markers', 
        name="∆F",
        marker=dict(color='#28a745', size=8)
    ))
    
    # Апроксимація
    if fit_values is not None and len(fit_values) > 0:
        # Сортуємо для коректного відображення лінії
        sorted_indices = np.argsort(windows)
        fig.add_trace(go.Scatter(
            x=np.array(windows)[sorted_indices], 
            y=np.array(fit_values)[sorted_indices], 
            name="fit: aw^b",
            line=dict(color='#dc3545', dash='dash')
        ))
        
    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        height=400,
        xaxis_title="Window size (w)",
        yaxis_title="Fluctuation (F)",
        xaxis_type=scale,
        yaxis_type=scale,
        hovermode="x unified"
    )
    return fig

def create_gamma_r_plot(
    r_values: Union[List, np.ndarray], 
    gamma_values: Union[List, np.ndarray], 
    labels: List[str],
    highlight_index: Optional[int] = None,
    scale: str = "linear"
) -> go.Figure:
    """Генерує карту розсіювання gamma vs R."""
    fig = go.Figure()
    
    # Основні точки
    fig.add_trace(go.Scatter(
        x=r_values, 
        y=gamma_values, 
        mode="markers", 
        text=labels,
        name="n-grams",
        marker=dict(color='#6c757d', opacity=0.6)
    ))
    
    # Виділена точка (активна комірка)
    if highlight_index is not None and 0 <= highlight_index < len(r_values):
        fig.add_trace(go.Scatter(
            x=[r_values[highlight_index]],
            y=[gamma_values[highlight_index]],
            mode="markers",
            text=[labels[highlight_index]],
            marker=dict(size=15, color="red", symbol="circle-open", line=dict(width=2)),
            name="Selected"
        ))
        
    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        height=400,
        xaxis_title="Coefficient of variation (R)",
        yaxis_title="Scaling exponent (gamma)",
        xaxis_type=scale,
        yaxis_type=scale,
        showlegend=False
    )
    return fig