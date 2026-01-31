import numpy as np
import plotly.graph_objs as go
from typing import List, Optional, Dict, Any, Tuple

from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

from core.math_utils import fit


def create_empty_figure(message: str = "No data to display") -> go.Figure:
    """
    Створює порожній графік з повідомленням.
    
    Args:
        message: Повідомлення для відображення
        
    Returns:
        go.Figure: Порожній графік
    """
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray")
    )
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white'
    )
    return fig


def create_distribution_plot(positions: List[int], L: int, ngram: str) -> go.Figure:
    """
    Створює графік розподілу n-грами в тексті.
    
    Args:
        positions: Список позицій n-грами
        L: Довжина тексту
        ngram: Назва n-грами
        
    Returns:
        go.Figure: Графік розподілу
    """
    if not positions or L == 0:
        return create_empty_figure("No positions data")
 
    binary_array = np.zeros(L, dtype=np.uint8)
    for pos in positions:
        if 0 <= pos < L:
            binary_array[pos] = 1
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(L)),
        y=binary_array,
        mode='markers',
        marker=dict(
            size=3,
            color='#007bff',
            opacity=0.7
        ),
        name=f'"{ngram}" positions'
    ))
    
    fig.update_layout(
        title=dict(
            text=f'Distribution of "{ngram}" in text',
            font=dict(size=14, color='#333')
        ),
        xaxis=dict(
            title='Position in text',
            showgrid=True,
            gridcolor='#eee'
        ),
        yaxis=dict(
            title='Occurrence',
            showgrid=True,
            gridcolor='#eee',
            tickvals=[0, 1],
            ticktext=['0', '1']
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=20, t=40, b=40)
    )
    
    return fig


def create_fluctuation_plot(
    window_sizes: np.ndarray,
    fa_values: np.ndarray,
    scale: str = "linear",
    ngram: str = "",
    show_fit: bool = True
) -> Tuple[go.Figure, Optional[float], Optional[float], Optional[float]]:
    """
    Створює графік флуктуацій FA/DFA.
    
    Args:
        window_sizes: Масив розмірів вікон
        fa_values: Масив значень FA
        scale: Тип шкали ("linear" або "log")
        ngram: Назва n-грами
        show_fit: Чи показувати лінію апроксимації
        
    Returns:
        Tuple[go.Figure, Optional[float], Optional[float], Optional[float]]: 
            (графік, gamma, a, r2)
    """
    if len(window_sizes) == 0 or len(fa_values) == 0:
        return create_empty_figure("No fluctuation data"), None, None, None
    
    fig = go.Figure()
    
    gamma = None
    a = None
    r2 = None
    
    if scale == "log":
        x_data = np.log10(window_sizes)
        y_data = np.log10(fa_values + 1e-10)  
        
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='markers',
            marker=dict(size=8, color='#007bff'),
            name='FA values (log)'
        ))
        
        if show_fit and len(x_data) > 1:
            try:
                coeffs = np.polyfit(x_data, y_data, 1)
                gamma = coeffs[0]
                a = 10 ** coeffs[1]
                
                fit_line = np.polyval(coeffs, x_data)
                r2 = r2_score(y_data, fit_line)
                
                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=fit_line,
                    mode='lines',
                    line=dict(color='red', width=2),
                    name=f'Fit: γ={gamma:.4f}, R²={r2:.4f}'
                ))
            except Exception:
                pass
        
        fig.update_layout(
            xaxis_title='log₁₀(Window Size)',
            yaxis_title='log₁₀(FA)'
        )
    else:
        # Лінійний масштаб
        fig.add_trace(go.Scatter(
            x=window_sizes,
            y=fa_values,
            mode='markers',
            marker=dict(size=8, color='#007bff'),
            name='FA values'
        ))
        
        if show_fit and len(window_sizes) > 1:
            try:
                popt, _ = curve_fit(fit, window_sizes, fa_values, p0=[1, 0.5], maxfev=5000)
                a, gamma = popt
                
                fit_values = fit(window_sizes, *popt)
                r2 = r2_score(fa_values, fit_values)
 
                x_smooth = np.linspace(window_sizes.min(), window_sizes.max(), 100)
                y_smooth = fit(x_smooth, *popt)
                
                fig.add_trace(go.Scatter(
                    x=x_smooth,
                    y=y_smooth,
                    mode='lines',
                    line=dict(color='red', width=2),
                    name=f'Fit: γ={gamma:.4f}, R²={r2:.4f}'
                ))
            except Exception:
                pass
        
        fig.update_layout(
            xaxis_title='Window Size',
            yaxis_title='FA'
        )
    
    fig.update_layout(
        title=dict(
            text=f'Fluctuation Analysis: "{ngram}"' if ngram else 'Fluctuation Analysis',
            font=dict(size=14, color='#333')
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(l=50, r=20, t=40, b=40)
    )
    
    return fig, gamma, a, r2


def create_gamma_r_plot(
    r_values: List[float],
    gamma_values: List[float],
    ngram_labels: Optional[List[str]] = None
) -> go.Figure:
    """
    Створює графік gamma vs R.
    
    Args:
        r_values: Список значень R
        gamma_values: Список значень gamma
        ngram_labels: Опціональні мітки для точок
        
    Returns:
        go.Figure: Графік gamma vs R
    """
    if not r_values or not gamma_values:
        return create_empty_figure("No gamma-R data")
    
    fig = go.Figure()
    
    hover_text = ngram_labels if ngram_labels else [f"Point {i}" for i in range(len(r_values))]
    
    fig.add_trace(go.Scatter(
        x=r_values,
        y=gamma_values,
        mode='markers',
        marker=dict(
            size=10,
            color=gamma_values,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='γ')
        ),
        text=hover_text,
        hovertemplate='<b>%{text}</b><br>R: %{x:.4f}<br>γ: %{y:.4f}<extra></extra>',
        name='n-grams'
    ))
    
    fig.update_layout(
        title=dict(
            text='Gamma vs R',
            font=dict(size=14, color='#333')
        ),
        xaxis=dict(
            title='R (Coefficient of Variation)',
            showgrid=True,
            gridcolor='#eee'
        ),
        yaxis=dict(
            title='γ (Gamma)',
            showgrid=True,
            gridcolor='#eee'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=20, t=40, b=40)
    )
    
    return fig


def create_batch_comparison_plot(
    batch_results: List[Dict[str, Any]],
    metric: str = "gamma_avg"
) -> go.Figure:
    """
    Створює графік порівняння результатів batch processing.
    
    Args:
        batch_results: Список результатів batch processing
        metric: Метрика для порівняння
        
    Returns:
        go.Figure: Графік порівняння
    """
    if not batch_results:
        return create_empty_figure("No batch results")
    
    filenames = [r.get('filename', f'File {i}') for i, r in enumerate(batch_results)]
    values = [r.get(metric, 0) for r in batch_results]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=filenames,
        y=values,
        marker_color='#007bff',
        name=metric
    ))
    
    fig.update_layout(
        title=dict(
            text=f'Batch Comparison: {metric}',
            font=dict(size=14, color='#333')
        ),
        xaxis=dict(
            title='File',
            tickangle=45
        ),
        yaxis=dict(
            title=metric
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=20, t=40, b=80)
    )
    
    return fig