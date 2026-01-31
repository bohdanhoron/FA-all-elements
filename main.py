import dash
import dash_bootstrap_components as dbc

from config import APP_TITLE, APP_DEBUG, APP_PORT, APP_HOST
from interface.layout import layout
from interface.callbacks import register_callbacks


def create_app() -> dash.Dash:
    """
    Створює та налаштовує Dash додаток.
    
    Returns:
        dash.Dash: Налаштований Dash додаток
    """
    # Ініціалізуємо Dash з Bootstrap темою
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        title=APP_TITLE,
        suppress_callback_exceptions=True
    )
    
    # Встановлюємо layout
    app.layout = layout
    
    # Реєструємо callbacks
    register_callbacks(app)
    
    return app


def main():
    """
    Головна функція запуску програми.
    """
    app = create_app()
    
    print(f"Starting {APP_TITLE}...")
    print(f"Open http://{APP_HOST}:{APP_PORT} in your browser")
    
    app.run_server(
        debug=APP_DEBUG,
        host=APP_HOST,
        port=APP_PORT
    )


if __name__ == "__main__":
    main()