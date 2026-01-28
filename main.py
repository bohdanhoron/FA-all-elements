import dash
import webbrowser
from interface.layout import layout
from interface.callbacks import register_callbacks
from config import APP_TITLE, THEME, HOST, PORT, DEBUG

def create_app():
    app = dash.Dash(
        __name__, 
        external_stylesheets=[THEME],
        title=APP_TITLE,
        suppress_callback_exceptions=True
    )
    
    app.layout = layout
    
    register_callbacks(app)
    
    return app

if __name__ == "__main__":
    app = create_app()

    if not DEBUG:
        webbrowser.open_new(f"http://127.0.0.1:{PORT}/")

    app.run_server(host=HOST, port=PORT, debug=DEBUG)