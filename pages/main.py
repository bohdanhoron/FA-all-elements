import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import dash_table

toast_visible = False
error_visible = False
analyze_visible = False

dash.register_page(__name__,
                   path='/',
                   title="FA all words")


layout = html.Div([
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
                                            dbc.Select(
                                                id="select-types-mode",
                                                options=[
                                                    {"label": "all types", "value": "true"},
                                                    {"label": "select types", "value": "false"}
                                                ],
                                                value="true"
                                            )],
                                            style={'marginBottom': '5px'}
                                        ),
                                    html.Div(id="select-types", style={"marginTop": "5px", "fontSize": "small", "textAlign": "center", "marginBottom": "10px"}),
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
