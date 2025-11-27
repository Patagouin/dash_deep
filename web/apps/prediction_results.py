from dash import dcc, html

def get_results_layout():
    return html.Div([
        html.H4('Résultats du Modèle', style={
            'marginBottom': '0px',
            'padding': '20px',
            'color': '#FF8C00'
        }),

        # Sélection de modèles sauvegardés
        html.Div([
            html.Div([
                html.Label('Modèle sauvegardé'),
                dcc.Dropdown(
                    id='saved_model_dropdown',
                    options=[],
                    placeholder='Choisir un modèle entraîné pour l\'action sélectionnée',
                    style={'width': '100%', 'color': '#FF8C00'},
                    persistence=True, persistence_type='session'
                ),
            ], style={'flex': '3', 'minWidth': '260px'}),
            html.Div([
                html.Button('Charger le modèle', id='load_saved_model', n_clicks=0, className='update-button', style={'width': '100%'}),
            ], style={'flex': '1', 'minWidth': '160px', 'alignSelf': 'end'}),
            html.Div([
                html.Button('Mettre à jour (J-1)', id='update_model_last_day', n_clicks=0, className='update-button', style={'width': '100%', 'backgroundColor': '#4CAF50'}),
            ], style={'flex': '1', 'minWidth': '180px', 'alignSelf': 'end'}),
            html.Div([
                html.Div(id='update_model_status', style={'color': '#4CAF50', 'whiteSpace': 'pre-wrap'})
            ], style={'flex': '1', 'minWidth': '160px', 'alignSelf': 'end'})
        ], style={'display': 'grid', 'gridTemplateColumns': '1fr 200px', 'gap': '10px', 'padding': '0 20px 10px 20px', 'backgroundColor': '#1E1E1E', 'borderRadius': '8px'}),

        # Container for displaying metrics and graph side by side
        html.Div([
            # Container for metrics (2/3 of the width)
            html.Div([
                html.Div(id='model_metrics', style={
                    'color': '#4CAF50',
                    'fontSize': '16px',
                    'padding': '20px',
                    'backgroundColor': '#2E2E2E',
                    'borderRadius': '8px',
                    'marginBottom': '20px'
                })
            ], style={
                'flex': '2',  # 2/3 of the width
                'padding': '20px'
            }),

            # Graph for training and validation accuracy (1/3 of the width)
            html.Div([
                dcc.Graph(
                    id='accuracy_graph',
                    config={'scrollZoom': False},
                    style={'height': '40vh'}
                )
            ], style={
                'flex': '1',  # 1/3 of the width
                'padding': '10px',
                'backgroundColor': '#2E2E2E',
                'borderRadius': '8px',
                'marginBottom': '10px'
            })
        ], style={
            'display': 'grid',
            'gridTemplateColumns': '2fr 1fr',
            'width': '100%',
            'backgroundColor': '#1E1E1E',
            'borderRadius': '8px',
            'marginBottom': '10px',
            'gap': '10px',
            'alignItems': 'start'
        })
    ], style={
        'width': '100%',
        'backgroundColor': '#1E1E1E',
        'borderRadius': '8px',
        'marginBottom': '10px'
    }) 