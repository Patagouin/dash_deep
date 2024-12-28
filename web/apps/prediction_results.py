from dash import dcc, html

def get_results_layout():
    return html.Div([
        html.H4('Résultats du Modèle', style={
            'marginBottom': '0px',
            'padding': '20px',
            'color': '#FF8C00'
        }),

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
                    style={'height': '50vh'}
                )
            ], style={
                'flex': '1',  # 1/3 of the width
                'padding': '20px',
                'backgroundColor': '#2E2E2E',
                'borderRadius': '8px',
                'marginBottom': '20px'
            })
        ], style={
            'display': 'flex',  # Flexbox container
            'flexDirection': 'row',  # Horizontal layout
            'width': '100%',
            'backgroundColor': '#1E1E1E',
            'borderRadius': '8px',
            'marginBottom': '20px'
        })
    ], style={
        'width': '100%',
        'backgroundColor': '#1E1E1E',
        'borderRadius': '8px',
        'marginBottom': '20px'
    }) 