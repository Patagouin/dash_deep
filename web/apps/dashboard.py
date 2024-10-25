#dashboard.py
from dash import dcc, html

layout = html.Div([
    html.H3('Dashboard'),
    
    # Contenu du dashboard (à définir selon vos besoins)
    html.Div([
        html.P("Welcome to the dashboard")
    ], style={'margin': '20px'}),

    # Navigation standardisée
    html.Div([
        html.Hr(style={
            'width': '50%',
            'margin': '20px auto',
            'borderTop': '1px solid #666'
        }),
        html.Div([
            dcc.Link('Prediction', href='/prediction', style={'color': '#4CAF50', 'textDecoration': 'none'}),
            html.Span(' | ', style={'margin': '0 10px', 'color': '#666'}),
            dcc.Link('Update', href='/update', style={'color': '#4CAF50', 'textDecoration': 'none'}),
            html.Span(' | ', style={'margin': '0 10px', 'color': '#666'}),
            dcc.Link('Config', href='/config', style={'color': '#4CAF50', 'textDecoration': 'none'})
        ], style={'textAlign': 'center'})
    ], style={
        'width': '100%',
        'textAlign': 'center',
        'backgroundColor': 'black',
        'padding': '20px 0',
        'color': 'white'
    })
], style={'backgroundColor': 'black', 'minHeight': '100vh'})
